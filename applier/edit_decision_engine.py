"""Style template + user footage → Edit Decision List (EDL).

plan_edit() is the single entry point. Creative/judgment decisions (clip
selection, in/out points, caption phrasing, overlay placement) may go through
an optional LLM; pacing, the cut grid, beat-sync and transition cadence always
come from the template — the template's "DNA" is respected even when the LLM
proposes something else (its output is normalized against the template).

When no LLM is available (no key / offline / Ollama not running), a fully
deterministic heuristic planner is used: beat-grid or uniform slicing of the
footage at the template's pacing, sentence-level captions from the script.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.analysis import ClipSource
from core.edl import AudioItem, CaptionItem, Edl, TextOverlayItem, TimelineItem
from core.errors import EdlError
from core.template import StyleTemplate, now_iso

_TRANSITION_DURATIONS = {
    "crossfade": 0.4,
    "dip_to_black": 0.4,
    "whip_pan": 0.6,
    "zoom_cut": 0.6,
}


@dataclass
class PlanOptions:
    fps: float = 30.0
    min_shot_sec: float = 0.4
    overlay_min_duration_sec: float = 0.8
    max_shot_multiplier: float = 1.5  # how far an LLM proposal may stretch the template's shot length
    # Pacing mode: cuts may slide up to this fraction of the template's shot
    # duration away from the ideal pacing grid to land on a real scene change
    # in the footage. Set to 0.0 for a pure pacing grid (no scene preference).
    pacing_snap_band: float = 0.4


def plan_edit(
    template: StyleTemplate | Dict[str, Any],
    clips: List[ClipSource],
    script: str = "",
    music: Optional[str] = None,
    llm=None,
    options: Optional[PlanOptions] = None,
) -> Edl:
    """Plan an edit from a style template + raw clips. Returns an Edl."""
    opts = options or PlanOptions()
    tpl = template if isinstance(template, StyleTemplate) else StyleTemplate.from_dict(template)
    if not clips:
        raise EdlError("plan_edit needs at least one clip")
    for clip in clips:
        if not clip.path:
            raise EdlError("every clip needs a path")

    if llm is not None:
        try:
            return _plan_with_llm(tpl, clips, script, music, llm, opts)
        except Exception as exc:
            notes = [f"LLM planning failed ({exc}); fell back to the deterministic planner"]
            fallback = _plan_heuristically(tpl, clips, script, music, opts)
            fallback.notes = notes + fallback.notes
            return fallback
    return _plan_heuristically(tpl, clips, script, music, opts)


# ---------------------------------------------------------------------------
# Deterministic heuristic planner (offline / no-LLM path)
# ---------------------------------------------------------------------------


def _plan_heuristically(
    tpl: StyleTemplate,
    clips: List[ClipSource],
    script: str,
    music: Optional[str],
    opts: PlanOptions,
) -> Edl:
    notes: List[str] = []
    total = 0.0
    for clip in clips:
        d = clip.duration
        if not d or d <= 0:
            raise EdlError(
                f"no duration known for {clip.path}; run `editdna content` on it "
                "or pass --durations on the command line"
            )
        total += d

    target = tpl.pacing.avg_shot_duration_sec or 2.0
    beats: List[float] = []
    if tpl.music_sync.cuts_aligned_to_beats and music:
        try:
            from analyzer.audio_beat_detect import analyze_audio

            audio = analyze_audio(music)
            beats = [b.time_sec for b in (audio.beats if audio else [])]
            notes.append(f"cut grid snapped to {len(beats)} beats from {music}")
        except Exception as exc:
            notes.append(f"could not analyze music ({exc}); cuts use template pacing instead")

    beat_mode = any(0 < b < total for b in beats)
    segments = _plan_shots(clips, total, target, beats, opts)
    items = _build_items(clips, segments, opts)
    _apply_transitions(items, tpl)
    if not beat_mode and any(c.analysis and c.analysis.scene_changes for c in clips):
        notes.append(
            "cut points prefer real scene changes in your footage "
            "(template pacing preserved within the snap band)"
        )

    overlays, captions = _text_items(tpl, items, script, beats, opts)
    audio_items = []
    if music:
        audio_items.append(AudioItem(path=music, timeline_start_sec=0.0, track="A2"))
    if not script or not script.strip():
        notes.append("no script provided; no text overlays or captions were generated")

    edl = Edl(
        name=tpl.template_name or "untitled-edit",
        template_source=tpl.source or tpl.template_name,
        created_at=now_iso(),
        fps=round(opts.fps, 2),
        items=items,
        text_overlays=overlays,
        captions=captions,
        audio=audio_items,
        notes=notes,
    )
    return edl


def _plan_shots(
    clips: List[ClipSource],
    total: float,
    target: float,
    beats: List[float],
    opts: PlanOptions,
) -> List[Tuple[int, float, float]]:
    """Slice all clips into (clip_index, in_sec, out_sec) segments.

    Two modes:

    * Beat mode (template wants beat-synced cuts and beats are available):
      timeline cut times are LOCKED to the beat grid - that is the template's
      DNA - and source ranges are fitted continuously to those slots so every
      beat cut uses footage with no skipping.

    * Pacing mode (no beats): each clip is segmented greedily so cuts land on
      real scene changes in the footage whenever one sits near the template's
      ideal pacing position (within +/- pacing_snap_band of the shot target).
      Otherwise the cut falls back to the pacing grid itself, so average shot
      length still matches the template.
    """
    scene_changes: List[List[float]] = []
    for clip in clips:
        dur = clip.duration or 0.0
        changes = sorted(
            {
                round(t, 3)
                for t in (clip.analysis.scene_changes if clip.analysis else [])
                if 0 < t < dur
            }
        )
        scene_changes.append(changes)

    if any(0 < b < total for b in beats):
        grid: List[float] = []
        for b in sorted(set(round(x, 3) for x in beats if 0 < x < total)):
            if not grid or b - grid[-1] >= opts.min_shot_sec:
                grid.append(b)
        boundaries = [0.0] + grid + [total]
        return _fit_slots_to_boundaries(clips, boundaries, opts)
    return _greedy_scene_aware(clips, target, scene_changes, opts)


def _fit_slots_to_boundaries(
    clips: List[ClipSource],
    boundaries: List[float],
    opts: PlanOptions,
) -> List[Tuple[int, float, float]]:
    """Fill timeline-space slots with continuous source ranges.

    Continues into the next clip when a clip runs out mid-slot; raises when
    the footage ends before the grid does.
    """
    segments: List[Tuple[int, float, float]] = []
    src_cursor = [0.0] * len(clips)
    ci = 0
    for a, b in zip(boundaries, boundaries[1:]):
        remaining = b - a
        while remaining > 1e-6:
            while ci < len(clips) and src_cursor[ci] >= (clips[ci].duration or 0.0) - 1e-6:
                ci += 1
            if ci >= len(clips):
                raise EdlError(
                    "footage is shorter than the planned edit - trim the script, "
                    "add clips, or loosen the template's pacing"
                )
            avail = (clips[ci].duration or 0.0) - src_cursor[ci]
            take = min(remaining, avail)
            segments.append((ci, round(src_cursor[ci], 3), round(src_cursor[ci] + take, 3)))
            src_cursor[ci] += take
            remaining -= take
    return segments


def _greedy_scene_aware(
    clips: List[ClipSource],
    target: float,
    scene_changes: List[List[float]],
    opts: PlanOptions,
) -> List[Tuple[int, float, float]]:
    """Segment each clip greedily, snapping cuts to real scene changes.

    For each shot the ideal end is (start + target). If a scene change in the
    footage falls within [start + target*(1-band), start + target*(1+band)]
    the cut moves to the scene change closest to the ideal point; otherwise
    the pacing grid holds. Shots never dip below min_shot_sec (except a final
    clip tail).
    """
    segments: List[Tuple[int, float, float]] = []
    for ci, clip in enumerate(clips):
        dur = clip.duration or 0.0
        changes = scene_changes[ci]
        cursor = 0.0
        while cursor < dur - 1e-6:
            ideal = cursor + target
            if ideal >= dur - 1e-6:
                segments.append((ci, round(cursor, 3), round(dur, 3)))
                break
            low = max(
                cursor + opts.min_shot_sec,
                cursor + target * (1.0 - opts.pacing_snap_band),
            )
            high = cursor + target * (1.0 + opts.pacing_snap_band)
            if low >= dur:
                segments.append((ci, round(cursor, 3), round(dur, 3)))
                break
            candidates = [c for c in changes if low <= c <= high]
            boundary = ideal
            if candidates:
                boundary = min(candidates, key=lambda c: abs(c - ideal))
            if boundary - cursor < opts.min_shot_sec:
                boundary = min(cursor + opts.min_shot_sec, dur)
            segments.append((ci, round(cursor, 3), round(boundary, 3)))
            cursor = boundary
    return segments


def _build_items(
    clips: List[ClipSource],
    segments: List[Tuple[int, float, float]],
    opts: PlanOptions,
) -> List[TimelineItem]:
    items: List[TimelineItem] = []
    t = 0.0
    for n, (ci, s, e) in enumerate(segments, start=1):
        items.append(
            TimelineItem(
                id=f"v{n}",
                source_path=clips[ci].path,
                track=1,
                in_sec=s,
                out_sec=e,
                timeline_start_sec=round(t, 3),
                duration_sec=round(e - s, 3),
            )
        )
        t += e - s
    return items


def _apply_transitions(items: List[TimelineItem], tpl: StyleTemplate) -> None:
    if len(items) < 2:
        return
    kinds = [tr.type for tr in tpl.transitions if tr.type and tr.type != "hard_cut"]
    if not kinds:
        return
    cadence = _transition_cadence(tpl)
    for i in range(len(items) - 1):
        if (i + 1) % cadence == 0:
            kind = kinds[(i // cadence) % len(kinds)]
            items[i].transition_after = kind
            items[i].transition_duration_sec = _TRANSITION_DURATIONS.get(kind, 0.4)


def _transition_cadence(tpl: StyleTemplate) -> int:
    text = " ".join(tr.frequency for tr in tpl.transitions).lower()
    m = re.search(r"every[_ -]?(\d+)", text)
    if m:
        return max(1, int(m.group(1)))
    if "always" in text or "every cut" in text:
        return 1
    if "sparing" in text or "rare" in text:
        return 4
    return 3


def _text_items(
    tpl: StyleTemplate,
    items: List[TimelineItem],
    script: str,
    beats: List[float],
    opts: PlanOptions,
) -> Tuple[List[TextOverlayItem], List[CaptionItem]]:
    overlays: List[TextOverlayItem] = []
    captions: List[CaptionItem] = []
    if not script or not script.strip():
        return overlays, captions

    sentences = _split_sentences(script)

    # ---- text overlays from script phrases, paced by the template ----
    overlay_style = (tpl.text_overlays.style or "").lower()
    if overlay_style not in ("", "none", "no"):
        phrases: List[str] = []
        for sentence in sentences:
            clauses = [c.strip() for c in re.split(r"[,;:]", sentence) if c.strip()]
            phrases.extend(clauses or [sentence.strip()])
        phrases = [p for p in phrases if p]
        if phrases:
            # one overlay per phrase, placed on the first shots (no repeats)
            for i in range(min(len(items), len(phrases))):
                item = items[i]
                phrase = phrases[i]
                start = item.timeline_start_sec
                if tpl.text_overlays.appears_on_beat and beats:
                    window_end = item.timeline_start_sec + item.duration_sec - 0.2
                    nb = next(
                        (b for b in beats if item.timeline_start_sec <= b < window_end),
                        None,
                    )
                    if nb is not None:
                        start = nb
                available = item.duration_sec - (start - item.timeline_start_sec)
                if available < 0.25:
                    continue
                words = len(phrase.split())
                dur = min(available, max(opts.overlay_min_duration_sec, 0.5 + 0.35 * words))
                cx, cy = _position_for(tpl.text_overlays.position)
                font = min(0.10, max(0.045, 0.05 + 0.004 * min(words, 8)))
                overlays.append(
                    TextOverlayItem(
                        id=f"t{i + 1}",
                        text=phrase,
                        timeline_start_sec=round(start, 3),
                        duration_sec=round(dur, 3),
                        cx=cx,
                        cy=cy,
                        font_size_rel=round(font, 3),
                        style=tpl.text_overlays.style,
                        track=2,
                    )
                )

    # ---- sentence-level captions, one per shot (no repeats) ----
    if tpl.captions.present and sentences:
        for i in range(min(len(items), len(sentences))):
            item = items[i]
            captions.append(
                CaptionItem(
                    id=f"c{i + 1}",
                    text=sentences[i],
                    timeline_start_sec=item.timeline_start_sec,
                    duration_sec=item.duration_sec,
                    cx=0.5,
                    cy=0.86,
                    style=tpl.captions.style or "sentence_lower_third",
                    track=2,
                )
            )
    return overlays, captions


def _position_for(position: str) -> Tuple[float, float]:
    return {
        "center": (0.5, 0.35),
        "upper_third": (0.5, 0.2),
        "lower_third": (0.5, 0.75),
        "left": (0.25, 0.5),
        "right": (0.75, 0.5),
    }.get(position.lower(), (0.5, 0.35))


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# LLM-assisted path
# ---------------------------------------------------------------------------


def _plan_with_llm(
    tpl: StyleTemplate,
    clips: List[ClipSource],
    script: str,
    music: Optional[str],
    llm,
    opts: PlanOptions,
) -> Edl:
    clips_info = []
    for clip in clips:
        a = clip.analysis
        clips_info.append(
            {
                "path": clip.path,
                "duration_sec": clip.duration,
                "scene_changes": (a.scene_changes[:60] if a else []),
                "avg_motion": round(a.avg_motion, 3) if a else 0.0,
                "has_audio": bool(a and a.has_audio),
            }
        )
    context = json.dumps(
        {
            "style_template": tpl.to_dict(),
            "clips": clips_info,
            "script": script or None,
            "music_track": music or None,
        },
        indent=2,
    )
    schema = (
        '{"items": [{"source_path": str, "in_sec": float, "out_sec": float}], '
        '"text_overlays": [{"text": str, "timeline_start_sec": float, "duration_sec": float}], '
        '"captions": [{"text": str, "timeline_start_sec": float, "duration_sec": float}], '
        '"notes": [str]}'
    )
    prompt = (
        "You are the creative editor inside an automatic editing pipeline. "
        "The style template below is sacred: keep its pacing (avg shot "
        "duration), beat-sync and transition cadence — do not invent new "
        "pacing. Your job is the judgment calls: which clip segments to use "
        "(in/out points), the order, where captions go, and where text "
        "overlays land. Only reference source paths from the clips list. "
        "Return ONLY valid JSON.\n\n"
        f"{context}\n\nExpected JSON shape:\n{schema}"
    )
    data = llm.complete_json(
        prompt,
        system="You are a video editor producing structured Edit Decision Lists.",
        schema_description="an Edit Decision List JSON object",
    )
    return _edl_from_llm(tpl, clips, data, opts)


def _edl_from_llm(
    tpl: StyleTemplate,
    clips: List[ClipSource],
    data: Dict[str, Any],
    opts: PlanOptions,
) -> Edl:
    if not isinstance(data, dict) or not isinstance(data.get("items"), list) or not data["items"]:
        raise EdlError("LLM returned no usable edit items")

    path_map = {Path(c.path).resolve(): c for c in clips}
    name_map = {Path(c.path).name: c for c in clips}

    def _find_clip(raw: Any) -> Optional[ClipSource]:
        if not raw:
            return None
        s = str(raw)
        clip = path_map.get(Path(s).resolve())
        if clip is None:
            clip = name_map.get(Path(s).name)
        if clip is None:
            clip = next((c for c in clips if Path(c.path).name == Path(s).name), None)
        return clip

    target = tpl.pacing.avg_shot_duration_sec or 2.0
    items: List[TimelineItem] = []
    t = 0.0
    for row in data["items"]:
        if not isinstance(row, dict):
            continue
        clip = _find_clip(row.get("source_path") or row.get("source"))
        if clip is None:
            continue
        clip_dur = clip.duration or target
        in_s = float(row.get("in_sec", 0.0) or 0.0)
        in_s = min(max(0.0, in_s), max(0.0, clip_dur - 0.1))
        proposed = float(row.get("duration_sec", 0.0) or 0.0) or target
        dur = min(proposed, max(opts.min_shot_sec, target * opts.max_shot_multiplier))
        dur = max(opts.min_shot_sec, dur)
        out_s = min(clip_dur, in_s + dur)
        items.append(
            TimelineItem(
                id=f"v{len(items) + 1}",
                source_path=clip.path,
                track=1,
                in_sec=round(in_s, 3),
                out_sec=round(out_s, 3),
                timeline_start_sec=round(t, 3),
                duration_sec=round(out_s - in_s, 3),
            )
        )
        t += out_s - in_s
    if not items:
        raise EdlError("LLM proposed no usable items")

    _apply_transitions(items, tpl)
    total = t

    overlays: List[TextOverlayItem] = []
    for i, row in enumerate(data.get("text_overlays") or []):
        text = str(row.get("text") or "").strip()
        if not text or not isinstance(row, dict):
            continue
        start = float(row.get("timeline_start_sec", 0.0) or 0.0)
        dur = max(0.5, float(row.get("duration_sec", 1.0) or 1.0))
        start = min(max(0.0, start), max(0.0, total - 0.2))
        dur = min(dur, max(0.2, total - start))
        overlays.append(
            TextOverlayItem(
                id=f"t{i + 1}",
                text=text,
                timeline_start_sec=round(start, 3),
                duration_sec=round(dur, 3),
                cx=float(row.get("cx", 0.5)),
                cy=float(row.get("cy", 0.35)),
                font_size_rel=float(row.get("font_size_rel", 0.06) or 0.06),
                style=tpl.text_overlays.style,
            )
        )

    captions: List[CaptionItem] = []
    for i, row in enumerate(data.get("captions") or []):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        start = float(row.get("timeline_start_sec", 0.0) or 0.0)
        dur = max(0.5, float(row.get("duration_sec", 1.0) or 1.0))
        start = min(max(0.0, start), max(0.0, total - 0.2))
        dur = min(dur, max(0.2, total - start))
        captions.append(
            CaptionItem(
                id=f"c{i + 1}",
                text=text,
                timeline_start_sec=round(start, 3),
                duration_sec=round(dur, 3),
                cx=float(row.get("cx", 0.5)),
                cy=float(row.get("cy", 0.86)),
                style=tpl.captions.style or "sentence_lower_third",
            )
        )

    audio_items = []
    if music:
        audio_items.append(AudioItem(path=music, timeline_start_sec=0.0, track="A2"))

    notes = [str(n) for n in (data.get("notes") or []) if n]
    return Edl(
        name=tpl.template_name or "untitled-edit",
        template_source=tpl.source or tpl.template_name,
        created_at=now_iso(),
        fps=round(opts.fps, 2),
        items=items,
        text_overlays=overlays,
        captions=captions,
        audio=audio_items,
        notes=notes,
    )
