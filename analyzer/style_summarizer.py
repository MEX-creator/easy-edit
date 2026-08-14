"""Turns numeric StyleAnalysis into the human-readable llm_style_summary.

The LLM only ever sees a compact text digest (never raw video) and only needs
to produce one short JSON object. Everything numeric stays deterministic; the
LLM adds only the "vibe" layer. With no LLM configured, a heuristic summary is
used instead.
"""

from __future__ import annotations

from typing import Optional

from core.analysis import StyleAnalysis

MAX_CUTS_IN_DIGEST = 200
MAX_OVERLAYS_IN_DIGEST = 30


def build_compact_context(analysis: StyleAnalysis) -> str:
    """Serialize the analysis into a compact, cheap-to-tokenize digest."""
    lines: list[str] = []
    lines.append(f"video: {analysis.video_path}")
    lines.append(f"duration_sec: {analysis.duration_sec:.1f}, fps: {analysis.fps:.2f}")
    lines.append(f"shots: {len(analysis.shots)}, avg_shot_duration_sec: {analysis.avg_shot_duration_sec:.2f}")
    lines.append(f"cuts_per_10s: {analysis.cuts_per_10s:.1f}")

    cuts = [s.start_sec for s in analysis.shots[1:]]
    if len(cuts) > MAX_CUTS_IN_DIGEST:
        cuts = cuts[:MAX_CUTS_IN_DIGEST]
    lines.append("cut_timestamps_sec: [" + ", ".join(f"{c:.2f}" for c in cuts) + "]")

    kinds: dict[str, int] = {}
    for tr in analysis.transitions:
        kinds[tr.kind] = kinds.get(tr.kind, 0) + 1
    lines.append("transition_kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    if analysis.text_overlays:
        lines.append("on_screen_text:")
        for o in analysis.text_overlays[:MAX_OVERLAYS_IN_DIGEST]:
            lines.append(
                f"  - {o.text!r} start={o.start_sec:.1f}s dur={o.duration:.1f}s "
                f"center=({o.cx:.2f},{o.cy:.2f}) rel_h={o.rel_height:.2f} conf={o.confidence:.2f}"
            )
    else:
        lines.append("on_screen_text: none detected")

    audio = analysis.audio
    if audio and audio.beats:
        lines.append(f"music: {len(audio.beats)} beats, tempo_bpm={audio.tempo_bpm:.1f}" if audio.tempo_bpm else f"music: {len(audio.beats)} beats")
        lines.append(f"beat_synced_fraction: {analysis.beat_synced_fraction:.2f}")
        lines.append(f"energy_shape: {analysis.energy_shape()}")
    else:
        lines.append("music: no audio detected")

    return "\n".join(lines)


def heuristic_summary(analysis: StyleAnalysis) -> str:
    """Deterministic summary used when no LLM is available."""
    n_shots = len(analysis.shots)
    avg = analysis.avg_shot_duration_sec
    pace = "fast and punchy" if avg < 1.5 else ("moderate" if avg < 4.0 else "slow and cinematic")
    cuts = analysis.cuts_per_10s
    kinds = [t.kind for t in analysis.transitions if t.kind not in ("unknown", "hard_cut")]
    trans_text = ", ".join(sorted(set(kinds))[:3]) if kinds else "hard cuts throughout"
    beat = analysis.beat_synced_fraction
    beat_text = "cuts land on the beat" if beat >= 0.5 else "cuts are not beat-synced"
    overlays = len(analysis.text_overlays)
    text_text = f"{overlays} on-screen text elements detected" if overlays else "no on-screen text detected"
    return (
        f"{pace.capitalize()} {n_shots}-shot edit ({avg:.1f}s average shot, {cuts:.1f} cuts per 10s) "
        f"using {trans_text}; {text_text}; {beat_text}; "
        f"energy {analysis.energy_shape().replace('_', ' ')}."
    )


def summarize_style(analysis: StyleAnalysis, llm) -> str:
    """Produce llm_style_summary via the LLM, falling back to heuristics."""
    if llm is None:
        return heuristic_summary(analysis)
    context = build_compact_context(analysis)
    prompt = (
        "You are an expert video editing analyst. Below is a compact, numeric "
        "digest of a reference video's editing (cut timestamps, transition "
        "kinds, on-screen text, music beats). Summarize its EDITING STYLE in "
        "2-3 sentences: pacing, transition usage, text/caption style, whether "
        "cuts are beat-synced, and overall vibe tags (e.g. 'fast hookline "
        "reel', 'cinematic vlog', 'ASMR'). Return ONLY a JSON object of the "
        "form {\"summary\": \"<your 2-3 sentence summary>\"}.\n\n"
        f"{context}"
    )
    data = llm.complete_json(
        prompt,
        system="You summarize video editing styles concisely.",
        schema_description='{"summary": "..."}',
    )
    summary = str((data or {}).get("summary") or "").strip()
    return summary if summary else heuristic_summary(analysis)
