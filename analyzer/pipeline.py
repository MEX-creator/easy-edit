"""Orchestrates reference-video style analysis and user-footage content analysis.

Two deliberately distinct entry points (they extract different things):

  * analyze_style()   — reference video → StyleTemplate
                        (WHAT the edit looks like: pacing, transitions, text,
                        music sync). The reusable, saveable artifact.
  * analyze_content() — the user's own footage → ClipAnalysis
                        (WHAT the footage contains: duration, scene changes,
                        motion, audio peaks). Feeds the applier.

Missing optional dependencies degrade gracefully: each stage reports a warning
and is skipped, except shot detection which is the backbone of style analysis.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

from core.analysis import (
    AudioAnalysis,
    ClipAnalysis,
    StyleAnalysis,
    TextOverlay,
)
from core.config import Settings
from core.errors import AnalysisError, MissingDependency
from core.template import (
    CaptionStyle,
    MusicSyncProfile,
    PacingProfile,
    StyleTemplate,
    TextOverlayStyle,
    TransitionProfile,
    now_iso,
)
from llm.providers import get_llm

from . import audio_beat_detect, ocr_overlay_detect, shot_detect, transition_detect
from .style_summarizer import heuristic_summary, summarize_style

Progress = Optional[Callable[[str], None]]


def _report(progress: Progress, msg: str) -> None:
    if progress:
        progress(msg)


# ---------------------------------------------------------------------------
# Reference video → style template
# ---------------------------------------------------------------------------


def analyze_style(
    video_path: str,
    settings: Optional[Settings] = None,
    do_ocr: bool = True,
    do_audio: bool = True,
    do_llm: bool = True,
    progress: Progress = None,
) -> StyleTemplate:
    settings = settings or Settings.from_env()
    _report(progress, f"Analyzing {video_path}")

    meta = shot_detect.probe_video(video_path)
    if meta is None:
        raise MissingDependency("Video analysis", "shots")
    duration = meta.get("duration_sec") or 0.0
    fps = meta.get("fps") or 30.0
    _report(progress, f"Video: {duration:.1f}s @ {fps:.2f} fps")

    shots = shot_detect.detect_shots(video_path)
    if not shots:
        raise AnalysisError(f"no shots detected in {video_path}")
    _report(progress, f"Detected {len(shots)} shots")

    transitions = []
    try:
        transitions = transition_detect.classify_transitions(video_path, shots)
        _report(progress, f"Classified {len(transitions)} cut transitions")
    except MissingDependency as exc:
        _report(progress, f"warning: {exc}")
    except Exception as exc:
        _report(progress, f"warning: transition classification failed: {exc}")

    overlays: List[TextOverlay] = []
    if do_ocr:
        try:
            overlays = ocr_overlay_detect.detect_text_overlays(
                video_path,
                languages=settings.ocr_languages,
                min_confidence=settings.ocr_min_confidence,
                progress=progress,
            )
            _report(progress, f"OCR detected {len(overlays)} on-screen text segments")
        except MissingDependency as exc:
            _report(progress, f"warning: {exc}")
        except Exception as exc:
            _report(progress, f"warning: OCR failed: {exc}")

    audio: Optional[AudioAnalysis] = None
    if do_audio:
        try:
            audio = audio_beat_detect.analyze_audio(video_path)
            n_beats = len(audio.beats) if audio else 0
            _report(progress, f"Audio analysis: {n_beats} beats")
        except MissingDependency as exc:
            _report(progress, f"warning: {exc}")
        except Exception as exc:
            _report(progress, f"warning: audio analysis skipped: {exc}")

    analysis = StyleAnalysis(
        video_path=video_path,
        duration_sec=duration,
        fps=fps,
        shots=shots,
        transitions=transitions,
        text_overlays=overlays,
        audio=audio,
    )
    return _build_template(analysis, settings, do_llm, progress)


def _build_template(
    analysis: StyleAnalysis,
    settings: Settings,
    do_llm: bool,
    progress: Progress,
) -> StyleTemplate:
    analysis.require_shots()
    beat_frac = analysis.beat_synced_fraction
    pacing = PacingProfile(
        avg_shot_duration_sec=round(analysis.avg_shot_duration_sec, 2),
        cut_style=analysis.dominant_transition_kind(),
        beat_synced=beat_frac >= 0.5,
        cuts_per_10s=round(analysis.cuts_per_10s, 1),
    )

    text_style, caption_style = _classify_text_styles(analysis.text_overlays)
    if text_style.appears_on_beat:
        text_style.appears_on_beat = beat_frac >= 0.3

    summary = ""
    if do_llm:
        try:
            llm = get_llm(settings)
            if llm is not None:
                _report(progress, f"LLM style summary via {llm.model}")
                summary = summarize_style(analysis, llm)
        except Exception as exc:
            _report(progress, f"warning: LLM style summary skipped: {exc}")
    if not summary:
        summary = heuristic_summary(analysis)

    return StyleTemplate(
        template_name=Path(analysis.video_path).stem,
        source=analysis.video_path,
        created_at=now_iso(),
        pacing=pacing,
        transitions=_transition_profiles(analysis.transitions),
        text_overlays=text_style,
        captions=caption_style,
        music_sync=MusicSyncProfile(
            cuts_aligned_to_beats=pacing.beat_synced,
            energy_curve=analysis.energy_shape(),
        ),
        llm_style_summary=summary,
    )


def _transition_profiles(transitions) -> List[TransitionProfile]:
    """Collapse observed transition kinds into template entries with frequency."""
    kinds = Counter(t.kind for t in transitions)
    interesting = [(k, n) for k, n in kinds.items() if k not in ("hard_cut", "unknown")]
    if not interesting:
        return []
    total = max(1, len(transitions))
    profiles = []
    for kind, n in sorted(interesting, key=lambda kv: -kv[1])[:2]:
        frac = n / total
        if frac >= 0.5:
            frequency = "every_other_cut"
        elif frac >= 0.25:
            frequency = "every_3rd_cut"
        else:
            frequency = "sparingly"
        profiles.append(TransitionProfile(type=kind, frequency=frequency))
    return profiles


def _classify_text_styles(overlays: List[TextOverlay]):
    """Split OCR segments into captions (bottom third, frequent) vs. styled text."""
    captions = [
        o
        for o in overlays
        if o.cy > 0.6 or (o.rel_height < 0.07 and o.duration > 0.8)
    ]
    styled = [o for o in overlays if o not in captions]

    if styled:
        mean_cx = sum(o.cx for o in styled) / len(styled)
        mean_cy = sum(o.cy for o in styled) / len(styled)
        mean_h = sum(o.rel_height for o in styled) / len(styled)
        avg_words = sum(len(o.text.split()) for o in styled) / len(styled)
        centerish = abs(mean_cx - 0.5) < 0.2 and mean_cy < 0.55
        if centerish:
            style, position = "bold_center_pop_in", "center"
        elif mean_cy < 0.45:
            style, position = "upper_third_pop_in", "upper_third"
        else:
            style, position = "pop_in", "lower_third"
        text_style = TextOverlayStyle(
            style=style,
            font_weight="heavy" if mean_h > 0.06 else "regular",
            position=position,
            avg_words_per_overlay=round(avg_words, 1),
            appears_on_beat=True,
        )
    else:
        text_style = TextOverlayStyle(
            style="none",
            font_weight="regular",
            position="center",
            avg_words_per_overlay=0.0,
            appears_on_beat=False,
        )

    caption_style = CaptionStyle(
        present=bool(captions),
        style="sentence_lower_third" if captions else "none",
        position="lower_third" if captions else "center",
    )
    return text_style, caption_style


# ---------------------------------------------------------------------------
# User footage → content analysis (feeds the applier)
# ---------------------------------------------------------------------------


def analyze_content(
    video_path: str,
    settings: Optional[Settings] = None,
    progress: Progress = None,
) -> ClipAnalysis:
    settings = settings or Settings.from_env()
    meta = shot_detect.probe_video(video_path)
    if meta is None:
        raise MissingDependency("Video analysis", "shots")
    duration = meta.get("duration_sec") or 0.0
    fps = meta.get("fps") or 30.0

    scene_changes: List[float] = []
    try:
        shots = shot_detect.detect_shots(video_path)
        scene_changes = [s.start_sec for s in shots[1:]]
    except MissingDependency as exc:
        _report(progress, f"warning: {exc}")

    motion: List[tuple] = []
    audio_peaks: List[float] = []
    has_audio = False
    try:
        import cv2  # noqa: F401
        from . import motion_detect

        motion = motion_detect.sample_motion(video_path, duration)
    except MissingDependency:
        pass
    except Exception as exc:
        _report(progress, f"warning: motion sampling skipped: {exc}")
    try:
        audio = audio_beat_detect.analyze_audio(video_path)
        if audio is not None:
            has_audio = True
            audio_peaks = [b.time_sec for b in audio.beats]
    except Exception:
        pass

    return ClipAnalysis(
        path=video_path,
        duration_sec=duration,
        fps=fps,
        scene_changes=[round(t, 3) for t in scene_changes],
        motion=motion,
        audio_peaks=[round(t, 3) for t in audio_peaks],
        has_audio=has_audio,
    )
