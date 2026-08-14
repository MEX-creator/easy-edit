"""Offline demo artifacts — no media files needed.

`demo_files()` writes a sample style template and a sample EDL so the whole
CLI surface (templates list/show, apply, build --plan) can be exercised
end-to-end without installing any heavy dependency or owning any footage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .edl import AudioItem, CaptionItem, Edl, TextOverlayItem, TimelineItem
from .template import StyleTemplate

SAMPLE_TEMPLATE = {
    "template_name": "fast-hookline-reel",
    "source": "reference_video.mp4",
    "created_at": "2026-01-01T00:00:00Z",
    "pacing": {
        "avg_shot_duration_sec": 1.2,
        "cut_style": "hard_cut",
        "beat_synced": True,
        "cuts_per_10s": 8,
    },
    "transitions": [{"type": "whip_pan", "frequency": "every_3rd_cut"}],
    "text_overlays": {
        "style": "bold_center_pop_in",
        "font_weight": "heavy",
        "position": "center",
        "avg_words_per_overlay": 4,
        "appears_on_beat": True,
    },
    "captions": {
        "present": True,
        "style": "karaoke_word_highlight",
        "position": "lower_third",
    },
    "music_sync": {
        "cuts_aligned_to_beats": True,
        "energy_curve": "build_to_drop",
    },
    "llm_style_summary": (
        "Fast punchy hookline-style reel edit, cuts land on beat, bold pop-in "
        "text for key phrases, whip-pan transitions used sparingly for emphasis "
        "on payoff moments."
    ),
}


def sample_template() -> StyleTemplate:
    return StyleTemplate.from_dict(SAMPLE_TEMPLATE)


def build_demo_edl() -> Edl:
    """A demo EDL referencing placeholder footage files (they need not exist)."""
    items = [
        TimelineItem(
            id="v1", source_path="footage_a.mp4", track=1,
            in_sec=0.0, out_sec=2.4, timeline_start_sec=0.0, duration_sec=2.4,
        ),
        TimelineItem(
            id="v2", source_path="footage_b.mp4", track=1,
            in_sec=0.0, out_sec=1.2, timeline_start_sec=2.4, duration_sec=1.2,
            transition_after="whip_pan", transition_duration_sec=0.6,
        ),
        TimelineItem(
            id="v3", source_path="footage_a.mp4", track=1,
            in_sec=2.4, out_sec=4.8, timeline_start_sec=3.6, duration_sec=2.4,
        ),
        TimelineItem(
            id="v4", source_path="footage_c.mp4", track=1,
            in_sec=0.0, out_sec=2.4, timeline_start_sec=6.0, duration_sec=2.4,
            transition_after="crossfade", transition_duration_sec=0.4,
        ),
    ]
    overlays = [
        TextOverlayItem(
            id="t1", text="The hook", timeline_start_sec=0.2, duration_sec=1.4,
            cx=0.5, cy=0.35, font_size_rel=0.07, style="bold_center_pop_in",
        ),
        TextOverlayItem(
            id="t2", text="Watch this", timeline_start_sec=3.6, duration_sec=1.4,
            cx=0.5, cy=0.35, font_size_rel=0.06, style="bold_center_pop_in",
        ),
    ]
    captions = [
        CaptionItem(
            id="c1", text="The hook.", timeline_start_sec=0.0, duration_sec=2.4,
            cx=0.5, cy=0.86,
        ),
        CaptionItem(
            id="c2", text="Watch this.", timeline_start_sec=2.4, duration_sec=1.2,
            cx=0.5, cy=0.86,
        ),
    ]
    return Edl(
        name="demo-edit",
        template_source="fast-hookline-reel",
        created_at="2026-01-01T00:00:00Z",
        fps=30.0,
        items=items,
        text_overlays=overlays,
        captions=captions,
        audio=[AudioItem(path="music_track.mp3", timeline_start_sec=0.0, track="A2")],
        notes=["Demo EDL - replace placeholder footage paths with real files."],
    )


def demo_files(out_dir: str | Path) -> Tuple[Path, Path]:
    """Write sample template + demo EDL into out_dir; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tpl_path = out / "sample_fast_hookline_reel.json"
    tpl_path.write_text(sample_template().to_json() + "\n", encoding="utf-8")
    edl_path = out / "demo_edl.json"
    edl_path.write_text(build_demo_edl().to_json() + "\n", encoding="utf-8")
    return tpl_path, edl_path
