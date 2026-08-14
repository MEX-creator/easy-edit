"""The style template: the reusable, human-readable artifact.

Schema version 1, kept intentionally close to the project spec's example JSON
so templates can be written by hand, diffed, and shared. Unknown keys in input
are ignored (forward-compatible); missing *required* keys raise TemplateError.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import TemplateError

SCHEMA_VERSION = 1


@dataclass
class PacingProfile:
    avg_shot_duration_sec: float = 1.2
    cut_style: str = "hard_cut"
    beat_synced: bool = True
    cuts_per_10s: float = 8.0


@dataclass
class TransitionProfile:
    type: str = "whip_pan"
    frequency: str = "every_3rd_cut"


@dataclass
class TextOverlayStyle:
    style: str = "bold_center_pop_in"
    font_weight: str = "heavy"
    position: str = "center"
    avg_words_per_overlay: float = 4.0
    appears_on_beat: bool = True


@dataclass
class CaptionStyle:
    present: bool = True
    style: str = "sentence_lower_third"
    position: str = "lower_third"


@dataclass
class MusicSyncProfile:
    cuts_aligned_to_beats: bool = True
    energy_curve: str = "build_to_drop"


@dataclass
class StyleTemplate:
    template_name: str
    source: str = ""
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION
    pacing: PacingProfile = field(default_factory=PacingProfile)
    transitions: List[TransitionProfile] = field(default_factory=list)
    text_overlays: TextOverlayStyle = field(default_factory=TextOverlayStyle)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    music_sync: MusicSyncProfile = field(default_factory=MusicSyncProfile)
    llm_style_summary: str = ""

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "schema_version": self.schema_version,
            "source": self.source,
            "created_at": self.created_at,
            "pacing": {
                "avg_shot_duration_sec": self.pacing.avg_shot_duration_sec,
                "cut_style": self.pacing.cut_style,
                "beat_synced": self.pacing.beat_synced,
                "cuts_per_10s": self.pacing.cuts_per_10s,
            },
            "transitions": [
                {"type": t.type, "frequency": t.frequency} for t in self.transitions
            ],
            "text_overlays": {
                "style": self.text_overlays.style,
                "font_weight": self.text_overlays.font_weight,
                "position": self.text_overlays.position,
                "avg_words_per_overlay": self.text_overlays.avg_words_per_overlay,
                "appears_on_beat": self.text_overlays.appears_on_beat,
            },
            "captions": {
                "present": self.captions.present,
                "style": self.captions.style,
                "position": self.captions.position,
            },
            "music_sync": {
                "cuts_aligned_to_beats": self.music_sync.cuts_aligned_to_beats,
                "energy_curve": self.music_sync.energy_curve,
            },
            "llm_style_summary": self.llm_style_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleTemplate":
        if not isinstance(data, dict):
            raise TemplateError("style template must be a JSON object")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise TemplateError(
                f"unsupported template schema_version {version}; "
                f"this build supports version {SCHEMA_VERSION}"
            )
        name = str(data.get("template_name") or "").strip()
        if not name:
            raise TemplateError("style template is missing required key 'template_name'")

        pacing = data.get("pacing") or {}
        avg_shot = pacing.get("avg_shot_duration_sec", 2.0)
        try:
            avg_shot = float(avg_shot)
        except (TypeError, ValueError):
            raise TemplateError("pacing.avg_shot_duration_sec must be a number")
        if avg_shot <= 0:
            raise TemplateError("pacing.avg_shot_duration_sec must be positive")

        transitions = []
        for item in data.get("transitions") or []:
            if not isinstance(item, dict):
                continue
            transitions.append(
                TransitionProfile(
                    type=str(item.get("type") or "hard_cut"),
                    frequency=str(item.get("frequency") or ""),
                )
            )

        text = data.get("text_overlays") or {}
        captions = data.get("captions") or {}
        music = data.get("music_sync") or {}

        return cls(
            template_name=name,
            source=str(data.get("source") or ""),
            created_at=str(data.get("created_at") or ""),
            schema_version=version,
            pacing=PacingProfile(
                avg_shot_duration_sec=avg_shot,
                cut_style=str(pacing.get("cut_style") or "hard_cut"),
                beat_synced=bool(pacing.get("beat_synced", False)),
                cuts_per_10s=float(pacing.get("cuts_per_10s", 0.0) or 0.0),
            ),
            transitions=transitions,
            text_overlays=TextOverlayStyle(
                style=str(text.get("style") or "pop_in"),
                font_weight=str(text.get("font_weight") or "regular"),
                position=str(text.get("position") or "center"),
                avg_words_per_overlay=float(text.get("avg_words_per_overlay", 4.0) or 0.0),
                appears_on_beat=bool(text.get("appears_on_beat", False)),
            ),
            captions=CaptionStyle(
                present=bool(captions.get("present", False)),
                style=str(captions.get("style") or "sentence_lower_third"),
                position=str(captions.get("position") or "lower_third"),
            ),
            music_sync=MusicSyncProfile(
                cuts_aligned_to_beats=bool(music.get("cuts_aligned_to_beats", False)),
                energy_curve=str(music.get("energy_curve") or "unknown"),
            ),
            llm_style_summary=str(data.get("llm_style_summary") or ""),
        )

    @classmethod
    def from_json(cls, text: str) -> "StyleTemplate":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TemplateError(f"template is not valid JSON: {exc}")
        return cls.from_dict(data)

    # ---- file helpers ----

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "StyleTemplate":
        p = Path(path)
        if not p.is_file():
            raise TemplateError(f"template file not found: {p}")
        return cls.from_json(p.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
