"""Edit Decision List (EDL): the intermediate format between the applier and
the DaVinci Resolve driver.

A plain JSON file, versioned and human-readable, so a user can hand-tune it
before driving Resolve (or have the resolve_driver build it directly).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import EdlError

EDL_VERSION = 1


@dataclass
class TimelineItem:
    id: str
    source_path: str
    track: int = 1
    in_sec: float = 0.0  # source in point
    out_sec: float = 0.0  # source out point
    timeline_start_sec: float = 0.0
    duration_sec: float = 0.0
    transition_after: Optional[str] = None  # kind of transition to the NEXT item
    transition_duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "track": self.track,
            "in_sec": self.in_sec,
            "out_sec": self.out_sec,
            "timeline_start_sec": self.timeline_start_sec,
            "duration_sec": self.duration_sec,
            "transition_after": self.transition_after,
            "transition_duration_sec": self.transition_duration_sec,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineItem":
        return cls(
            id=str(data.get("id") or ""),
            source_path=str(data.get("source_path") or data.get("source") or ""),
            track=int(data.get("track", 1) or 1),
            in_sec=float(data.get("in_sec", 0.0) or 0.0),
            out_sec=float(data.get("out_sec", 0.0) or 0.0),
            timeline_start_sec=float(data.get("timeline_start_sec", 0.0) or 0.0),
            duration_sec=float(data.get("duration_sec", 0.0) or 0.0),
            transition_after=data.get("transition_after"),
            transition_duration_sec=float(data.get("transition_duration_sec", 0.0) or 0.0),
        )


@dataclass
class TextOverlayItem:
    id: str
    text: str
    timeline_start_sec: float
    duration_sec: float
    cx: float = 0.5
    cy: float = 0.35
    font_size_rel: float = 0.06  # relative to frame height
    style: str = "pop_in"
    track: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "timeline_start_sec": self.timeline_start_sec,
            "duration_sec": self.duration_sec,
            "cx": self.cx,
            "cy": self.cy,
            "font_size_rel": self.font_size_rel,
            "style": self.style,
            "track": self.track,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextOverlayItem":
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            timeline_start_sec=float(data.get("timeline_start_sec", 0.0) or 0.0),
            duration_sec=float(data.get("duration_sec", 1.0) or 1.0),
            cx=float(data.get("cx", 0.5)),
            cy=float(data.get("cy", 0.35)),
            font_size_rel=float(data.get("font_size_rel", 0.06) or 0.06),
            style=str(data.get("style") or "pop_in"),
            track=int(data.get("track", 2) or 2),
        )


@dataclass
class CaptionItem:
    id: str
    text: str
    timeline_start_sec: float
    duration_sec: float
    cx: float = 0.5
    cy: float = 0.86
    style: str = "sentence_lower_third"
    track: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "timeline_start_sec": self.timeline_start_sec,
            "duration_sec": self.duration_sec,
            "cx": self.cx,
            "cy": self.cy,
            "style": self.style,
            "track": self.track,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptionItem":
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            timeline_start_sec=float(data.get("timeline_start_sec", 0.0) or 0.0),
            duration_sec=float(data.get("duration_sec", 1.0) or 1.0),
            cx=float(data.get("cx", 0.5)),
            cy=float(data.get("cy", 0.86)),
            style=str(data.get("style") or "sentence_lower_third"),
            track=int(data.get("track", 2) or 2),
        )


@dataclass
class AudioItem:
    path: str
    timeline_start_sec: float = 0.0
    track: str = "A2"
    volume_db: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "timeline_start_sec": self.timeline_start_sec,
            "track": self.track,
            "volume_db": self.volume_db,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioItem":
        return cls(
            path=str(data.get("path") or ""),
            timeline_start_sec=float(data.get("timeline_start_sec", 0.0) or 0.0),
            track=str(data.get("track") or "A2"),
            volume_db=float(data.get("volume_db", 0.0) or 0.0),
        )


@dataclass
class Edl:
    name: str
    template_source: str = ""
    created_at: str = ""
    edl_version: int = EDL_VERSION
    fps: float = 30.0
    items: List[TimelineItem] = field(default_factory=list)
    text_overlays: List[TextOverlayItem] = field(default_factory=list)
    captions: List[CaptionItem] = field(default_factory=list)
    audio: List[AudioItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        if not self.items:
            return 0.0
        last = self.items[-1]
        return last.timeline_start_sec + last.duration_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "template_source": self.template_source,
            "created_at": self.created_at,
            "edl_version": self.edl_version,
            "fps": self.fps,
            "items": [i.to_dict() for i in self.items],
            "text_overlays": [o.to_dict() for o in self.text_overlays],
            "captions": [c.to_dict() for c in self.captions],
            "audio": [a.to_dict() for a in self.audio],
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edl":
        if not isinstance(data, dict):
            raise EdlError("EDL must be a JSON object")
        version = data.get("edl_version", EDL_VERSION)
        if version != EDL_VERSION:
            raise EdlError(
                f"unsupported EDL version {version}; this build supports version {EDL_VERSION}"
            )
        return cls(
            name=str(data.get("name") or "untitled"),
            template_source=str(data.get("template_source") or ""),
            created_at=str(data.get("created_at") or ""),
            edl_version=version,
            fps=float(data.get("fps", 30.0) or 30.0),
            items=[TimelineItem.from_dict(i) for i in data.get("items") or []],
            text_overlays=[TextOverlayItem.from_dict(o) for o in data.get("text_overlays") or []],
            captions=[CaptionItem.from_dict(c) for c in data.get("captions") or []],
            audio=[AudioItem.from_dict(a) for a in data.get("audio") or []],
            notes=[str(n) for n in data.get("notes") or []],
        )

    @classmethod
    def from_json(cls, text: str) -> "Edl":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EdlError(f"EDL is not valid JSON: {exc}")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Edl":
        p = Path(path)
        if not p.is_file():
            raise EdlError(f"EDL file not found: {p}")
        return cls.from_json(p.read_text(encoding="utf-8"))

    def validate(self) -> List[str]:
        """Sanity checks; returns a list of problems (empty = OK)."""
        problems = []
        if not self.items:
            problems.append("EDL has no timeline items")
        prev_end = None
        for item in self.items:
            if item.duration_sec <= 0:
                problems.append(f"item {item.id}: non-positive duration")
            if item.out_sec <= item.in_sec:
                problems.append(f"item {item.id}: source range empty (in={item.in_sec}, out={item.out_sec})")
            if prev_end is not None and abs(item.timeline_start_sec - prev_end) > 1e-3:
                problems.append(
                    f"item {item.id}: timeline gap/overlap (starts {item.timeline_start_sec}, previous ended {prev_end})"
                )
            prev_end = item.timeline_start_sec + item.duration_sec
        return problems
