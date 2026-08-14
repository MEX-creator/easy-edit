"""Shared data structures for analysis results.

Two distinct kinds of analysis exist (see analyzer/pipeline.py):
  * StyleAnalysis   — what a reference video's EDIT looks like (pacing,
                      transitions, on-screen text, music sync). Feeds the
                      style template.
  * ClipAnalysis    — what the USER'S FOOTAGE contains (duration, scene
                      changes, motion, audio peaks). Feeds the applier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .errors import AnalysisError


@dataclass
class Shot:
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class Transition:
    at_sec: float
    kind: str  # hard_cut | crossfade | whip_pan | zoom_cut | dip_to_black | unknown
    confidence: float = 0.5
    index: int = 0  # index of the cut this transition belongs to


@dataclass
class TextOverlay:
    text: str
    start_sec: float
    end_sec: float
    cx: float  # normalized center x (0..1)
    cy: float  # normalized center y (0..1)
    rel_width: float  # bbox width relative to frame width
    rel_height: float  # bbox height relative to frame height
    confidence: float = 0.5

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class Beat:
    time_sec: float
    strength: float = 1.0


@dataclass
class AudioAnalysis:
    beats: List[Beat] = field(default_factory=list)
    energy_curve: List[Tuple[float, float]] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    has_audio: bool = True


@dataclass
class StyleAnalysis:
    video_path: str
    duration_sec: float
    fps: float
    shots: List[Shot] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    text_overlays: List[TextOverlay] = field(default_factory=list)
    audio: Optional[AudioAnalysis] = None

    # ---- derived pacing metrics ----
    @property
    def avg_shot_duration_sec(self) -> float:
        if not self.shots:
            return 0.0
        return sum(s.duration for s in self.shots) / len(self.shots)

    @property
    def cuts_per_10s(self) -> float:
        if self.duration_sec <= 0:
            return 0.0
        n_cuts = max(0, len(self.shots) - 1)
        return n_cuts / self.duration_sec * 10.0

    @property
    def beat_synced_fraction(self) -> float:
        """Fraction of cuts that land within ~0.12s of a detected beat."""
        if not self.audio or not self.audio.beats or len(self.shots) < 2:
            return 0.0
        cuts = [s.start_sec for s in self.shots[1:]]
        beat_times = [b.time_sec for b in self.audio.beats]
        near = sum(1 for c in cuts if any(abs(c - b) <= 0.12 for b in beat_times))
        return near / len(cuts)

    def dominant_transition_kind(self) -> str:
        kinds = [t.kind for t in self.transitions if t.kind not in ("unknown", "hard_cut")]
        if not kinds:
            return "hard_cut"
        return max(set(kinds), key=kinds.count)

    def energy_shape(self) -> str:
        """Classify the energy curve: build_to_drop | front_loaded | flat | unknown."""
        curve = self.audio.energy_curve if self.audio else []
        if len(curve) < 6:
            return "unknown"
        values = [e for _, e in curve]
        n = len(values)
        third = max(1, n // 3)
        first = sum(values[:third]) / third
        last = sum(values[-third:]) / third
        if last > first * 1.15:
            return "build_to_drop"
        if first > last * 1.15:
            return "front_loaded"
        return "flat"

    def require_shots(self) -> None:
        if not self.shots:
            raise AnalysisError("no shots detected; cannot build a style template")


@dataclass
class ClipAnalysis:
    path: str
    duration_sec: float
    fps: float = 30.0
    scene_changes: List[float] = field(default_factory=list)
    motion: List[Tuple[float, float]] = field(default_factory=list)  # (time, motion score)
    audio_peaks: List[float] = field(default_factory=list)
    has_audio: bool = False

    @property
    def avg_motion(self) -> float:
        if not self.motion:
            return 0.0
        return sum(score for _, score in self.motion) / len(self.motion)


@dataclass
class ClipSource:
    """A user-supplied raw clip plus (optional) analysis of its content."""

    path: str
    analysis: Optional[ClipAnalysis] = None
    duration_hint: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.analysis and self.analysis.duration_sec:
            return self.analysis.duration_sec
        return self.duration_hint
