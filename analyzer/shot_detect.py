"""Shot / cut detection.

Primary backend: PySceneDetect (ContentDetector) — a solved local-CV problem
that we deliberately do NOT outsource to an LLM. Fallback: a naive OpenCV
frame-difference detector (lower quality) used only when PySceneDetect is not
installed. Both are imported lazily so the rest of EditDNA loads without them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from core.analysis import Shot
from core.errors import AnalysisError, MissingDependency


@lru_cache(maxsize=1)
def _have_scenedetect() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def probe_video(path: str) -> Optional[Dict[str, float]]:
    """Return {duration_sec, fps, width, height} or None if no backend exists."""
    if _have_scenedetect():
        return _probe_scenedetect(path)
    if _have_cv2():
        return _probe_cv2(path)
    return None


def detect_shots(
    path: str,
    threshold: float = 27.0,
    min_scene_len_sec: float = 0.15,
) -> List[Shot]:
    """Detect cut timestamps. Returns [] only when the video has one shot."""
    if _have_scenedetect():
        return _detect_shots_scenedetect(path, threshold, min_scene_len_sec)
    if _have_cv2():
        return _detect_shots_cv2_fallback(path, min_scene_len_sec)
    raise MissingDependency("Shot detection", "shots")


def _probe_scenedetect(path: str) -> Dict[str, float]:
    from scenedetect import open_video

    try:
        video = open_video(path)
    except Exception as exc:
        raise AnalysisError(f"could not open video: {path} ({exc})")
    fps = float(video.frame_rate or 30.0)
    duration = float(video.duration.get_seconds()) if video.duration else 0.0
    info: Dict[str, float] = {"duration_sec": duration, "fps": fps}
    try:
        w, h = video.frame_size
        info["width"] = float(w)
        info["height"] = float(h)
    except Exception:
        pass
    return info


def _probe_cv2(path: str) -> Dict[str, float]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise AnalysisError(f"could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info = {
            "duration_sec": n_frames / fps if n_frames and n_frames > 0 else 0.0,
            "fps": float(fps),
            "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return info
    finally:
        cap.release()


def _detect_shots_scenedetect(
    path: str, threshold: float, min_scene_len_sec: float
) -> List[Shot]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    try:
        video = open_video(path)
    except Exception as exc:
        raise AnalysisError(f"could not open video: {path} ({exc})")
    duration = float(video.duration.get_seconds()) if video.duration else None
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_sec))
    sm.detect_scenes(video)
    cuts = [float(c.get_seconds()) for c in sm.get_cut_list()]
    if duration is None:
        duration = cuts[-1] + 1.0 if cuts else 0.0
    return _shots_from_cuts(cuts, duration)


def _detect_shots_cv2_fallback(path: str, min_scene_len_sec: float) -> List[Shot]:
    """Naive fallback: thresholded mean frame difference at ~6 fps."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise AnalysisError(f"could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_frames / fps if n_frames and n_frames > 0 else 0.0
        step = max(1, int(round(fps / 6.0)))
        prev = None
        diffs: List[float] = []
        times: List[float] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (160, 90))
                if prev is not None:
                    d = float(np.abs(small.astype(np.int16) - prev.astype(np.int16)).mean())
                    diffs.append(d)
                    times.append(idx / fps)
                prev = small
            idx += 1
        if not diffs:
            return []
        mean, std = float(np.mean(diffs)), float(np.std(diffs))
        threshold = mean + 2.5 * std
        cuts = [t for t, d in zip(times, diffs) if d > threshold]
        # enforce min gap
        filtered: List[float] = []
        for t in cuts:
            if not filtered or t - filtered[-1] >= min_scene_len_sec:
                filtered.append(t)
        if duration <= 0 and times:
            duration = times[-1] + 1.0 / 6.0
        return _shots_from_cuts(filtered, duration)
    finally:
        cap.release()


def _shots_from_cuts(cuts: List[float], duration: float) -> List[Shot]:
    starts = [0.0] + cuts
    ends = cuts + [duration]
    shots = []
    for s, e in zip(starts, ends):
        if e - s >= 0.05:
            shots.append(Shot(start_sec=round(s, 3), end_sec=round(e, 3)))
    if not shots and duration > 0:
        shots = [Shot(start_sec=0.0, end_sec=round(duration, 3))]
    return shots
