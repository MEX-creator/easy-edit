"""Lightweight motion sampling for user footage (OpenCV, optional).

Samples the mean inter-frame difference at ~2 fps and returns (time, score)
pairs. This is a coarse activity measure — enough for the applier/LLM to know
which parts of a clip are busy vs. static — not a real optical-flow analysis.
"""

from __future__ import annotations

from typing import List, Tuple

from core.errors import MissingDependency


def sample_motion(video_path: str, duration_sec: float, sample_fps: float = 2.0) -> List[Tuple[float, float]]:
    try:
        import cv2
    except ImportError:
        raise MissingDependency("Motion sampling", "cv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / sample_fps)))
        prev = None
        out: List[Tuple[float, float]] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (128, 72))
                if prev is not None:
                    diff = float(cv2.absdiff(small, prev).mean())
                    out.append((round(idx / fps, 3), round(diff, 3)))
                prev = small
            idx += 1
        return out
    finally:
        cap.release()
