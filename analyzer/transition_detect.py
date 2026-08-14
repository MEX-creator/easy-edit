"""Best-effort transition classification around detected cut boundaries.

The pure function classify_transition() operates on 5 grayscale frames
straddling a cut ([t-2, t-1, t, t+1, t+2]) and is unit-tested on synthetic
frames. classify_transitions() samples real frames around each cut via OpenCV.

Heuristic (documented limits): a sharp frame-to-frame jump with incoherent
motion is a hard cut; a coherent large translation is a whip pan; a coherent
scale change is a zoom cut; a gradual blend across several sampled frames is a
crossfade; a dark middle frame is a dip to black.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

from core.analysis import Shot, Transition
from core.errors import AnalysisError, MissingDependency

VALID_KINDS = ("hard_cut", "crossfade", "whip_pan", "zoom_cut", "dip_to_black", "unknown")

# numpy is imported lazily so this module loads without any optional deps.
# `ArrayLike` here means any object numpy accepts (np.ndarray, lists of ints).
ArrayLike = Union["np.ndarray", Sequence]


def _mean_abs_diff(a, b) -> float:
    import numpy as np

    return float(np.abs(np.asarray(a).astype(np.int16) - np.asarray(b).astype(np.int16)).mean())


def classify_transition(frames: Sequence) -> Tuple[str, float]:
    """Classify the transition at a cut boundary from 5 grayscale uint8 frames.

    Returns (kind, confidence). Kind is one of VALID_KINDS.
    """
    import numpy as np

    if len(frames) != 5:
        raise ValueError("classify_transition expects exactly 5 frames")
    a, b, mid, c, d = frames
    pre_mid = _mean_abs_diff(b, mid)
    post_mid = _mean_abs_diff(mid, c)
    pre_outer = _mean_abs_diff(a, b)
    post_outer = _mean_abs_diff(c, d)
    outer = (pre_outer + post_outer) / 2.0
    inner = max(pre_mid, post_mid)

    mid_bright = float(np.mean(mid))
    side_bright = max(float(np.mean(b)), float(np.mean(c)))

    # 1) dip to black
    if side_bright > 25.0 and mid_bright < 0.10 * side_bright:
        return "dip_to_black", 0.7

    # 2) sharp boundary: the frames immediately around the cut differ far more
    #    than within-shot frames. Check whether that difference is coherent
    #    motion (whip/zoom) or an instantaneous cut.
    if outer < 1e-6 or inner / (outer + 1e-6) >= 1.6:
        kind, conf = _classify_motion(b, c)
        if kind == "hard_cut":
            sharpness = inner / (outer + 1e-6)
            return "hard_cut", min(0.95, 0.55 + 0.05 * sharpness)
        return kind, conf

    # 3) gradual blend across sampled frames → crossfade. If nothing is
    #    changing on either side of the boundary it's not a real cut at all.
    if outer < 3.0:
        return "unknown", 0.2
    return "crossfade", 0.55


def _classify_motion(a, b) -> Tuple[str, float]:
    """Is the difference between two same-shot-adjacent frames motion or noise?"""
    import cv2
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    h, w = a.shape
    fa = np.float32(a)
    fb = np.float32(b)

    # translation via phase correlation (robust on structured content).
    # Phase correlation alone can false-positive between unrelated images with
    # strong gradients, so we validate the estimated displacement with
    # normalized cross-correlation at the aligned position.
    (dx, dy), response = cv2.phaseCorrelate(fa, fb)
    shift = math.hypot(dx, dy)
    ncc = 0.0
    if 0.0 < shift < min(h, w) * 0.5:
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(b, m, (w, h))
        ncc = _ncc(a, aligned)

    base_err = _mean_abs_diff(a, b)
    best_scale, best_err = 1.0, base_err
    for s in (0.8, 0.9, 1.1, 1.2):
        scaled = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        scaled = _center_fit(scaled, h, w)
        err = _mean_abs_diff(scaled, b)
        if err < best_err:
            best_scale, best_err = s, err
    zoom = abs(math.log(best_scale))

    if zoom > 0.08 and best_err < 0.85 * base_err:
        return "zoom_cut", 0.6
    if response > 0.08 and shift > 2.0 and ncc > 0.25:
        return "whip_pan", min(0.8, 0.5 + shift / 40.0)
    return "hard_cut", 0.6


def _ncc(a, b) -> float:
    """Normalized cross-correlation between two same-sized grayscale images."""
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    am = a - a.mean()
    bm = b - b.mean()
    denom = float(np.sqrt((am * am).sum() * (bm * bm).sum())) + 1e-6
    return float((am * bm).sum()) / denom


def _center_fit(img, h: int, w: int):
    """Resize result to exactly (h, w), cropping or padding from the center."""
    import numpy as np

    ih, iw = img.shape[:2]
    if ih == h and iw == w:
        return img
    if ih < h or iw < w:
        out = np.zeros((h, w), dtype=img.dtype)
        y0, x0 = (h - ih) // 2, (w - iw) // 2
        out[y0 : y0 + ih, x0 : x0 + iw] = img
        return out
    y0, x0 = (ih - h) // 2, (iw - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w]


def classify_transitions(
    video_path: str,
    shots: List[Shot],
    step_sec: float = 0.2,
) -> List[Transition]:
    """Classify every cut boundary in a real video via OpenCV frame sampling."""
    if len(shots) < 2:
        return []
    try:
        import cv2
    except ImportError:
        raise MissingDependency("Transition classification", "cv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisError(f"could not open video: {video_path}")
    results: List[Transition] = []
    try:
        for i in range(1, len(shots)):
            t = shots[i].start_sec
            times = [
                t - 2 * step_sec,
                t - step_sec,
                t,
                t + step_sec,
                t + 2 * step_sec,
            ]
            frames = []
            for tt in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, tt) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    frames = None
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, (320, 180)))
            if frames is None:
                continue
            kind, conf = classify_transition(frames)
            results.append(
                Transition(at_sec=round(t, 3), kind=kind, confidence=conf, index=i)
            )
    finally:
        cap.release()
    return results
