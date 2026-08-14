"""OCR-based on-screen text / caption detection (EasyOCR).

Per-frame detections are merged into temporal segments so each returned
TextOverlay carries an approximate on-screen window (appear → disappear),
position and relative size. OCR on busy backgrounds is noisy, so detections
below a confidence threshold are dropped and the results are meant to be
reviewed before a template is saved (see `editdna analyze --keep-ocr`).

EasyOCR is imported lazily; the first call downloads its models (~100 MB).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from core.analysis import TextOverlay
from core.errors import MissingDependency

_READERS: dict = {}


def _get_reader(languages: Sequence[str]):
    key = tuple(languages)
    if key not in _READERS:
        try:
            import easyocr
        except ImportError:
            raise MissingDependency("OCR text detection", "ocr")
        _READERS[key] = easyocr.Reader(list(languages), gpu=False, verbose=False)
    return _READERS[key]


def detect_text_overlays(
    video_path: str,
    languages: Sequence[str] = ("en",),
    min_confidence: float = 0.5,
    sample_interval_sec: float = 0.5,
    min_duration_sec: float = 0.4,
    max_samples: int = 800,
    max_width: int = 1280,
    progress: Optional[Callable[[str], None]] = None,
) -> List[TextOverlay]:
    """Detect on-screen text across a video and merge it into segments."""
    import cv2

    reader = _get_reader(languages)
    cap = cv2.VideoCapture(video_path)
    # A path that cannot be opened yields no OCR results; callers should probe
    # the video first via analyzer.shot_detect.probe_video().

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = n_frames / fps if n_frames > 0 else None
    step_frames = max(1, int(round(sample_interval_sec * fps)))

    active: List[dict] = []  # in-progress segments
    overlays: List[TextOverlay] = []

    def finalize(seg: dict) -> None:
        seg_end = seg["end"]
        if seg_end - seg["start"] >= min_duration_sec:
            overlays.append(
                TextOverlay(
                    text=seg["text"],
                    start_sec=round(seg["start"], 3),
                    end_sec=round(seg_end, 3),
                    cx=round(seg["cx"], 3),
                    cy=round(seg["cy"], 3),
                    rel_width=round(seg["rel_w"], 3),
                    rel_height=round(seg["rel_h"], 3),
                    confidence=round(seg["conf_sum"] / seg["conf_n"], 3),
                )
            )

    try:
        idx = 0
        samples = 0
        while True:
            if samples >= max_samples:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = idx / fps
            samples += 1
            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))
            H, W = frame.shape[:2]

            if progress and samples % 10 == 0:
                progress(f"OCR sample {samples} @ {t:.1f}s")

            results = reader.readtext(frame, paragraph=False)
            for bbox, text, conf in results:
                text = " ".join((text or "").split())
                if not text or conf < min_confidence:
                    continue
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = (min(xs) + max(xs)) / 2.0 / W
                cy = (min(ys) + max(ys)) / 2.0 / H
                rel_w = (max(xs) - min(xs)) / W
                rel_h = (max(ys) - min(ys)) / H

                seg = _find_active(active, text, cx, cy)
                if seg is not None:
                    seg["end"] = t
                    seg["last"] = t
                    seg["conf_sum"] += conf
                    seg["conf_n"] += 1
                    seg["cx"] = 0.7 * seg["cx"] + 0.3 * cx
                    seg["cy"] = 0.7 * seg["cy"] + 0.3 * cy
                    seg["rel_w"] = max(seg["rel_w"], rel_w)
                    seg["rel_h"] = max(seg["rel_h"], rel_h)
                else:
                    active.append(
                        {
                            "text": text,
                            "start": t,
                            "end": t,
                            "last": t,
                            "cx": cx,
                            "cy": cy,
                            "rel_w": rel_w,
                            "rel_h": rel_h,
                            "conf_sum": conf,
                            "conf_n": 1,
                        }
                    )

            # finalize segments that have not been seen for a couple of samples
            still = []
            for seg in active:
                if t - seg["last"] > sample_interval_sec * 2.0:
                    finalize(seg)
                else:
                    still.append(seg)
            active = still
            idx += step_frames
    finally:
        cap.release()

    for seg in active:
        finalize(seg)

    # merge adjacent segments that are really the same text with a gap
    overlays.sort(key=lambda o: (o.start_sec, o.cy))
    merged: List[TextOverlay] = []
    for o in overlays:
        if merged and _same_text(merged[-1], o) and o.start_sec - merged[-1].end_sec <= 1.0:
            prev = merged[-1]
            prev.end_sec = max(prev.end_sec, o.end_sec)
        else:
            merged.append(o)
    return merged


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def _same_text(a: TextOverlay, b: TextOverlay) -> bool:
    na, nb = _normalize(a.text), _normalize(b.text)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(short) >= 4 and short in long


def _find_active(active: List[dict], text: str, cx: float, cy: float) -> Optional[dict]:
    for seg in active:
        if (
            abs(seg["cx"] - cx) <= 0.12
            and abs(seg["cy"] - cy) <= 0.15
            and _texts_match(seg["text"], text)
        ):
            return seg
    return None


def _texts_match(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(short) >= 4 and short in long
