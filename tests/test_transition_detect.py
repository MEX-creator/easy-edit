"""Transition classification on synthetic frames (needs cv2 + numpy)."""

import unittest

try:
    import numpy as np  # noqa: F401
    import cv2  # noqa: F401

    HAVE_CV_DEPS = True
except ImportError:
    HAVE_CV_DEPS = False

from analyzer.transition_detect import classify_transition


def _pattern(size=96, seed=0):
    """A structured image: gradient background + bright square — phase
    correlation and scaling behave predictably on it."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    ramp = np.linspace(30, 90, size, dtype=np.uint8)
    img += ramp[None, :]
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, (size, size)), 0, 255).astype(np.uint8)
    img[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 200
    return img


def _noise(size=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def _shift(img, dx, dy=0):
    import cv2

    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


def _scale(img, factor):
    import cv2

    h, w = img.shape
    small = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w), dtype=np.uint8)
    ih, iw = small.shape
    y0, x0 = (h - ih) // 2, (w - iw) // 2
    out[y0 : y0 + ih, x0 : x0 + iw] = small
    return out


def _blend(a, b, t):
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


@unittest.skipUnless(HAVE_CV_DEPS, "cv2/numpy not installed")
class TransitionClassifyTest(unittest.TestCase):
    def test_hard_cut(self):
        a, b = _pattern(seed=1), _noise(seed=2)
        frames = [a, a, b, b, b]
        kind, conf = classify_transition(frames)
        self.assertEqual(kind, "hard_cut", f"got {kind}")
        self.assertGreaterEqual(conf, 0.5)

    def test_crossfade(self):
        a, b = _pattern(seed=1), _noise(seed=2)
        frames = [
            a,
            _blend(a, b, 0.3),
            _blend(a, b, 0.5),
            _blend(a, b, 0.7),
            b,
        ]
        kind, conf = classify_transition(frames)
        self.assertEqual(kind, "crossfade", f"got {kind}")

    def test_dip_to_black(self):
        a, b = _pattern(seed=1), _noise(seed=2)
        black = np.zeros_like(a)
        frames = [a, a, black, b, b]
        kind, conf = classify_transition(frames)
        self.assertEqual(kind, "dip_to_black", f"got {kind}")

    def test_whip_pan(self):
        base = _pattern(seed=3)
        panned = _shift(base, dx=10)
        # frames[1] is pre-cut, frames[3] is post-cut; the cut is a fast pan
        frames = [base, base, panned, panned, panned]
        kind, conf = classify_transition(frames)
        self.assertEqual(kind, "whip_pan", f"got {kind}")

    def test_zoom_cut(self):
        base = _pattern(seed=4)
        zoomed = _scale(base, 0.85)
        frames = [base, base, zoomed, zoomed, zoomed]
        kind, conf = classify_transition(frames)
        self.assertEqual(kind, "zoom_cut", f"got {kind}")

    def test_requires_5_frames(self):
        a = _pattern()
        with self.assertRaises(ValueError):
            classify_transition([a, a, a])


if __name__ == "__main__":
    unittest.main()
