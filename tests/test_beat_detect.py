"""Beat detection on a synthetic click track (needs librosa + numpy)."""

import unittest

import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    import librosa  # noqa: F401

    HAVE_AUDIO_DEPS = True
except ImportError:
    HAVE_AUDIO_DEPS = False

from analyzer.audio_beat_detect import analyze_audio, detect_beats_from_waveform


def _click_track(bpm=120.0, seconds=10.0, sr=22050):
    """A train of short noise clicks at a fixed tempo."""
    beat_interval = 60.0 / bpm
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    n = int(seconds / beat_interval)
    click = np.hanning(int(0.02 * sr)) * 0.9
    for i in range(n):
        start = int(i * beat_interval * sr)
        if start + len(click) < len(y):
            y[start : start + len(click)] += click * (1.0 + 0.1 * (i % 3))
    return y, sr


@unittest.skipUnless(HAVE_AUDIO_DEPS, "librosa/numpy not installed")
class BeatDetectTest(unittest.TestCase):
    def test_tempo_and_beat_count(self):
        y, sr = _click_track(bpm=120.0)
        analysis = detect_beats_from_waveform(y, sr)
        self.assertTrue(analysis.has_audio)
        self.assertIsNotNone(analysis.tempo_bpm)
        self.assertGreater(analysis.tempo_bpm, 100.0)
        self.assertLess(analysis.tempo_bpm, 140.0)
        # 10s at 120 bpm ≈ 20 beats
        self.assertGreaterEqual(len(analysis.beats), 15)

    def test_beats_roughly_on_grid(self):
        y, sr = _click_track(bpm=120.0)
        analysis = detect_beats_from_waveform(y, sr)
        interval = 60.0 / 120.0
        for beat in analysis.beats[:10]:
            drift = abs(beat.time_sec % interval)
            self.assertLess(min(drift, interval - drift), 0.12)

    def test_energy_curve_normalized(self):
        y, sr = _click_track(bpm=120.0)
        analysis = detect_beats_from_waveform(y, sr)
        self.assertGreater(len(analysis.energy_curve), 10)
        values = [e for _, e in analysis.energy_curve]
        self.assertLessEqual(max(values), 1.0 + 1e-6)

    def test_silence_raises(self):
        y = np.zeros(22050, dtype=np.float32)
        with self.assertRaises(Exception):
            detect_beats_from_waveform(y, 22050)


@unittest.skipUnless(HAVE_AUDIO_DEPS and shutil.which("ffmpeg"), "librosa + ffmpeg not available")
class FfmpegFallbackTest(unittest.TestCase):
    def test_decode_audio_from_mp4(self):
        # soundfile cannot read aac-in-mp4, so this exercises the ffmpeg fallback
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "tone.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"), "-v", "error",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                    "-c:a", "aac", str(mp4),
                ],
                check=True,
            )
            analysis = analyze_audio(str(mp4))
            self.assertIsNotNone(analysis)
            self.assertTrue(analysis.has_audio)
            self.assertGreater(len(analysis.energy_curve), 5)

    def test_video_without_audio_returns_none(self):
        # a silent video must yield None (no crash, no exception)
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "silent.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"), "-v", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-c:v", "libx264", "-an", str(silent),
                ],
                check=True,
            )
            self.assertIsNone(analyze_audio(str(silent)))


if __name__ == "__main__":
    unittest.main()
