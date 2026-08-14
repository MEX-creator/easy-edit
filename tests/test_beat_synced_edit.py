"""Integration test: a real music file drives the beat grid in plan_edit().

End-to-end chain: WAV file on disk -> librosa beat detection (analyze_audio)
-> beat grid -> EDL. Asserts that every planned cut lands exactly on the
beats the analyzer actually detected, and that the template's
cuts_aligned_to_beats flag is what switches beat-locked vs pacing grids.

Skips cleanly when librosa/numpy are not installed (stdlib-only runs).
"""

import tempfile
import unittest
import wave
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    import librosa  # noqa: F401

    HAVE_AUDIO_DEPS = True
except ImportError:
    HAVE_AUDIO_DEPS = False

from analyzer.audio_beat_detect import analyze_audio
from applier.edit_decision_engine import PlanOptions, plan_edit
from core.analysis import ClipSource
from core.template import StyleTemplate

BEAT_SYNCED_TEMPLATE = StyleTemplate.from_dict(
    {
        "template_name": "beat-locked-test",
        "pacing": {
            "avg_shot_duration_sec": 0.5,
            "cut_style": "hard_cut",
            "beat_synced": True,
            "cuts_per_10s": 20,
        },
        "transitions": [],
        "text_overlays": {"style": "none"},
        "captions": {"present": False},
        "music_sync": {"cuts_aligned_to_beats": True, "energy_curve": "pulsing"},
    }
)


def _write_click_wav(path: Path, bpm: float = 120.0, seconds: float = 10.0, sr: int = 22050) -> None:
    """Write a real WAV file containing a click train at the given tempo."""
    beat_interval = 60.0 / bpm
    n_clicks = int(seconds / beat_interval)
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    click = np.hanning(int(0.02 * sr)) * 0.9
    for i in range(n_clicks):
        start = int(i * beat_interval * sr)
        if start + len(click) < len(y):
            y[start : start + len(click)] += click * (1.0 + 0.1 * (i % 3))
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


@unittest.skipUnless(HAVE_AUDIO_DEPS, "librosa/numpy not installed")
class BeatSyncedEditTest(unittest.TestCase):
    def test_cuts_land_on_detected_beat_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp) / "music.wav"
            _write_click_wav(music)

            # 1. detect the actual beats the analyzer will use
            audio = analyze_audio(str(music))
            self.assertIsNotNone(audio)
            beats = [b.time_sec for b in audio.beats]
            self.assertGreaterEqual(len(beats), 15, "click track should yield ~19 beats")
            self.assertIsNotNone(audio.tempo_bpm)
            self.assertGreater(audio.tempo_bpm, 100.0)
            self.assertLess(audio.tempo_bpm, 140.0)

            # 2. plan the edit with the music file driving the beat grid
            clip = ClipSource(path="footage.mp4", duration_hint=10.0)
            edl = plan_edit(
                BEAT_SYNCED_TEMPLATE,
                [clip],
                script="",
                music=str(music),
                options=PlanOptions(),
            )

            # 3. every cut lands exactly on a detected beat
            cuts = [item.timeline_start_sec for item in edl.items[1:]]
            self.assertEqual(len(cuts), len(beats))
            for cut, beat in zip(cuts, beats):
                self.assertAlmostEqual(cut, round(beat, 3), places=2,
                                       msg=f"cut at {cut} off the beat grid ({beat})")

            # 4. shots are beat-spaced (120 bpm => ~0.5s), no gaps or overlaps
            for item in edl.items[:-1]:
                self.assertAlmostEqual(item.duration_sec, 0.5, delta=0.05)
            self.assertEqual(edl.validate(), [])
            self.assertTrue(any("snapped to" in note for note in edl.notes))

    def test_flag_off_uses_pacing_grid_not_music(self):
        # cuts_aligned_to_beats=False must ignore the music file entirely
        template = StyleTemplate.from_dict(
            {
                "template_name": "paced-test",
                "pacing": {
                    "avg_shot_duration_sec": 0.5,
                    "cut_style": "hard_cut",
                    "beat_synced": False,
                    "cuts_per_10s": 20,
                },
                "transitions": [],
                "text_overlays": {"style": "none"},
                "captions": {"present": False},
                "music_sync": {"cuts_aligned_to_beats": False, "energy_curve": "flat"},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp) / "music.wav"
            _write_click_wav(music)
            clip = ClipSource(path="footage.mp4", duration_hint=10.0)
            edl = plan_edit(template, [clip], script="", music=str(music), options=PlanOptions())
            cuts = [item.timeline_start_sec for item in edl.items[1:]]
            expected = [round(0.5 * k, 3) for k in range(1, int(10.0 / 0.5))]
            self.assertEqual(cuts, expected)
            self.assertFalse(any("snapped to" in note for note in edl.notes))


if __name__ == "__main__":
    unittest.main()
