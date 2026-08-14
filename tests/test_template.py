"""Style template schema round-trips. Pure stdlib — always runs."""

import json
import tempfile
import unittest
from pathlib import Path

from core.demo import SAMPLE_TEMPLATE
from core.errors import TemplateError
from core.template import StyleTemplate

SPEC_EXAMPLE = {
    "template_name": "fast-hookline-reel",
    "source": "reference_video.mp4",
    "created_at": "...",
    "pacing": {
        "avg_shot_duration_sec": 1.2,
        "cut_style": "hard_cut",
        "beat_synced": True,
        "cuts_per_10s": 8,
    },
    "transitions": [{"type": "whip_pan", "frequency": "every_3rd_cut"}],
    "text_overlays": {
        "style": "bold_center_pop_in",
        "font_weight": "heavy",
        "position": "center",
        "avg_words_per_overlay": 4,
        "appears_on_beat": True,
    },
    "captions": {
        "present": True,
        "style": "karaoke_word_highlight",
        "position": "lower_third",
    },
    "music_sync": {
        "cuts_aligned_to_beats": True,
        "energy_curve": "build_to_drop",
    },
    "llm_style_summary": "some summary",
}


class TemplateRoundTripTest(unittest.TestCase):
    def test_spec_example_roundtrip(self):
        template = StyleTemplate.from_dict(SPEC_EXAMPLE)
        data = template.to_dict()
        for key, value in SPEC_EXAMPLE.items():
            self.assertEqual(data[key], value, f"key {key} changed")
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(template.pacing.avg_shot_duration_sec, 1.2)
        self.assertEqual(template.transitions[0].type, "whip_pan")

    def test_sample_template_loads(self):
        template = StyleTemplate.from_dict(SAMPLE_TEMPLATE)
        self.assertEqual(template.template_name, "fast-hookline-reel")
        self.assertTrue(template.music_sync.cuts_aligned_to_beats)

    def test_missing_schema_version_defaults_to_1(self):
        data = dict(SPEC_EXAMPLE)
        data.pop("schema_version", None)
        template = StyleTemplate.from_dict(data)
        self.assertEqual(template.schema_version, 1)

    def test_unknown_keys_ignored(self):
        data = dict(SPEC_EXAMPLE)
        data["future_field"] = {"x": 1}
        StyleTemplate.from_dict(data)  # must not raise

    def test_version_mismatch_raises(self):
        data = dict(SPEC_EXAMPLE)
        data["schema_version"] = 99
        with self.assertRaises(TemplateError):
            StyleTemplate.from_dict(data)

    def test_missing_name_raises(self):
        data = dict(SPEC_EXAMPLE)
        data.pop("template_name")
        with self.assertRaises(TemplateError):
            StyleTemplate.from_dict(data)

    def test_bad_pacing_raises(self):
        data = dict(SPEC_EXAMPLE)
        data["pacing"] = {"avg_shot_duration_sec": -5}
        with self.assertRaises(TemplateError):
            StyleTemplate.from_dict(data)

    def test_save_load_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tpl.json"
            StyleTemplate.from_dict(SPEC_EXAMPLE).save(path)
            loaded = StyleTemplate.load(path)
            expected = StyleTemplate.from_dict(SPEC_EXAMPLE).to_dict()
            self.assertEqual(loaded.to_dict(), expected)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["pacing"]["avg_shot_duration_sec"], 1.2)


if __name__ == "__main__":
    unittest.main()
