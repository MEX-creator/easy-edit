"""EDL schema round-trips. Pure stdlib — always runs."""

import json
import tempfile
import unittest
from pathlib import Path

from core.demo import build_demo_edl
from core.edl import AudioItem, CaptionItem, Edl, TextOverlayItem, TimelineItem
from core.errors import EdlError


def make_edl() -> Edl:
    items = [
        TimelineItem(
            id="v1", source_path="a.mp4", in_sec=0.0, out_sec=2.0,
            timeline_start_sec=0.0, duration_sec=2.0,
        ),
        TimelineItem(
            id="v2", source_path="b.mp4", in_sec=0.0, out_sec=2.0,
            timeline_start_sec=2.0, duration_sec=2.0, transition_after="crossfade",
            transition_duration_sec=0.4,
        ),
    ]
    return Edl(
        name="test",
        fps=30.0,
        items=items,
        text_overlays=[
            TextOverlayItem(id="t1", text="Hello", timeline_start_sec=0.1, duration_sec=1.0)
        ],
        captions=[
            CaptionItem(id="c1", text="Hi.", timeline_start_sec=0.0, duration_sec=2.0)
        ],
        audio=[AudioItem(path="m.mp3", track="A2")],
    )


class EdlTest(unittest.TestCase):
    def test_roundtrip(self):
        edl = make_edl()
        data = edl.to_dict()
        restored = Edl.from_dict(data)
        self.assertEqual(restored.to_dict(), data)
        self.assertEqual(restored.items[1].transition_after, "crossfade")

    def test_missing_sections_default(self):
        edl = Edl.from_dict({"name": "empty", "fps": 24})
        self.assertEqual(edl.items, [])
        self.assertEqual(edl.duration_sec, 0.0)

    def test_version_mismatch(self):
        with self.assertRaises(EdlError):
            Edl.from_dict({"name": "x", "edl_version": 42, "items": []})

    def test_validate_ok(self):
        self.assertEqual(make_edl().validate(), [])

    def test_validate_flags_gap(self):
        edl = make_edl()
        edl.items[1].timeline_start_sec = 3.0
        self.assertTrue(any("gap" in p for p in edl.validate()))

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edl.json"
            make_edl().save(path)
            loaded = Edl.load(path)
            self.assertEqual(loaded.to_dict(), make_edl().to_dict())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["name"], "test")

    def test_demo_edl_is_valid(self):
        edl = build_demo_edl()
        self.assertEqual(edl.validate(), [])
        self.assertEqual(len(edl.items), 4)
        self.assertAlmostEqual(edl.duration_sec, 8.4)


if __name__ == "__main__":
    unittest.main()
