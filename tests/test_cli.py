"""CLI smoke tests for the offline commands (no media, no heavy deps)."""

import json
import tempfile
import unittest
from pathlib import Path

import cli


class CliOfflineTest(unittest.TestCase):
    def test_demo_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = cli.main(["demo", "--out-dir", tmp])
            self.assertEqual(rc, 0)
            tpl = Path(tmp) / "sample_fast_hookline_reel.json"
            edl = Path(tmp) / "demo_edl.json"
            self.assertTrue(tpl.is_file())
            self.assertTrue(edl.is_file())
            data = json.loads(tpl.read_text(encoding="utf-8"))
            self.assertEqual(data["template_name"], "fast-hookline-reel")

    def test_templates_show_repo_sample(self):
        rc = cli.main(["templates", "show", "templates/sample_fast_hookline_reel.json"])
        self.assertEqual(rc, 0)

    def test_templates_list(self):
        rc = cli.main(["templates", "list"])
        self.assertEqual(rc, 0)

    def test_demo_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            rc = cli.main(["demo-apply", "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["edl_version"], 1)
            self.assertGreaterEqual(len(data["items"]), 1)

    def test_build_preview_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = cli.main(["demo", "--out-dir", tmp])
            self.assertEqual(rc, 0)
            edl_path = Path(tmp) / "demo_edl.json"
            rc = cli.main(["build", str(edl_path)])
            self.assertEqual(rc, 0)

    def test_unknown_command_errors(self):
        with self.assertRaises(SystemExit):
            cli.main(["nope"])

    def test_missing_video_analyze_fails_gracefully(self):
        # analyze on a nonexistent file should exit non-zero, not traceback
        with tempfile.TemporaryDirectory() as tmp:
            rc = cli.main(["analyze", str(Path(tmp) / "missing.mp4"), "--no-llm"])
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
