"""Deterministic (no-LLM) edit planning. Pure stdlib — always runs."""

import unittest

from applier.edit_decision_engine import PlanOptions, _plan_shots, plan_edit
from core.analysis import ClipAnalysis, ClipSource
from core.errors import EdlError
from core.template import StyleTemplate

TEMPLATE = StyleTemplate.from_dict(
    {
        "template_name": "fast-hookline-reel",
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
        "captions": {"present": True, "style": "sentence_lower_third", "position": "lower_third"},
        "music_sync": {"cuts_aligned_to_beats": True, "energy_curve": "build_to_drop"},
    }
)

SCRIPT = "The hook. Watch this. Now the payoff."


def clips_with_durations(*durations: float) -> list:
    out = []
    for i, d in enumerate(durations):
        out.append(ClipSource(path=f"clip{i}.mp4", duration_hint=d))
    return out


class PlanShotsTest(unittest.TestCase):
    def test_uniform_grid(self):
        clips = clips_with_durations(8.0, 6.0)
        segments = _plan_shots(clips, 14.0, 1.2, [], PlanOptions())
        # per-clip greedy with no scene changes: 7 shots (6x1.2 + 0.8 tail)
        # for the 8s clip, 5 shots (5x1.2) for the 6s clip => 12 total
        self.assertEqual(len(segments), 12)
        for ci, s, e in segments:
            self.assertGreater(e - s, 0.0)
        # clip 1 supplies exactly 8.0s, clip 2 exactly 6.0s
        self.assertEqual(sum(e - s for ci, s, e in segments if ci == 0), 8.0)
        self.assertEqual(sum(e - s for ci, s, e in segments if ci == 1), 6.0)

    def test_beat_grid_snaps(self):
        clips = clips_with_durations(10.0)
        beats = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        segments = _plan_shots(clips, 10.0, 1.2, beats, PlanOptions())
        cuts = [s for _, s, _ in segments[1:]]
        self.assertEqual(cuts, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_single_short_clip_single_shot(self):
        clips = clips_with_durations(0.5)
        segments = _plan_shots(clips, 0.5, 1.2, [], PlanOptions())
        self.assertEqual(segments, [(0, 0.0, 0.5)])

    def test_dense_beats_deduplicated_to_min_gap(self):
        clips = clips_with_durations(5.0)
        beats = [0.1, 0.12, 0.15, 2.0, 2.02, 4.0]
        segments = _plan_shots(clips, 5.0, 1.2, beats, PlanOptions(min_shot_sec=0.4))
        cuts = [s for _, s, _ in segments[1:]]
        self.assertEqual(cuts, [0.1, 2.0, 4.0])


class PlanEditTest(unittest.TestCase):
    def test_plan_structure(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(8.0, 6.0), script=SCRIPT)
        # no scene changes -> pure pacing grid: 7 + 5 items
        self.assertEqual(len(edl.items), 12)
        for item in edl.items:
            self.assertLessEqual(item.duration_sec, 1.2 + 1e-6)
        self.assertEqual(edl.validate(), [])

    def test_timeline_is_continuous(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(8.0, 6.0), script=SCRIPT)
        prev_end = 0.0
        for item in edl.items:
            self.assertAlmostEqual(item.timeline_start_sec, prev_end, places=3)
            prev_end = item.timeline_start_sec + item.duration_sec
        self.assertAlmostEqual(prev_end, 14.0, places=3)

    def test_transition_cadence_every_3rd_cut(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(8.0, 6.0), script=SCRIPT)
        # 12 items -> every 3rd internal cut (indices 2, 5, 8); the last item
        # has no next clip so it can't carry a transition
        transitions = [i for i, item in enumerate(edl.items) if item.transition_after]
        self.assertEqual(transitions, [2, 5, 8])
        for idx in transitions:
            self.assertEqual(edl.items[idx].transition_after, "whip_pan")

    def test_captions_one_per_shot_no_repeat(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(8.0, 6.0), script=SCRIPT)
        self.assertEqual(len(edl.captions), 3)
        self.assertEqual([c.text for c in edl.captions], ["The hook.", "Watch this.", "Now the payoff."])

    def test_overlays_from_script_phrases(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(8.0, 6.0), script=SCRIPT)
        self.assertEqual(len(edl.text_overlays), 3)
        for i, overlay in enumerate(edl.text_overlays):
            # overlay i lands on item i, starts at the item start (no beats here),
            # and never outlives its shot
            self.assertEqual(overlay.timeline_start_sec, edl.items[i].timeline_start_sec)
            self.assertLessEqual(
                overlay.timeline_start_sec + overlay.duration_sec,
                edl.items[i].timeline_start_sec + edl.items[i].duration_sec + 1e-6,
            )

    def test_no_script_no_text(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(4.0), script="")
        self.assertEqual(edl.text_overlays, [])
        self.assertEqual(edl.captions, [])
        self.assertTrue(any("no script" in n for n in edl.notes))

    def test_music_adds_audio_item(self):
        edl = plan_edit(TEMPLATE, clips_with_durations(4.0), script="", music="track.mp3")
        self.assertEqual(len(edl.audio), 1)
        self.assertEqual(edl.audio[0].path, "track.mp3")

    def test_pacing_mode_prefers_scene_changes(self):
        analysis = ClipAnalysis(
            path="clip0.mp4", duration_sec=8.0,
            scene_changes=[1.0, 2.5, 4.0, 6.0],
        )
        clip = ClipSource(path="clip0.mp4", analysis=analysis)
        edl = plan_edit(TEMPLATE, [clip], script="")
        cuts = [item.timeline_start_sec for item in edl.items[1:]]
        for scene in (1.0, 2.5, 4.0, 6.0):
            self.assertIn(scene, cuts, f"scene change {scene} not used as a cut")
        # pacing preserved: every non-tail shot stays within +/- 0.4 of target
        for item in edl.items[:-1]:
            self.assertGreaterEqual(item.duration_sec, 1.2 * 0.6 - 1e-6)
            self.assertLessEqual(item.duration_sec, 1.2 * 1.4 + 1e-6)

    def test_no_scene_snap_band_zero_gives_pure_grid(self):
        analysis = ClipAnalysis(
            path="clip0.mp4", duration_sec=8.0,
            scene_changes=[1.0, 2.5, 4.0, 6.0],
        )
        clip = ClipSource(path="clip0.mp4", analysis=analysis)
        opts = PlanOptions(pacing_snap_band=0.0)
        edl = plan_edit(TEMPLATE, [clip], script="", options=opts)
        cuts = [item.timeline_start_sec for item in edl.items[1:]]
        self.assertEqual(cuts, [1.2, 2.4, 3.6, 4.8, 6.0, 7.2])

    def test_beat_mode_keeps_beat_grid_with_scene_changes(self):
        analysis = ClipAnalysis(
            path="clip0.mp4", duration_sec=10.0,
            scene_changes=[1.05, 3.1, 5.0, 7.9],
        )
        clip = ClipSource(path="clip0.mp4", analysis=analysis)
        beats = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        segments = _plan_shots([clip], 10.0, 1.2, beats, PlanOptions())
        cuts = [s for _, s, _ in segments[1:]]
        # the beat grid is the template's DNA: it stays exactly on the beats
        self.assertEqual(cuts, beats)
        self.assertEqual(sum(e - s for _, s, e in segments), 10.0)

    def test_missing_duration_raises(self):
        clips = [ClipSource(path="unknown.mp4")]
        with self.assertRaises(EdlError):
            plan_edit(TEMPLATE, clips)

    def test_hard_cut_template_no_transitions(self):
        tpl = StyleTemplate.from_dict(
            {
                "template_name": "plain",
                "pacing": {"avg_shot_duration_sec": 2.0},
                "captions": {"present": False},
                "text_overlays": {"style": "none"},
                "music_sync": {"cuts_aligned_to_beats": False},
            }
        )
        edl = plan_edit(tpl, clips_with_durations(6.0), script="")
        self.assertEqual(len(edl.items), 3)
        self.assertTrue(all(i.transition_after is None for i in edl.items))

    def test_clip_analysis_scene_changes_used_for_llm_context_only(self):
        # scene changes now drive the pacing-mode cuts (they are also sent to
        # the LLM in the assisted path)
        analysis = ClipAnalysis(
            path="clip0.mp4", duration_sec=8.0,
            scene_changes=[1.0, 3.0, 5.0],
            motion=[(0.0, 1.0), (1.0, 2.0)],
            has_audio=True,
        )
        clip = ClipSource(path="clip0.mp4", analysis=analysis)
        edl = plan_edit(TEMPLATE, [clip], script="Hey.")
        cuts = [item.timeline_start_sec for item in edl.items[1:]]
        # every scene change that sits near the pacing grid becomes a cut
        for scene in (1.0, 3.0, 5.0):
            self.assertIn(scene, cuts, f"scene change {scene} not used as a cut")
        # no cut is ever shorter than the min shot length (except the tail)
        for item in edl.items[:-1]:
            self.assertGreaterEqual(item.duration_sec, 0.4 - 1e-6)


if __name__ == "__main__":
    unittest.main()
