"""EditDNA command-line interface.

Usage overview:
  editdna demo                     # offline demo: sample template + EDL
  editdna analyze VIDEO            # reference video → style template
  editdna content VIDEO            # user footage → content analysis (JSON)
  editdna templates list|show      # manage saved style templates
  editdna apply TEMPLATE CLIPS...  # template + footage → EDL
  editdna build EDL                # EDL → plan table (and optionally Resolve)

All heavy dependencies (PySceneDetect, OpenCV, EasyOCR, librosa, litellm) are
imported lazily, so `--help`, `demo`, template management and plan previews
work with a bare Python install.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from core.config import Settings, templates_dir
from core.demo import build_demo_edl, demo_files, sample_template
from core.edl import Edl
from core.errors import EditDNAError, EdlError, MissingDependency, TemplateError
from core.template import StyleTemplate

VERSION = "0.1.0"

TEMPLATE_EXTENSIONS = (".json",)


def _settings() -> Settings:
    return Settings.from_env()


def _find_template(arg: str) -> Path:
    """Resolve a template argument: a path, a name, or a template in the dir."""
    p = Path(arg)
    if p.is_file():
        return p
    for candidate in templates_dir().glob(f"*{arg}*"):
        if candidate.is_file():
            return candidate
    raise TemplateError(f"template not found: {arg}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    tpl_path, edl_path = demo_files(out)
    print(f"wrote sample template: {tpl_path}")
    print(f"wrote demo EDL:        {edl_path}")
    print()
    print("next steps:")
    print(f"  editdna templates show {tpl_path}")
    print(f"  editdna build {edl_path}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from analyzer.pipeline import analyze_style

    settings = _settings()
    template = analyze_style(
        args.video,
        settings=settings,
        do_ocr=not args.no_ocr,
        do_audio=not args.no_audio,
        do_llm=not args.no_llm,
        progress=lambda msg: print(f"  {msg}"),
    )
    if args.name:
        template.template_name = args.name

    out = Path(args.out) if args.out else templates_dir() / f"{template.template_name}.json"
    template.save(out)
    print()
    print(f"saved template: {out}")
    print(f"  pacing:   {template.pacing.avg_shot_duration_sec}s avg shot, "
          f"{template.pacing.cuts_per_10s} cuts/10s, "
          f"beat-synced={template.pacing.beat_synced}")
    print(f"  cuts:     {template.pacing.cut_style}")
    print(f"  text:     {template.text_overlays.style} "
          f"({template.text_overlays.position})")
    print(f"  captions: present={template.captions.present} "
          f"({template.captions.style})")
    print(f"  summary:  {template.llm_style_summary}")
    if template.transitions:
        print("  transitions:", ", ".join(f"{t.type} ({t.frequency})" for t in template.transitions))
    return 0


def cmd_content(args: argparse.Namespace) -> int:
    from analyzer.pipeline import analyze_content

    analysis = analyze_content(
        args.video,
        settings=_settings(),
        progress=lambda msg: print(f"  {msg}"),
    )
    data = {
        "path": analysis.path,
        "duration_sec": analysis.duration_sec,
        "fps": analysis.fps,
        "scene_changes": analysis.scene_changes,
        "motion_samples": len(analysis.motion),
        "audio_peaks": analysis.audio_peaks,
        "has_audio": analysis.has_audio,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"saved content analysis: {args.out}")
    else:
        print(json.dumps(data, indent=2))
    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    if args.subcommand == "list":
        d = templates_dir()
        files = sorted(d.glob("*.json"))
        if not files:
            print(f"no templates in {d}")
            print("hint: editdna analyze <video>  or  editdna demo")
            return 0
        for f in files:
            try:
                t = StyleTemplate.load(f)
                print(f"{f.name}\t{t.pacing.cuts_per_10s} cuts/10s\t{t.llm_style_summary[:60]}")
            except EditDNAError as exc:
                print(f"{f.name}\t(ERROR: {exc})")
        return 0
    # show
    path = _find_template(args.template)
    t = StyleTemplate.load(path)
    print(t.to_json())
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from applier.edit_decision_engine import PlanOptions, plan_edit
    from core.analysis import ClipSource
    from llm.providers import get_llm

    template = StyleTemplate.load(_find_template(args.template))
    script = ""
    if args.script:
        script = Path(args.script).read_text(encoding="utf-8")

    clips: List[ClipSource] = []
    durations = None
    if args.durations:
        durations = [float(x) for x in args.durations.split(",")]
        if len(durations) != len(args.clips):
            raise EdlError("--durations must provide one value per clip")
    for i, clip_path in enumerate(args.clips):
        p = Path(clip_path)
        if not p.is_file():
            raise EdlError(f"clip not found: {clip_path}")
        analysis = None
        if durations is None:
            try:
                from analyzer.pipeline import analyze_content

                analysis = analyze_content(str(p), settings=_settings())
                print(f"  analyzed {p.name}: {analysis.duration_sec:.1f}s, "
                      f"{len(analysis.scene_changes)} scene changes")
            except MissingDependency:
                print(f"  warning: no analysis backend for {p.name}; "
                      "install editdna[shots] or pass --durations")
            except Exception as exc:
                print(f"  warning: analysis failed for {p.name}: {exc}")
        hint = durations[i] if durations else None
        clips.append(ClipSource(path=str(p), analysis=analysis, duration_hint=hint))

    llm = None
    if not args.no_llm:
        try:
            llm = get_llm(_settings())
        except Exception as exc:
            print(f"  warning: {exc}; using deterministic planner")

    opts = PlanOptions(
        fps=args.fps,
        pacing_snap_band=0.0 if args.no_scene_snap else 0.4,
    )
    edl = plan_edit(template, clips, script=script, music=args.music, llm=llm, options=opts)
    out = Path(args.out) if args.out else Path("edit_plan.json")
    edl.save(out)
    print(f"saved EDL: {out}")
    _print_edl_summary(edl)
    if edl.notes:
        print("notes:")
        for note in edl.notes:
            print(f"  - {note}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    edl = Edl.load(args.edl)
    _print_edl_summary(edl)
    if edl.notes:
        print("notes:")
        for note in edl.notes:
            print(f"  - {note}")
    if args.execute:
        from resolve_driver.resolve_bridge import build_timeline

        timeline = build_timeline(edl, project_name=args.project)
        print(f"built timeline in Resolve: {timeline}")
    else:
        print()
        print("plan preview above - run with --execute to build in DaVinci Resolve")
        print("  (Resolve must be running with scripting enabled: Preferences -> "
              "System -> General -> 'External scripting using' -> Local)")
    return 0


def _print_edl_summary(edl: Edl) -> None:
    print(f"EDL '{edl.name}'  ({edl.duration_sec:.1f}s, {len(edl.items)} items, "
          f"fps {edl.fps:g})")
    print(f"{'#':>3}  {'source':<24} {'in->out':<12} {'timeline':<9} transition")
    for i, item in enumerate(edl.items, start=1):
        name = Path(item.source_path).name
        src = f"{item.in_sec:.1f}->{item.out_sec:.1f}"
        tr = item.transition_after or "-"
        print(f"{i:>3}  {name:<24} {src:<12} {item.timeline_start_sec:<9.1f} {tr}")
    if edl.text_overlays:
        print(f"text overlays: {len(edl.text_overlays)}")
    if edl.captions:
        print(f"captions: {len(edl.captions)}")
    if edl.audio:
        print(f"audio tracks: {', '.join(a.track for a in edl.audio)}")


def cmd_demo_apply(args: argparse.Namespace) -> int:
    """demo apply: plan an edit from the sample template on placeholder clips."""
    from applier.edit_decision_engine import PlanOptions, plan_edit
    from core.analysis import ClipSource

    template = sample_template()
    clips = [
        ClipSource(path="footage_a.mp4", duration_hint=10.0),
        ClipSource(path="footage_b.mp4", duration_hint=8.0),
    ]
    edl = plan_edit(
        template,
        clips,
        script="The hook. Watch this. Now the payoff.",
        music="music_track.mp3",
        options=PlanOptions(fps=30.0),
    )
    out = Path(args.out) if args.out else Path("demo_plan.json")
    edl.save(out)
    print(f"saved demo EDL: {out}")
    _print_edl_summary(edl)
    return 0


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="editdna",
        description="Clone the editing DNA of a reference video onto your own "
        "footage, then build the timeline in DaVinci Resolve.",
    )
    parser.add_argument("--version", action="version", version=f"editdna {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="write offline demo template + EDL (no media needed)")
    p_demo.add_argument("--out-dir", default=".", help="output directory (default: .)")
    p_demo.set_defaults(func=cmd_demo)

    p_analyze = sub.add_parser("analyze", help="reference video -> style template")
    p_analyze.add_argument("video", help="path to the reference video (Reel/short)")
    p_analyze.add_argument("--out", help="output template path (default: <templates>/<name>.json)")
    p_analyze.add_argument("--name", help="template name override (default: video stem)")
    p_analyze.add_argument("--no-ocr", action="store_true", help="skip caption/text OCR")
    p_analyze.add_argument("--no-audio", action="store_true", help="skip beat/energy analysis")
    p_analyze.add_argument("--no-llm", action="store_true", help="skip LLM style summary (heuristic only)")
    p_analyze.set_defaults(func=cmd_analyze)

    p_content = sub.add_parser(
        "content",
        help="user footage -> content analysis (scene changes, motion, peaks)",
    )
    p_content.add_argument("video")
    p_content.add_argument("--out", help="write analysis JSON to a file")
    p_content.set_defaults(func=cmd_content)

    p_templates = sub.add_parser("templates", help="manage saved style templates")
    tsub = p_templates.add_subparsers(dest="subcommand", required=True)
    tsub.add_parser("list", help="list saved templates").set_defaults(func=cmd_templates)
    p_show = tsub.add_parser("show", help="print a template as JSON")
    p_show.add_argument("template", help="template name or path")
    p_show.set_defaults(func=cmd_templates)

    p_apply = sub.add_parser("apply", help="style template + footage -> EDL")
    p_apply.add_argument("template", help="template name or path")
    p_apply.add_argument("clips", nargs="+", help="raw footage files, in order")
    p_apply.add_argument("--script", help="script/voiceover text file (drives captions + overlays)")
    p_apply.add_argument("--music", help="music track (drives beat-synced cuts)")
    p_apply.add_argument("--out", help="output EDL path (default: edit_plan.json)")
    p_apply.add_argument("--durations", help="comma-separated clip durations in seconds (fallback if no analysis backend)")
    p_apply.add_argument("--fps", type=float, default=30.0, help="timeline fps (default: 30)")
    p_apply.add_argument("--no-llm", action="store_true", help="skip the LLM; deterministic planner only")
    p_apply.add_argument(
        "--no-scene-snap",
        action="store_true",
        help="cut on a pure pacing grid instead of preferring real scene changes",
    )
    p_apply.set_defaults(func=cmd_apply)

    p_build = sub.add_parser("build", help="EDL -> plan preview, or build in Resolve with --execute")
    p_build.add_argument("edl", help="path to an EDL JSON file")
    p_build.add_argument("--execute", action="store_true", help="actually drive DaVinci Resolve")
    p_build.add_argument("--project", help="Resolve project name (created if missing)")
    p_build.set_defaults(func=cmd_build)

    p_demo_apply = sub.add_parser(
        "demo-apply",
        help="plan an edit from the sample template on placeholder clips (offline)",
    )
    p_demo_apply.add_argument("--out", default="demo_plan.json")
    p_demo_apply.set_defaults(func=cmd_demo_apply)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Tolerate non-ASCII (e.g. LLM summaries) on any console/pipe encoding.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except EditDNAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
