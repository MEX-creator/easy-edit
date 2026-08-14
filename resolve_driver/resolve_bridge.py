"""Translate an EditDNA EDL into DaVinci Resolve API calls.

Two ways to run this:

1. From a normal terminal: `editdna build edl.json --execute` — the bridge
   finds the bundled DaVinciResolveScript module, connects to a running
   Resolve instance (scripting must be enabled: Preferences → System →
   General → External scripting using → Local), and builds the timeline.
2. From inside Resolve (Workspace → Console): paste a one-liner that calls
   `build_timeline_from_file("path/to/edl.json")`. Resolve's Python env has
   DaVinciResolveScript importable already.

Design notes / known limits (Resolve Free):
- Every API interaction is defensive (try/except + logging) because the
  scripting surface varies across Resolve versions.
- Source in/out via SetProperty("START"/"END") and timeline position via
  "TIMELINEPOS" work on current Resolve; if a version rejects them we log a
  warning instead of failing the whole build.
- Text+ styling via clip properties is best-effort; precise animated titles
  (Fusion) are explicitly deferred to v2.
- Resolve Free can't export H.265/ProRes on some platforms and has no Neural
  Engine features — flag those at export time, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.edl import Edl
from core.errors import ResolveError

# Known install locations of the DaVinciResolveScript module (in addition to
# the environment Resolve exposes when running scripts from its console).
SCRIPT_MODULE_CANDIDATES = [
    Path(r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules"),
    Path("/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"),
    Path("/opt/resolve/Developer/Scripting/Modules"),
    Path.home() / "AppData/Roaming/Blackmagic Design/DaVinci Resolve/Support/Developer/Scripting/Modules",
]

# kind -> Resolve transition identifier (name may vary by version; we verify
# against the list Resolve reports when available).
TRANSITION_IDS = {
    "crossfade": "cross dissolve",
    "dip_to_black": "dip to color dissolution",
    "whip_pan": "swish pan",
    "zoom_cut": "swish pan",
}


def _import_resolve_script():
    try:
        import DaVinciResolveScript  # noqa: F401

        return DaVinciResolveScript
    except ImportError:
        pass
    for candidate in SCRIPT_MODULE_CANDIDATES:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            try:
                import DaVinciResolveScript  # noqa: F401

                return DaVinciResolveScript
            except ImportError:
                continue
    return None


def get_resolve():
    """Connect to a running Resolve instance or raise ResolveError."""
    module = _import_resolve_script()
    if module is None:
        raise ResolveError(
            "DaVinciResolveScript not found. Make sure DaVinci Resolve is "
            "installed and that scripting is enabled (Preferences -> System -> "
            "General -> 'External scripting using' -> Local), then try again."
        )
    resolve = module.scriptapp("Resolve")
    if resolve is None:
        raise ResolveError(
            "Could not connect to DaVinci Resolve. Is it running with "
            "'External scripting using' set to Local?"
        )
    return resolve


Log = Optional[Callable[[str], None]]


def _default_log(msg: str) -> None:
    print(f"[resolve] {msg}")


def _frames(sec: float, fps: float) -> int:
    return max(0, int(round(sec * fps)))


def _available_transitions(resolve) -> Dict[str, Any]:
    try:
        getter = getattr(resolve, "GetTransitionList", None) or getattr(
            resolve, "GetTransitionsList", None
        )
        if getter is None:
            return {}
        data = getter()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _transition_id_for(kind: str, available: Dict[str, Any]) -> Optional[str]:
    preferred = TRANSITION_IDS.get(kind)
    if preferred and (not available or preferred in available):
        return preferred
    # fall back to any available transition that looks similar
    for key in available:
        if kind.split("_")[0] in str(key).lower():
            return key
    return preferred or None


def build_timeline(
    edl: Edl,
    resolve=None,
    project_name: Optional[str] = None,
    log: Log = None,
) -> str:
    """Build (or update) a Resolve project/timeline from an Edl.

    Returns the timeline name. All API calls are defensive: failures log and
    continue so a partial build is still useful for human polish.
    """
    log = log or _default_log
    problems = edl.validate()
    for p in problems:
        log(f"warning: {p}")

    resolve = resolve or get_resolve()
    pm = resolve.GetProjectManager()
    project = None
    if project_name:
        project = pm.CreateProject(project_name)
        if project is None:
            project = pm.LoadProject(project_name)
    if project is None:
        project = pm.GetCurrentProject()
    if project is None:
        raise ResolveError("no project available in Resolve")

    mp = project.GetMediaPool()
    fps = edl.fps or 30.0

    # ---- 1. import media ----
    paths: List[str] = []
    for item in edl.items:
        if item.source_path not in paths:
            paths.append(item.source_path)
    for audio in edl.audio:
        if audio.path not in paths:
            paths.append(audio.path)

    imported = mp.ImportMedia(paths) if paths else None
    imported = imported if isinstance(imported, list) else []
    pool_items: Dict[str, Any] = {}
    for i, item in enumerate(imported):
        if i < len(paths):
            pool_items[paths[i]] = item
    missing = [p for p in paths if p not in pool_items or pool_items[p] is None]
    for p in missing:
        log(f"warning: could not import {p} (missing file? already imported?)")

    # ---- 2. timeline from clips ----
    ordered_paths: List[str] = []
    for item in edl.items:
        if item.source_path not in ordered_paths:
            ordered_paths.append(item.source_path)
    clip_objs = [pool_items[p] for p in ordered_paths if p in pool_items and pool_items[p] is not None]
    if not clip_objs:
        raise ResolveError("no footage could be imported; nothing to build")
    timeline = mp.CreateTimelineFromClips(clip_objs)
    if timeline is None:
        timeline = pm.GetCurrentProject().GetCurrentTimeline()
    if timeline is None:
        raise ResolveError("could not create timeline from imported clips")
    timeline_name = timeline.GetName()
    log(f"timeline '{timeline_name}' ready")

    # ---- 3. set source in/out + timeline position ----
    track_items = timeline.GetItemListInTrack("video", 1) or []
    for idx, edl_item in enumerate(edl.items):
        if idx >= len(track_items):
            log(f"warning: more EDL items than clips on V1 ({idx})")
            break
        clip = track_items[idx]
        try:
            clip.SetProperty("START", _frames(edl_item.in_sec, fps))
            clip.SetProperty("END", _frames(edl_item.out_sec, fps))
        except Exception as exc:
            log(f"warning: could not set in/out for item {edl_item.id}: {exc}")
        try:
            clip.SetProperty("TIMELINEPOS", _frames(edl_item.timeline_start_sec, fps))
        except Exception:
            log(
                f"warning: TIMELINEPOS not supported by this Resolve version; "
                f"item {edl_item.id} may need manual positioning (gaps possible)"
            )

    # ---- 4. transitions ----
    available = _available_transitions(resolve)
    for i in range(len(edl.items) - 1):
        item = edl.items[i]
        kind = item.transition_after
        if not kind or kind == "hard_cut":
            continue
        if i + 1 >= len(track_items):
            break
        trans_id = _transition_id_for(kind, available)
        if trans_id is None:
            log(f"warning: no Resolve transition for {kind}; skipping")
            continue
        try:
            ok = timeline.AddTransition(
                track_items[i].GetName(),
                track_items[i + 1].GetName(),
                trans_id,
                _frames(item.transition_duration_sec or 0.4, fps),
            )
            if not ok:
                log(f"warning: AddTransition({kind}) returned False")
        except Exception as exc:
            log(f"warning: AddTransition({kind}) failed: {exc}")

    # ---- 5. Text+ overlays + captions on an overlay track ----
    overlay_specs = []
    for o in edl.text_overlays:
        overlay_specs.append((o.text, o.timeline_start_sec, o.duration_sec, o.cx, o.cy, o.font_size_rel))
    for c in edl.captions:
        overlay_specs.append((c.text, c.timeline_start_sec, c.duration_sec, c.cx, c.cy, 0.05))
    if overlay_specs:
        try:
            overlay_track = timeline.AddTrack("Video", "Overlays")
        except Exception as exc:
            log(f"warning: could not create overlay track: {exc}")
            overlay_track = None
        for text, start, dur, cx, cy, font in overlay_specs:
            title = None
            try:
                title = mp.CreateTitleFromText("Text+", text)
            except Exception as exc:
                log(f"warning: CreateTitleFromText failed ({exc}); overlay skipped")
            if title is None:
                continue
            append_kwargs: Dict[str, Any] = {
                "mediaPoolItem": title,
                "startFrame": 0,
                "endFrame": _frames(dur, fps),
                "recordFrame": _frames(start, fps),
            }
            if overlay_track is not None:
                append_kwargs["trackIndex"] = overlay_track
            try:
                ok = timeline.AppendToTimeline([append_kwargs])
                if not ok:
                    log(f"warning: AppendToTimeline failed for overlay {text!r}")
            except Exception as exc:
                log(f"warning: overlay placement failed for {text!r}: {exc}")
            _set_generator_props(title, cx, cy, font, log)

    # ---- 6. music ----
    for audio in edl.audio:
        item = pool_items.get(audio.path)
        if item is None:
            log(f"warning: music file {audio.path} not imported")
            continue
        try:
            timeline.AddTrack("Audio", "Music")
            ok = timeline.AppendToTimeline(
                [{"mediaPoolItem": item, "startFrame": 0, "recordFrame": _frames(audio.timeline_start_sec, fps)}]
            )
            if not ok:
                log(f"warning: could not place music {audio.path}")
        except Exception as exc:
            log(f"warning: music placement failed: {exc}")

    project.SaveProject()
    log(f"built '{timeline_name}' - open it in Resolve for final polish")
    return timeline_name


def _set_generator_props(title, cx: float, cy: float, font_size_rel: float, log: Log) -> None:
    """Best-effort Text+ positioning. Property names vary by Resolve version."""
    if title is None:
        return
    attempts = [
        ("Horizontal Position", str(cx)),
        ("Vertical Position", str(cy)),
        ("Position X", str(cx)),
        ("Position Y", str(cy)),
        ("Size", str(max(1, round(font_size_rel * 100)))),
    ]
    for prop, value in attempts:
        try:
            title.SetClipProperty(prop, value)
        except Exception:
            pass  # unsupported property name in this version


def build_timeline_from_file(edl_path: str, **kwargs) -> str:
    """Bridge entry point for pasting into Resolve's Console."""
    return build_timeline(Edl.load(edl_path), **kwargs)
