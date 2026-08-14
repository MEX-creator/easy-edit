# Contributing to EditDNA

Thanks for helping out! This file explains how the project is organized, what
the module boundaries are, and how to test and extend it. If you're new, start
with [README.md](README.md) for the product story, then come back here.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # core + pytest
python -m unittest discover -s tests -v             # stdlib-only test run
```

The test suite runs on **bare Python** — no dependencies required. Tests that
need OpenCV / librosa / ffmpeg detect their imports at load time and skip
themselves with a clear message (see [Testing](#testing)).

CI runs the same commands on every push: `.github/workflows/ci.yml` has a
`stdlib-core` job (Python 3.9–3.13, zero deps) and an `analysis-extras` job
(OpenCV + PySceneDetect + librosa, so the cv2/librosa suites actually run).

---

## Repository layout and module boundaries

```
cli.py                 # argparse CLI; imports core eagerly, everything else lazily
core/                  # shared models, config, errors — ZERO dependencies
analyzer/              # reference video → StyleTemplate, footage → ClipAnalysis
applier/               # StyleTemplate + footage → Edl (plan_edit)
llm/                   # litellm provider abstraction
resolve_driver/        # Edl → DaVinci Resolve API calls
tests/                 # unittest suite (stdlib-compatible)
templates/             # saved style templates (data, not code)
```

### Import graph

```
                 ┌───────────┐
                 │   core/   │   (stdlib only — never import analyzer/applier/llm/resolve_driver)
                 └─────┬─────┘
        ┌──────────────┼───────────────┬───────────────┐
        ▼              ▼               ▼               ▼
     llm/          analyzer/       applier/       resolve_driver/
        │              │               │               │
        └──────┬───────┴───────┐       │               │
               ▼               ▼       │               │
             cli.py ◄──────────┴───────┴───────────────┘
```

Rules that keep this clean:

- **`core/` never imports anything else in the project** and only uses the
  stdlib. Everything else may import `core`.
- **Module-level imports must stay dependency-free.** Every heavy library —
  `scenedetect`, `cv2`, `numpy`, `easyocr`, `librosa`, `litellm`,
  `DaVinciResolveScript` — is imported *inside the function that needs it*.
  This is what makes `editdna demo`, template management, plan previews and
  the stdlib CI job work with nothing installed.
- **Cross-package imports happen lazily too.** `applier/` only imports
  `core` at module level; it reaches into `analyzer.audio_beat_detect` inside
  `_plan_heuristically`. `cli.py` imports `analyzer`/`applier`/`llm`/
  `resolve_driver` inside command handlers.
- **`llm/` is a leaf** — `analyzer` may call `get_llm()`, but `llm/` never
  imports `analyzer`.

### The two data artifacts

Everything flows through two versioned JSON schemas:

1. **`StyleTemplate`** (`core/template.py`, `schema_version: 1`) — the
   reusable description of an edit's DNA. Produced by `analyze_style()`,
   consumed by `plan_edit()`.
2. **`Edl`** (`core/edl.py`, `edl_version: 1`) — the Edit Decision List.
   Produced by `plan_edit()`, consumed by `resolve_driver`.

Version bumps are **breaking**; unknown keys are ignored, missing required
keys raise. When you change a schema, bump the version and add a
`from_dict` migration path.

---

## Dependency policy and the extras matrix

Everything beyond the stdlib is an **optional extra**. The general principle:
the core product must run without a single pip package installed; extras buy
capability, not correctness.

| Extra | Installs | Unlocks | Used by |
|---|---|---|---|
| `cv` | `opencv-python-headless` | transition classification, naive shot-detection fallback, motion sampling | `analyzer/transition_detect.py`, `analyzer/shot_detect.py`, `analyzer/motion_detect.py` |
| `shots` | `scenedetect[opencv]` | real shot/cut detection | `analyzer/shot_detect.py` |
| `ocr` | `easyocr` (+ torch) | on-screen text / caption detection | `analyzer/ocr_overlay_detect.py` |
| `audio` | `librosa`, `numpy` | beat markers, tempo, energy curve | `analyzer/audio_beat_detect.py` |
| `llm` | `litellm` | LLM style summary, LLM-assisted planning | `llm/providers.py` |
| `dev` | `pytest` | running tests under pytest | tests |
| `all` | everything above | — | — |

Notes:

- `shots` and `ocr` pull different OpenCV flavors (`headless` vs full); the
  README's install section covers the conflict.
- `ocr` is the heavyweight extra (torch). CI deliberately does **not** install
  it — it runs `.[cv,shots,audio]` instead.
- Audio decoding of mp4/m4a requires an `ffmpeg` binary on `PATH`
  (`analyzer/audio_beat_detect.py` falls back to it automatically). WAV /
  FLAC / OGG never need it.

Adding a new capability usually means: add the heavy import inside a function,
raise `MissingDependency(feature, extra)` when it's absent, and let the
caller degrade gracefully (the analyzer pipeline warns and skips a stage
rather than failing the whole run).

---

## Design rules

- **Two-lane LLM architecture.** Judgment calls (style summary wording, clip
  selection, caption placement) may go through the LLM; deterministic
  numbers (pacing, beat grid, transition cadence) never do. `get_llm()`
  returns `None` when no provider is configured, and every consumer must fall
  back to heuristics. The EDL `notes` field records which lane ran.
- **The LLM never sees raw video.** Prompts are compact numeric digests
  (`analyzer/style_summarizer.build_compact_context`) and replies must be a
  single JSON object (`llm.providers.complete_json` enforces it).
- **ASCII-safe CLI output.** Printed strings never contain non-ASCII
  characters (Windows consoles), and `cli.main()` reconfigures stdout/stderr
  so unexpected characters can't crash the CLI.
- **Error hierarchy** in `core/errors.py`; everything raises an
  `EditDNAError` subclass so `cli.main()` can print one clean `error: ...`
  line. Raw `OSError`/library exceptions must be wrapped at the boundary.
- **`from __future__ import annotations`** in every module (Python 3.9
  support — the CI matrix runs 3.9–3.13).
- **Data classes over dicts** for anything that crosses a module boundary,
  with explicit `to_dict`/`from_dict` pairs and tolerant parsing.

---

## Testing

Tests are written with `unittest.TestCase` (runs under `pytest` too — no
conversion needed) and live in `tests/`.

**The stdlib rule:** every test file must import and run with zero optional
dependencies installed. Heavy-dependency tests use a module-level
try/import guard plus `@unittest.skipUnless(...)`:

```python
try:
    import numpy as np  # noqa: F401
    import librosa      # noqa: F401
    HAVE_AUDIO_DEPS = True
except ImportError:
    HAVE_AUDIO_DEPS = False

@unittest.skipUnless(HAVE_AUDIO_DEPS, "librosa/numpy not installed")
class BeatDetectTest(unittest.TestCase):
    ...
```

Run both ways:

```bash
python -m unittest discover -s tests -v   # bare stdlib (heavy tests skip)
pip install -e ".[dev]" && pytest         # same tests under pytest
pip install -e ".[cv,shots,audio]"        # + the cv2/librosa/ffmpeg suites
```

What belongs where:

- `test_template.py`, `test_edl.py` — schema round-trips, version guards.
- `test_heuristic_planner.py` — the deterministic planner (pure stdlib).
- `test_beat_synced_edit.py` — integration: real WAV → detected beats → EDL.
- `test_transition_detect.py`, `test_beat_detect.py` — cv2/librosa, skipped
  when absent.
- `test_cli.py` — offline CLI smoke tests (no media, no deps).

### Walking a new feature through the stack

1. **Analyzer stage** (`analyzer/…`) → new fields on `StyleAnalysis`/the
   template. Keep the heavy import lazy; degrade gracefully.
2. **Template** (`core/template.py`) → add the field + `from_dict` default.
3. **Planner** (`applier/edit_decision_engine.py`) → consume it in
   `_plan_heuristically` (or the LLM prompt), keep the template's numbers
   authoritative.
4. **EDL / driver** (`core/edl.py`, `resolve_driver/resolve_bridge.py`) →
   represent and apply it. The Resolve driver is **defensive**: every API
   call is try/except'd and logged, because the scripting surface varies by
   Resolve version and can't be tested without a running Resolve.
5. **CLI** (`cli.py`) → flag + help text (ASCII-only), lazy import.
6. **Tests** — a stdlib test in `test_heuristic_planner.py` (or a gated test
   if it needs cv2/librosa), and a README/CONTRIBUTING touch if behavior is
   user-visible.

---

## PR checklist

- [ ] `python -m unittest discover -s tests` passes on bare Python
- [ ] If cv2/librosa/ffmpeg logic changed: those suites pass too
  (`pip install -e ".[cv,shots,audio]"`, ffmpeg on PATH)
- [ ] `python cli.py --help` and the offline `demo`/`build` flow still work
- [ ] No module-level heavy imports; `core/` untouched by anything else
- [ ] CLI output stays ASCII-safe; errors raise `EditDNAError` subclasses
- [ ] Schema changes bump the version and stay backward-tolerable
- [ ] README updated for user-visible changes
