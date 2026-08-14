# EditDNA

[![CI](https://github.com/MEX-creator/easy-edit/actions/workflows/ci.yml/badge.svg)](https://github.com/MEX-creator/easy-edit/actions/workflows/ci.yml)

**Clone the edit, not the footage.**

EditDNA watches a reference video (a viral Reel / YouTube short), extracts its
editing **DNA** — cut pacing, transition types, on-screen text/caption style,
music beat-sync — and saves it as a reusable, human-readable **style template**.
It then applies that template to *your* raw footage and drives
**DaVinci Resolve (Free)** via its scripting API to assemble the timeline
automatically. You open Resolve, the edit is already built, and you do the
final human polish and export.

Works with **any LLM you bring** (Gemini free tier by default, OpenAI,
Anthropic, or a fully local Ollama), and every heavy analysis step runs
locally with free open-source libraries — no paid API is required to use the
tool at all.

---

## How it works

```
┌───────────────────────┐    ┌──────────────────────────┐    ┌───────────────────┐
│  Reference video       │    │  Your raw clips + script │    │  DaVinci Resolve   │
│  (viral Reel/short)    │    │  (+ optional music)      │    │  (Free)            │
└───────────┬───────────┘    └────────────┬─────────────┘    └─────────┬─────────┘
            │                             │                             ▲
            ▼                             ▼                             │
┌───────────────────────┐    ┌──────────────────────────┐              │
│  analyzer/             │    │  applier/                 │              │
│  analyze_style()       │    │  plan_edit()              │              │
│  • PySceneDetect cuts  │    │  • template pacing +      │              │
│  • transition classify │    │    beat grid (determin.)  │              │
│  • OCR text/captions   │    │  • LLM judgment calls     │              │
│  • librosa beats/energy│    │    (or heuristic planner) │              │
└───────────┬───────────┘    └────────────┬─────────────┘              │
            │                             │                             │
            ▼                             ▼                             │
┌────────────────────────────┐   ┌──────────────────┐   ┌──────────────┘
│  StyleTemplate (JSON)      │──▶│  EDL (JSON)      │──▶│  resolve_driver/
│  the reusable artifact     │   │  edit plan       │   │  resolve_bridge.py
└────────────────────────────┘   └──────────────────┘   └───────────────
```

Two analysis passes are deliberately distinct (they extract different things):

- **`analyze_style()`** — reference video → `StyleTemplate` (*what the edit
  looks like*: pacing, transitions, text, music sync).
- **`analyze_content()`** — your footage → `ClipAnalysis` (*what the footage
  contains*: duration, scene changes, motion, audio peaks).

---

## Features (v1 scope)

- **Reference → template**: shot detection (PySceneDetect), transition
  classification (hard cut / crossfade / whip pan / zoom cut / dip to black),
  OCR text + caption detection with temporal merging (EasyOCR), beat/energy
  analysis (librosa), and an optional LLM "vibe" summary — all into a
  versioned JSON template.
- **Template management**: save / load / list / inspect templates; edit them
  by hand; share them as plain JSON.
- **Footage → EDL**: deterministic cutting — scene-aware in pacing mode
  (cuts slide to real scene changes near the template's pacing grid),
  beat-locked in beat-synced mode — plus template-driven transitions, and
  sentence captions + text overlays from your script. Optional LLM pass for
  the judgment calls (clip order, in/out points, caption placement) — its
  output is *normalized* against the template so the template's pacing
  always wins.
- **Resolve driver**: imports media, builds the timeline, trims clips to
  in/out points, adds transitions, places Text+ overlays/captions and the
  music track — with a console-paste bridge for Resolve's own Python console.
- **Pluggable LLM layer** (litellm): Gemini as the free-tier default, any
  user key (OpenAI / Anthropic / OpenAI-compatible), or local Ollama.
  No key? Everything still works via deterministic heuristics.

Explicitly deferred to v2: community template sharing, auto color-matching,
complex Fusion compositing, auto b-roll, word-level karaoke captions.

---

## Installation

Requires **Python 3.9+**. The core (`demo`, `templates`, `apply --no-llm`,
`build` plan preview) runs with zero dependencies. Everything else is an
optional extra so you only install what you need:

```bash
git clone <repo-url> && cd editdna
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Everything (shot detection + transitions + OCR + audio + LLM):
pip install -e ".[all]"

# ...or just what you need:
pip install -e ".[shots,audio,llm]"   # analysis without OCR
pip install -e ".[ocr]"               # + caption/text OCR (heavy model download)
pip install -e ".[llm]"               # LLM style summary / LLM planning
pip install -e ".[dev]"               # run the test suite
```

> **Note on OpenCV**: `editdna[shots]` and `editdna[ocr]` pull different
> OpenCV flavors (`opencv-python-headless` vs `opencv-python`). If you install
> the `ocr` extra after `shots`, pip will upgrade to the non-headless build,
> which is fine. If you hit import conflicts, install `opencv-python` instead
> of `opencv-python-headless` manually.

> **Audio decoding**: beat analysis (`librosa`) reads WAV/FLAC/OGG out of the
> box, but video containers (mp4/m4a/aac) need **ffmpeg on your PATH** —
> EditDNA falls back to `ffmpeg` to extract the audio automatically. Without
> it, audio-based fields (`music_sync`, `beat_synced`) are skipped with a
> warning and the rest of the analysis still runs.

### DaVinci Resolve setup (one-time)

1. Install DaVinci Resolve (Free is fine).
2. Enable scripting: **Preferences → System → General → "External scripting
   using" → Local** (or Network), then restart Resolve.
3. Launch Resolve and keep it running while you use `editdna build --execute`.

The driver also works from inside Resolve itself — paste this into
**Workspace → Console** (Resolve's built-in Python has `DaVinciResolveScript`
already importable):

```python
from resolve_driver.resolve_bridge import build_timeline_from_file
build_timeline_from_file(r"path\to\edit_plan.json", project_name="My Edit")
```

---

## Quickstart

### 0. Offline demo (no media, no deps)

```bash
editdna demo --out-dir .
editdna templates show sample_fast_hookline_reel.json
editdna build demo_edl.json          # preview the plan table
editdna demo-apply                   # plan a real edit on placeholder clips
```

### 1. Analyze a reference video → style template

```bash
editdna analyze reference.mp4                    # saves to ~/.editdna/templates/
editdna analyze reference.mp4 --out my-style.json --no-ocr --no-llm
```

The analyzer prints what it found (pacing, transitions, text, music sync) and
saves a `StyleTemplate` JSON. OCR is noisy on busy backgrounds — review the
template (`editdna templates show <name>`) and correct it by hand before
sharing it.

### 2. Inspect your own footage

```bash
editdna content clip1.mp4            # duration, scene changes, motion, peaks
```

This is optional — `apply` auto-analyzes each clip, but a manual pass is a
good sanity check (and `--durations` is the fallback if no analysis backend
is installed).

### 3. Apply the template to your footage → EDL

```bash
editdna apply my-style.json clip1.mp4 clip2.mp4 clip3.mp4 \
    --script voiceover.txt --music beat.mp3 --out edit_plan.json
```

`edit_plan.json` is the **Edit Decision List** — inspect it, hand-tune it if
you like, and commit it.

### 4. Build it in Resolve

```bash
editdna build edit_plan.json          # preview the plan
editdna build edit_plan.json --execute --project "My Reel"
```

Open Resolve — the timeline is assembled (clips trimmed, transitions in,
Text+ overlays, music). Polish and export. Remember Resolve Free can't export
H.265/ProRes on some platforms — check your delivery settings at export.

---

## LLM configuration

Copy `config.example.env` to `.env` and set what you need. Everything is
optional — with no key configured, EditDNA runs fully deterministic.

| Variable | Default | Notes |
|---|---|---|
| `EDITDNA_LLM_PROVIDER` | `gemini` | `gemini` \| `openai` \| `anthropic` \| `ollama` \| `openai_compatible` |
| `EDITDNA_API_KEY` | — | Key for the chosen provider (Gemini: aistudio.google.com/apikey) |
| `EDITDNA_MODEL` | provider default | e.g. `gemini/gemini-2.5-flash`, `openai/gpt-4o-mini` |
| `EDITDNA_OPENAI_BASE_URL` | — | Custom OpenAI-compatible endpoint (OpenRouter, vLLM, LM Studio) |
| `EDITDNA_OLLAMA_BASE_URL` | `http://localhost:11434` | For local Ollama |
| `EDITDNA_OLLAMA_MODEL` | `llama3.1` | For local Ollama |
| `EDITDNA_OCR_LANGUAGES` | `en` | Comma-separated EasyOCR languages |
| `EDITDNA_OCR_MIN_CONFIDENCE` | `0.5` | Drop OCR detections below this confidence |
| `EDITDNA_TEMPLATES_DIR` | `./templates` → `~/.editdna/templates` | Where templates are saved/loaded |

The LLM only ever sees compact numeric digests (cut timestamps, overlay
lists, beat stats) — never raw video — so free tiers go a long way. If the
LLM fails (no key, network, quota), planning silently falls back to the
deterministic planner; the EDL notes field tells you which path was used.

---

## Style template schema (v1)

```json
{
  "template_name": "fast-hookline-reel",
  "schema_version": 1,
  "source": "reference_video.mp4",
  "created_at": "2026-01-01T00:00:00Z",
  "pacing": {
    "avg_shot_duration_sec": 1.2,
    "cut_style": "hard_cut",
    "beat_synced": true,
    "cuts_per_10s": 8
  },
  "transitions": [
    {"type": "whip_pan", "frequency": "every_3rd_cut"}
  ],
  "text_overlays": {
    "style": "bold_center_pop_in",
    "font_weight": "heavy",
    "position": "center",
    "avg_words_per_overlay": 4,
    "appears_on_beat": true
  },
  "captions": {
    "present": true,
    "style": "sentence_lower_third",
    "position": "lower_third"
  },
  "music_sync": {
    "cuts_aligned_to_beats": true,
    "energy_curve": "build_to_drop"
  },
  "llm_style_summary": "Fast punchy hookline-style reel edit, cuts land on beat, bold pop-in text for key phrases, whip-pan transitions used sparingly for emphasis on payoff moments."
}
```

Versioned (`schema_version`), human-readable, hand-editable, shareable. See
`templates/sample_fast_hookline_reel.json` for a real example.

## EDL schema (v1)

The intermediate plan `apply` produces and `build` consumes:

```json
{
  "name": "fast-hookline-reel",
  "edl_version": 1,
  "fps": 30.0,
  "items": [
    {
      "id": "v1",
      "source_path": "clip1.mp4",
      "track": 1,
      "in_sec": 0.0,
      "out_sec": 1.2,
      "timeline_start_sec": 0.0,
      "duration_sec": 1.2,
      "transition_after": "whip_pan",
      "transition_duration_sec": 0.6
    }
  ],
  "text_overlays": [{"id": "t1", "text": "The hook", "timeline_start_sec": 0.0, "duration_sec": 1.2, "cx": 0.5, "cy": 0.35, "font_size_rel": 0.07, "style": "bold_center_pop_in"}],
  "captions": [{"id": "c1", "text": "The hook.", "timeline_start_sec": 0.0, "duration_sec": 1.2, "cx": 0.5, "cy": 0.86}],
  "audio": [{"path": "beat.mp3", "timeline_start_sec": 0.0, "track": "A2", "volume_db": 0.0}],
  "notes": []
}
```

---

## Repository layout

```
├── cli.py                    # argparse CLI (editdna <command>)
├── core/                     # shared models & config (zero-dependency)
│   ├── template.py           # StyleTemplate schema + save/load
│   ├── edl.py                # Edit Decision List schema + save/load
│   ├── analysis.py           # Shot/Transition/TextOverlay/ClipAnalysis ...
│   ├── config.py             # .env loader + Settings
│   ├── demo.py               # offline demo artifacts
│   └── errors.py             # exception hierarchy
├── analyzer/                 # reference video → style template
│   ├── pipeline.py           # analyze_style() / analyze_content()
│   ├── shot_detect.py        # PySceneDetect (+ OpenCV fallback)
│   ├── transition_detect.py  # cut classification (hard/crossfade/whip/zoom/dip)
│   ├── ocr_overlay_detect.py # EasyOCR text/caption detection + temporal merge
│   ├── audio_beat_detect.py  # librosa beats + energy curve
│   ├── motion_detect.py      # coarse motion sampling for user footage
│   └── style_summarizer.py   # LLM "vibe" summary (heuristic fallback)
├── applier/
│   └── edit_decision_engine.py  # template + footage → EDL (LLM + heuristic)
├── resolve_driver/
│   └── resolve_bridge.py     # EDL → DaVinci Resolve API calls
├── llm/
│   └── providers.py          # litellm abstraction (gemini/openai/anthropic/ollama)
├── templates/                # saved style templates
├── tests/                    # unittest suite (runs with plain stdlib)
├── config.example.env
└── pyproject.toml
```

---

## Design decisions & known limitations

- **Shot detection is a solved local problem** — it never goes through the
  LLM. `PySceneDetect` is the backbone; the OpenCV fallback is lower quality
  (documented in `analyzer/shot_detect.py`).
- **Transition classification is best-effort.** It's a lightweight CV
  heuristic (frame deltas + phase-correlation motion + scale check) and can
  confuse fast whip cuts with hard cuts on busy footage. Confidence is
  reported per transition; review the template before sharing.
- **OCR caption detection is noisy** on busy backgrounds — a confidence
  threshold drops junk, and results are meant to be reviewed (and hand-fixed)
  in the saved template.
- **The heuristic planner is scene-aware.** In pacing mode (no music) cuts
  slide up to ±40% of the template's shot duration (configurable
  `pacing_snap_band`) to land on real scene changes in your footage, falling
  back to the pacing grid when none are nearby — so the template's pacing is
  preserved. In beat-synced mode the beat grid is the DNA and stays locked
  exactly; source ranges are fitted continuously to the beat slots with no
  skipping. Pass `--no-scene-snap` for a pure pacing grid. `Edl.validate()`
  checks the plan for gaps/overlaps.
- **Resolve Free limits are designed around, not hidden:** no Neural Engine
  features (auto-caption, speed warp) — exactly the gap this tool fills; no
  H.265/ProRes export on some platforms — flagged at export time in the
  README, never silently failed by the driver.
- **The Resolve driver is defensive** (every call try/except'd + logged)
  because the scripting surface varies by Resolve version. Known soft spots:
  `TIMELINEPOS` trimming and Text+ property names may not exist on older
  builds — you'll see warnings, and the timeline may need minor manual
  positioning. Precise animated Fusion titles are a v2 stretch goal.
- **LLM prompts are strictly JSON-in/JSON-out.** The model never receives
  raw video; it sees compact digests and returns structured JSON that is
  normalized against the template (pacing/beat-sync/transitions always come
  from the template, never re-invented by the model).

---

## Roadmap (v2+)

- Community template sharing / marketplace
- Auto color-grading matching from the reference
- Multi-track Fusion compositing; animated Text+ / title templates
- Word-level karaoke captions (started with sentence-level in v1)
- Auto b-roll suggestion / stock footage pull
- Socket/file-watch bridge to Resolve (so the app can trigger builds without
  you running the CLI)

## Development

The test suite runs on plain stdlib (`unittest`), with heavy-dependency tests
skipped automatically when their libraries aren't installed:

```bash
python -m unittest discover -s tests -v     # stdlib-only run
pip install -e ".[dev]" && pytest           # full run incl. cv2/librosa tests
```

## License

MIT — see [LICENSE](LICENSE).
