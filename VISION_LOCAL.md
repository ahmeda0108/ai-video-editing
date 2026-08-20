# Fully Local Vision AI

100% local, zero-cost, general-purpose visual understanding for the AutoEdit
pipeline. Analyze **any** image or video — anime, film, TV, real footage,
gameplay, screen recordings, product shots, photos, animation — with a small
vision-language model running **entirely on your machine**. No API keys, no
subscriptions, no cloud inference, nothing uploaded. Local electricity only.

This is an **addition** to the existing pipeline, not a rewrite: it plugs a real
local VLM into the pipeline's existing `VisionProvider` abstraction, so every
existing capability (analysis, clip library, selection, render, critique/revise
loop) is unchanged — it just gains genuine semantic vision for free.

---

## Architecture

```
                         VisionProvider  (swappable interface)
                                 |
        ┌────────────────────────┼─────────────────────────┐
   LocalProvider           OllamaProvider              AnthropicProvider
   (CV metrics only,   >>  (local VLM via Ollama,      (cloud, optional, paid;
    no VLM, always      *  100% local, zero cost) *     only if you set a key)
    available)
                                 |
                     Ollama HTTP API @ localhost:11434
                                 |
                     small quantized VLM (e.g. qwen2.5vl:3b)
```

Three responsibilities behind the one interface:

1. `analyze_image(path)` — general-purpose structured analysis of one image/frame.
2. `tag_clips(library)` — semantic tags per keyframe for smarter clip selection.
3. `critique(draft, plan, …)` — looks at rendered frames and critiques the edit.

The **Ollama provider** uses only the Python standard library (`urllib`) to talk
to the local Ollama server — it adds **no pip dependency**. The model is chosen
by one env var and is fully swappable.

### Key design decisions (and why)

- **Reuse the existing provider abstraction instead of a new subsystem.** The
  pipeline already had `LocalProvider` (metrics) + `AnthropicProvider` (cloud).
  The only real gap vs. "truly free + local" was a *local VLM*. Adding
  `OllamaProvider` as a peer is the minimal, non-destructive change that delivers
  it — and keeps the optional cloud path for anyone who wants it.
- **Ollama, CPU-first.** It's the most mature zero-config local runtime, runs
  CPU-only via llama.cpp, and pulls quantized models with one command. The
  abstraction means swapping to llama.cpp/another runtime later is a new class,
  not a refactor.
- **Small model by default.** On a laptop-class CPU / iGPU with limited RAM,
  a 2–3B VLM is the accuracy/speed sweet spot. `doctor` sizes the recommendation
  to your actual RAM. We never auto-download an enormous model.
- **Never fall back to cloud.** If local inference can't run, you get a clear
  error with fix steps — never a silent, billable API call.
- **Everything cached by content hash.** Unchanged frames/drafts are never
  re-analyzed (`analysis/cache/*_ollama.json`), so re-runs are cheap.

---

## Supported hardware

| Class | Example | Recommended model | Notes |
|---|---|---|---|
| Low RAM (<8 GB) | old laptop | `moondream` (~1.7 GB) | fastest, lightest |
| **CPU / iGPU, 8–24 GB** | **Ryzen 5700U, 15 GB (this machine)** | **`qwen2.5vl:3b` (~3.2 GB)** | default; use `moondream` if too slow |
| Discrete GPU / ≥24 GB | RTX / 32 GB+ | `qwen2.5vl:7b` (~6 GB) | stronger understanding |

CPU-only works — it's just slower. No CUDA/ROCm required (Ollama uses the GPU
automatically if you have a supported one; otherwise CPU).

---

## Install & setup

```bash
# 0. python deps for the pipeline (numpy/pillow/ffmpeg) — Ollama needs NO python pkg
python -m pip install -r pipeline/requirements.txt

# 1. see what your machine can run + exact next steps
python -m pipeline.run doctor

# 2. install Ollama once (it then runs as a background service)
#    Windows : winget install Ollama.Ollama
#    macOS   : brew install ollama        (or download from ollama.com)
#    Linux   : curl -fsSL https://ollama.com/install.sh | sh

# 3. pull the recommended model + run a real local inference to verify
python -m pipeline.run setup --pull --test
```

`doctor`/`setup` detect your OS/CPU/RAM, confirm ffmpeg (bundled) and Ollama,
list installed models, recommend one sized to your RAM, print the **exact
download size before pulling**, and run a single live inference as a smoke test.

---

## Basic commands

```bash
# general-purpose analysis of ANY image  -> structured JSON (see schema below)
python -m pipeline.run vision path/to/image.jpg

# general-purpose analysis of ANY video  -> per-timestamp structured report
python -m pipeline.run vision path/to/clip.mp4 --every 2 --max-frames 40

# run the full auto-edit critique/revise loop with LOCAL semantic vision
python -m pipeline.run all --profile amv --provider ollama

# force a provider for any subcommand
python -m pipeline.run analyze --provider ollama
```

Outputs for `vision` land in `analysis/vision/`. Everything else is unchanged
from `AUTOEDIT.md`.

---

## Configuration

All in `.env` (copy from `.env.example`) or the environment — nothing is
hard-coded in source:

| Var | Default | Meaning |
|---|---|---|
| `VISION_PROVIDER` | `auto` | `auto` \| `ollama` \| `anthropic` \| `local` |
| `VISION_MODEL` | `qwen2.5vl:3b` | any Ollama vision model tag |
| `OLLAMA_HOST` | `http://localhost:11434` | local Ollama endpoint |
| `ANTHROPIC_API_KEY` | *(empty)* | optional; enables the cloud provider |

`auto` resolves to: **anthropic** if a key is set, else **ollama** if a local
server is reachable, else **local** metrics-only. So with Ollama installed and
no key, you are fully local and free by default.

Analysis knobs (sampling, resolution, thresholds, cache dir) live in
`pipeline/config.py` and are all env-overridable — see `AUTOEDIT.md`.

---

## Structured output schema (`vision`)

Media-agnostic; every field may be `"unknown"` / `null` / `[]` rather than a
guess. `observed` vs `inferred` keeps the model honest.

```json
{
  "timestamp": 12.4,
  "scene": "…",
  "subjects": [], "people_count": 1,
  "actions": [], "objects": [], "emotion": [], "clothing": [],
  "environment": "", "setting_kind": "anime|film|game|photo|…|unknown",
  "camera": { "shot_type": "", "movement": "", "angle": "" },
  "composition": "", "lighting": "", "color_palette": [],
  "mood": "", "text": "(OCR)",
  "observed": [], "inferred": [],
  "confidence": 0.0
}
```

For the **edit pipeline**, `tag_clips` emits compact per-clip tags
(`subject/setting/camera/mood/action_type/composition/is_impact`) that feed clip
selection, and `critique` emits the same timestamped-issue schema the metrics
critic uses — so local vision merges seamlessly into the existing revise loop.

---

## Caching

Per-provider caches under `analysis/cache/`:
`analyze_ollama.json`, `tags_ollama.json`, `critique_ollama.json`. Keys embed the
frame/draft content signature **and the model name**, so switching models or
changing footage invalidates only what changed. Nothing is ever re-inferred
unnecessarily.

---

## Remotion / editing integration

Local vision exposes exactly the metadata a Remotion edit consumes — shot
intensity, emotion, subject, action, camera movement, composition, importance,
timestamp — through the same `clips.json` tags and `edit_plan.vN.json` contract
the pipeline already uses. The critique/revise loop (`iterate` / `all`) now runs
with **local** semantic critique: every iteration is rendered and its frames are
judged by the local model, then the plan is revised. Music/beat analysis
(`audio.py`) is untouched and still drives beat-synced cutting.

---

## Replacing / adding models & providers

**Swap the model** — no code change:
```bash
echo "VISION_MODEL=moondream" >> .env      # or qwen2.5vl:7b, llava-phi3, …
ollama pull moondream
```

**Add another local runtime** (e.g. llama.cpp server, LM Studio): subclass
`VisionProvider` in `pipeline/vision_provider.py`, implement `analyze_image`,
`tag_clips`, `critique` (reuse `sample_draft_frames`, `GENERAL_SCHEMA`,
`CRITIQUE_SCHEMA`), set `semantic = True`, and register it in `get_provider()`.
Nothing else in the pipeline needs to change — that's the whole point of the
abstraction.

---

## Character identity (reference-image matching)

Named characters in `analysis/vision_tags.json` come from **matching against your
own reference images**, not from the VLM guessing — a small local model can't
reliably name franchise characters. Fully local, zero cost, and it degrades
gracefully (no refs → `characters: []`, file still valid).

```
refs/
  reze/    1.jpg 2.jpg ...      # a few clear shots of each character
  makima/  1.jpg ...
  _model/  clip_image.onnx      # a CLIP-style ONNX image encoder (CPU)
```

- Ids must be from the contract vocab: `denji reze makima aki power pochita`.
- Embedding backend is ONNX CLIP via `onnxruntime` (CPU, no torch):
  `pip install onnxruntime`, then drop a CLIP image-encoder ONNX at
  `refs/_model/clip_image.onnx`. Match threshold is `REF_THRESHOLD` (default 0.75).
- Check readiness: `python -c "from pipeline import refmatch, json; print(json.dumps(refmatch.status(), indent=2))"`

## Producing `vision_tags.json` (selector contract)

A standalone producer (see [`VISION_TAGS_CONTRACT.md`](./VISION_TAGS_CONTRACT.md))
that the edit **selector** consumes. Run it directly — it does **not** go through
`run.py`, so it's safe to run alongside other sessions editing the CLI:

```bash
python -m pipeline.vision_tags --limit 20          # first 20 shots (test)
python -m pipeline.vision_tags --shots s00_003,s00_010
python -m pipeline.vision_tags                      # all shots (slow, cached)
```

Each shot gets `emotion` / `scene_type` / `description` / `confidence` from the
local VLM (enum-constrained) plus `characters` from reference matching. Cached by
shot-id + model; writes incrementally so a long run yields a valid, growing file.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach local Ollama…` | Install Ollama / start it: `ollama serve`. `python -m pipeline.run doctor`. |
| `Ollama model '…' is not installed` | `ollama pull <model>` or `python -m pipeline.run setup --pull`. |
| Tagging is very slow | Expected on CPU for many keyframes (it's cached + incremental). Use `moondream`, or rely on local vision mainly for the 24-frame-per-iteration **critique** + on-demand `vision`, not bulk tagging. |
| Model returns non-JSON | Handled: output is tolerant-parsed; on failure you get `confidence: 0.0` and a note, never a crash. |
| Want cloud quality instead | Set `ANTHROPIC_API_KEY` in `.env` (optional, paid). Local stays the default when no key is set. |

**Privacy:** with the Ollama provider, no image, frame, video, or metadata ever
leaves the machine.
