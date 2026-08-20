# AutoEdit — autonomous visual-model video editor

Give it **raw footage** (a file or a folder) and a **music track**; it analyzes
both, builds a searchable clip library, edits a beat-synced video in Remotion,
renders it, critiques the result, revises the edit, and repeats until the edit
stops improving.

It is a **general visual-model editing pipeline**, not an AMV-only tool. A
*profile* just parametrises the generic engine (cut density, transitions,
grade, pacing, audio behaviour), and a *driver* pins a concrete edit (song +
window + optional captions). The shipped profiles — `amv`/`hype`/`montage`
(beat-cut), `story` (slow emotional), `bomb` (hard on-beat hype), `action_arc`
(vision-aware arc) — are **examples, not limits**: mix the knobs freely or add
your own to express any style. The core (ingest → analysis → library →
edit-plan → render → critique → revise) is **edit-type-agnostic** — nothing
about the footage, song, character focus, or style is baked in.

---

## Architecture

```
 raw footage ─┐                      ┌── clips.json (searchable library) ──┐
              │  motion.py           │   quality · intensity · impact       │
              ├─ ingest.py  (shots,  │   cluster · palette · tags · usable  │
              │   adaptive keyframes)│                                      │
              ├─ visual.py  (pHash + │                                      ▼
              │   color embeddings,  │                            select.py + profiles.py
              │   dedup clusters)    │                            (music structure ─► cut grid
 music track ─┴─ audio.py  (tempo,   │                             ─► clip choices ─► edit plan)
                 beats, energy,      │                                      │
                 sections, drops) ───┘                                      ▼
                                                            edits/edit_plan.vN.json  (the contract)
                                                                            │
                                                                 render.py  ▼   (Remotion CLI)
                                                            src/AutoEdit/AutoEdit.tsx
                                                                 renders plan ─► out/draft.vN.mp4
                                                                            │
                                                        vision_provider.py  ▼
                                              critique (local metrics  OR  Claude multimodal)
                                                                            │
                                                            critic_metrics.py + revise.py
                                                       structured, timestamped issues ─► next plan
                                                                            │
                                                              loop until improvement is negligible
```

**Design choices**
- **No single model does everything.** ffmpeg decodes/detects; numpy does DSP +
  embeddings; Remotion renders deterministically; a vision model (optional) does
  semantic understanding + critique; the pipeline orchestrates.
- **The edit plan (JSON) is the contract.** Python decides *what to cut*;
  Remotion is a pure, deterministic *renderer* of that decision. Critique
  revises the **plan**, never blindly rewriting the project.
- **Everything is cached and inspectable.** Each stage writes JSON under
  `analysis/` / `edits/` keyed by an input hash; unchanged footage is never
  re-analyzed or re-sent to an API.
- **Fast fights are preserved.** Motion is measured on a 12 fps full-video pass,
  but keyframes for high-motion shots are extracted at **motion peaks (impact
  moments)**, not by uniform low-fps sampling — so impact frames survive.

---

## Commands

```bash
# 0. one-time: python deps for the pipeline
python -m pip install -r pipeline/requirements.txt
# (Node deps already installed for Remotion; ffmpeg is bundled via imageio-ffmpeg)

# 0b. one-time: set up FREE LOCAL vision (see VISION_LOCAL.md)
python -m pipeline.run doctor                 # hardware + prereq check + advice
python -m pipeline.run setup --pull --test    # after installing Ollama: pull model + verify

# general-purpose LOCAL analysis of ANY image or video (not AMV-specific)
python -m pipeline.run vision path/to/image.jpg
python -m pipeline.run vision path/to/clip.mp4 --every 2 --max-frames 40

# 1. analyze raw footage + music -> clip library (cached)
python -m pipeline.run analyze                      # uses public/footage.mp4 + public/track_v1.wav
python -m pipeline.run analyze /path/to/footage_folder --track /path/to/song.wav

# 2. build an initial edit plan (no render)
python -m pipeline.run plan --profile amv           # amv | hype | montage

# 3. render a specific plan
python -m pipeline.run render edits/edit_plan.v1.json

# 4. the full AI critique/revise loop (initial edit -> critique -> revise -> ... -> final)
python -m pipeline.run iterate --profile amv --iters 4

# everything end to end
python -m pipeline.run all --profile amv --iters 4

# interactive preview of any generated plan
npx remotion studio        # open the "AutoEdit" composition
```

Outputs:
- `out/draft.v1.mp4` — the initial edit
- `out/final.mp4` — the converged edit
- `edits/edit_plan.vN.json`, `edits/critique.vN.json`, `edits/history.json`
- `analysis/*.json`, `analysis/keyframes/*.jpg` — all intermediate data

---

## Vision providers (semantic understanding)

Semantic vision (tagging + rendered-draft critique + general image/video
analysis) has **three** interchangeable providers behind one abstraction — see
[`VISION_LOCAL.md`](./VISION_LOCAL.md) for the full local setup.

| Provider | Needed? | Notes |
|---|---|---|
| `local` | — | Metrics-only critic + CV tagging. No VLM, but the whole loop runs with zero setup. |
| **`ollama`** | **recommended** | **100% local, zero-cost VLM.** Install Ollama + a small model (`python -m pipeline.run setup --pull --test`). No key, nothing uploaded. This is the free path to *real* scene understanding. |
| `anthropic` | optional | Cloud multimodal via `ANTHROPIC_API_KEY` (paid). `ANTHROPIC_MODEL` defaults to `claude-opus-4-8`. Only used if you set a key. |

`VISION_PROVIDER=auto` picks: anthropic if a key is set, else **ollama** if a
local server is reachable, else `local`. So with Ollama installed and no key, you
get free local semantic vision by default. The pipeline **never** falls back to a
cloud API silently.

---

## Local vs remote

| Runs **locally** (no network) | Runs **remotely** (only if you set a key) |
|---|---|
| ffmpeg decode / scene detection / keyframes (bundled binary) | *(optional)* Anthropic tagging/critique — only when `ANTHROPIC_API_KEY` is set |
| motion + brightness analysis (numpy) | |
| audio tempo/beats/energy/sections/drops (numpy) | |
| pHash + color embeddings, dedup clustering | |
| clip library, selection, edit-plan generation | |
| Remotion render | |
| **metrics critic** (score + timestamped issues) + revise loop | |
| **semantic keyframe tagging + rendered-draft critique via local VLM (Ollama)** | |
| **general-purpose image/video analysis (`vision` command, local VLM)** | |

With the Ollama provider, semantic vision is **fully local** — the right column
is used only if you deliberately opt into the paid cloud provider.

---

## Known limitations

- **Local critic is metric-based**, not semantic: it reasons about pacing,
  repetition, intensity-vs-energy, drop emphasis, variety and wasted footage —
  but it can't tell you "the character's face is cut off" or "this is the wrong
  emotional beat." Add an `ANTHROPIC_API_KEY` for that.
- **Render is CPU-bound.** `OffthreadVideo` re-decodes the source per frame;
  a 33s 1080p edit takes a few minutes. With a vision key, each iteration is
  rendered, so runs are longer.
- **Beat/drop detection is heuristic** (numpy DSP, no librosa). Solid on clear
  electronic/pop tracks; sparse or rubato music may need threshold tuning in
  `pipeline/config.py`.
- **pHash dedup can over-merge flat frames** (near-black/near-white); the
  usability gate filters those out of selection, so it doesn't affect the edit.
- **Transitions** are per-clip entrance animations + crossfade overlaps; there
  is no motion-tracked morph / speed-ramp curve editor yet.
- **Single source resolution assumed per project** for framing; mixing very
  different aspect ratios in one folder may letterbox unevenly.
```
