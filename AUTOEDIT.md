# AutoEdit — autonomous visual-model video editor

Give it **raw footage** (a file or a folder) and a **music track**; it analyzes
both, builds a searchable clip library, edits a beat-synced video in Remotion,
renders it, critiques the result, revises the edit, and repeats until the edit
stops improving.

It is a **general visual-model editing pipeline**, not an AMV-only tool. "AMV"
is just one *profile* (`amv` / `hype` / `montage` — add your own); the core
(ingest → analysis → library → edit-plan → render → critique → revise) is
edit-type-agnostic.

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

## Required API keys / services

| Thing | Needed? | Notes |
|---|---|---|
| **None** (default) | — | The whole loop runs locally with the metrics critic + local CV tagging. |
| `ANTHROPIC_API_KEY` | optional | Upgrades tagging + critique to true multimodal understanding. Copy `.env.example` → `.env` and set it. `ANTHROPIC_MODEL` defaults to `claude-opus-4-8` (set `claude-haiku-4-5` / `claude-sonnet-5` to cut cost). |

No other external services. Provide a key only to unlock semantic vision; the
pipeline never *requires* one.

---

## Local vs remote

| Runs **locally** (no network) | Runs **remotely** (only with a key) |
|---|---|
| ffmpeg decode / scene detection / keyframes (bundled binary) | Semantic keyframe tagging (subject/setting/camera/mood) |
| motion + brightness analysis (numpy) | Rendered-draft critique by a vision model |
| audio tempo/beats/energy/sections/drops (numpy) | |
| pHash + color embeddings, dedup clustering | |
| clip library, selection, edit-plan generation | |
| Remotion render | |
| **metrics critic** (score + timestamped issues) + revise loop | |

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
