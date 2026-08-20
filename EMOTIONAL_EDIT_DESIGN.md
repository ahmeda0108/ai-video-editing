# Emotional-arc editing — selector-side design

The evolution from "energy/motion matching" to **emotion-driven** editing, per
feedback: edits felt random, cut too fast, showed irrelevant characters, and
carried unaligned text. This spec covers the **selector session's half**; the
**vision session** produces `analysis/vision_tags.json` (see
`VISION_TAGS_CONTRACT.md`).

## Goal

Given a movie (with subtitles) + a song + a target character set + a target
emotional arc, produce an edit that **conveys emotion** by aligning the story's
emotional beats with the music's, focusing on specific characters, cutting
slowly enough to breathe, and rendering smoothly.

## Inputs (all available)

- `analysis/shots.json`, `clips.json` — shot library (source times, motion).
- Embedded subtitles: track 3 = dialogue, track 4 = SDH (dialogue + `[sound]`
  cues). Extracted to `analysis/transcript.json`.
- `analysis/vision_tags.json` — per-shot `{characters, emotion, scene_type,
  description, confidence}`, **partial and best-effort** (missing id = untagged;
  `characters` may be `[]`). Consumed, never required.
- Song analysis (`audio_*.json`), synced lyrics (fetched to `analysis/lyrics/`).

## Components (this session)

### 1. `pipeline/transcript.py`
Extract SRT tracks via ffmpeg; parse to `[{start, end, text, kind}]` where
`kind ∈ {dialogue, sound_cue}`. Sound cues (`[explosion]`, `[scream]`) become
candidate diegetic-audio hits. Cached to `analysis/transcript.json`.

### 2. `pipeline/story.py`
Build an **emotional arc** = ordered acts, each a movie time-range + target
emotion + character focus. Signal priority (robust to sparse vision data):
1. **Transcript** (always present): dialogue density/tone + sound-cue density
   segment the film into calm-romance / action / aftermath regions; character
   name mentions hint at character-present ranges.
2. **Vision tags** (when present): `emotion`/`scene_type` refine act boundaries;
   `characters` filters to the target set. Untagged shots fall back to motion
   heuristics, never dropped for lack of a tag.
For the Reze arc the default arc is **happy (Reze+Denji) → fight → aftermath**,
configurable per edit.

### 3. Emotional-arc selector (`select.py`, new `arc` mode)
- Map arc acts onto song structure: intro/build → act 1, drop/hook → act 2,
  outro → act 3. The biggest musical hit gets the biggest emotional beat.
- Within an act, pick shots from its time-range, scored by: emotion match
  (vision) ▸ character presence (vision, bonus) ▸ quality ▸ shot variety ▸
  transcript relevance. Absent tags → degrade to motion/energy scoring.
- **Fewer, longer cuts**; slow-mo only where it won't judder.

### 4. Lyric underlays (aligned)
Fetch synced lyrics; place faint underlays on the **actual sung word/line**.

### 5. Smoothness (render)
- Render at **24 fps** (matches source 23.97) → removes pulldown judder.
- Drop/soften slow-mo (the juddery culprit); optional ffmpeg `minterpolate`
  frame-blending only on deliberately-slowed clips.

## Robustness contract

- `vision_tags.json` absent → full transcript+heuristic edit (no crash).
- Shot id missing from tags → "not analyzed yet," use fallback scoring.
- `characters == []` → skip the identity filter for that shot; rely on emotion.

## Reusability

`story.py` + arc selector are generic: any movie-with-subs + song + character
set + arc. Per-edit drivers (`come_home_edit.py`, etc.) just supply those.

## Out of scope (vision session owns)

`pipeline/vision_tags.py`, `vision_provider.py`, Ollama/config, `refs/`.

## Open items

- Song window/length for the Come Home edit (default: ~90s, happy→fight→aftermath).
- Lyric source (web-searched LRC).
