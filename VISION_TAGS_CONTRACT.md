# Vision ⇄ Selector contract: `analysis/vision_tags.json`

The integration point between the **vision session** (producer) and the
**selector session** (consumer). Locked so both halves line up.

## File

`analysis/vision_tags.json`, keyed by **shot id** (identical to `shots.json` /
`clips.json` ids, format `sNN_NNN`, e.g. `s00_003`).

```json
{
  "_meta": { "model": "qwen2.5vl:3b", "generated": 1699999999, "n": 3, "refs": ["reze","makima"] },
  "tags": {
    "s00_003": {
      "characters": ["reze"],
      "emotion": "tender",
      "scene_type": "dialogue",
      "description": "Reze smiling at the cafe",
      "confidence": 0.82
    }
  }
}
```

The `tags` map matches the selector's proposal exactly. `_meta` is an additive
sibling the selector can ignore.

## Field vocabularies (closed sets)

| field | type | allowed values |
|---|---|---|
| `characters` | string[] | `denji` `reze` `makima` `aki` `power` `pochita` `crowd` `other` — `[]` if none/unknown |
| `emotion` | string | `tender` `happy` `tense` `action` `fear` `grief` `neutral` |
| `scene_type` | string | `dialogue` `action` `establishing` `reaction` |
| `description` | string | one short caption |
| `confidence` | number | 0..1 — certainty of `emotion`/`scene_type`/`description` |

## Production & reliability (important for the selector)

- `emotion`, `scene_type`, `description`, `confidence` come from a **local VLM**
  (qwen2.5vl:3b, CPU) constrained to the enums above. These are reliable.
- `characters` comes from **reference-image matching**, NOT the VLM's own guess
  (a small VLM can't reliably name franchise characters). Identity is only as
  good as the reference set:
  - Put labeled references at `refs/<id>/*.jpg` (e.g. `refs/reze/1.jpg`).
  - Each shot's keyframe is embedded and cosine-matched to the references; ids
    above the match threshold are emitted.
  - **If no refs exist for a character, it will never appear.** With no `refs/`
    at all, every `characters` is `[]` — the file is still valid, so the
    selector can run against emotion/scene_type immediately and gain identity
    once refs are added. Treat `characters` as best-effort; gate hard decisions
    on its presence, not its absence.
- **Throughput:** ~80 s/shot on this CPU. A full 1221-shot pass is a one-time,
  cached ~27 h. Prefer `--limit`/`--shots` to tag only what the edit needs, or
  set `VISION_MODEL=moondream` for speed.

## Producing it

```bash
# tag the first N shots (test), or specific shots, or all (slow)
python -m pipeline.vision_tags --limit 20
python -m pipeline.vision_tags --shots s00_003,s00_010
python -m pipeline.vision_tags            # all usable shots (slow, cached)
```

Cached by `shot-id + model` — re-runs only tag new/changed shots. The selector
should read the file and tolerate missing shot ids (not yet tagged).
