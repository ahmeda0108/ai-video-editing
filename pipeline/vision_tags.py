"""
Vision-tags producer -> analysis/vision_tags.json  (see VISION_TAGS_CONTRACT.md).

Standalone on purpose: run as `python -m pipeline.vision_tags` so it never
touches run.py / config.py (which a concurrent editing session may also change).
The selector only reads the JSON, so it doesn't care how it's produced.

Per shot:
  emotion / scene_type / description / confidence  <- local VLM (enum-constrained)
  characters                                       <- reference-image matching

Reliable, cached by (shot-id + model). Writes incrementally so a slow/partial
run still yields a valid, growing file the selector can consume live.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config
from . import refmatch
from .io_util import load_json, save_json, sha1, file_sig
from .vision_provider import OllamaProvider

EMOTIONS = ["tender", "happy", "tense", "action", "fear", "grief", "neutral"]
SCENE_TYPES = ["dialogue", "action", "establishing", "reaction"]

OUT_PATH = config.ANALYSIS / "vision_tags.json"
CACHE_PATH = config.CACHE / "vision_tags_cache.json"

_SCHEMA = {
    "type": "object",
    "properties": {
        "emotion": {"type": "string", "enum": EMOTIONS},
        "scene_type": {"type": "string", "enum": SCENE_TYPES},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["emotion", "scene_type", "description", "confidence"],
}

_PROMPT = (
    "Analyze this single frame from an animated video for an editor. Reply as "
    "JSON with EXACTLY these fields:\n"
    f"  emotion: one of {EMOTIONS}\n"
    f"  scene_type: one of {SCENE_TYPES} "
    "(dialogue=characters talking/close faces; action=fighting/fast motion; "
    "establishing=wide scene/location; reaction=a character reacting)\n"
    "  description: one short sentence of what is happening\n"
    "  confidence: 0..1, your certainty.\n"
    "Pick the closest allowed value; do not invent new ones. Do NOT name any "
    "characters in the description unless a name is written on screen."
)


def _coerce(val: str, allowed: list[str], default: str) -> str:
    if not isinstance(val, str):
        return default
    v = val.strip().lower()
    if v in allowed:
        return v
    for a in allowed:                       # tolerate "action-packed" -> "action"
        if a in v:
            return a
    return default


def _keyframe_for(shot: dict) -> Path | None:
    """Middle keyframe for a shot, derived from src_idx+index (robust to clips.json churn)."""
    src = int(shot.get("src_idx", 0)); idx = int(shot.get("index", 0))
    cands = sorted(config.KEYFRAMES.glob(f"src{src:02d}_shot{idx:03d}_*.jpg"))
    if not cands:
        return None
    return cands[len(cands) // 2]


def _load_shots() -> list[dict]:
    doc = load_json(config.ANALYSIS / "shots.json") or {}
    shots = doc.get("shots", doc if isinstance(doc, list) else [])
    return [s for s in shots if s.get("id")]


def run(limit: int | None, only: set[str] | None, provider_name: str,
        force: bool) -> dict:
    prov = OllamaProvider()
    matcher = refmatch.Matcher()
    rstat = refmatch.status()
    print(f"provider {prov.name}:{prov.model} | ref-match "
          f"{'ENABLED ' + str(rstat['ref_ids']) if rstat['enabled'] else 'disabled (characters -> [])'}")

    shots = _load_shots()
    if only:
        shots = [s for s in shots if s["id"] in only]
    elif limit:
        shots = shots[:limit]

    doc = load_json(OUT_PATH) or {"_meta": {}, "tags": {}}
    tags = doc.get("tags", {})
    cache = load_json(CACHE_PATH) or {}

    done = 0
    for s in shots:
        sid = s["id"]
        kf = _keyframe_for(s)
        if kf is None:
            continue
        ckey = sha1(file_sig(kf), prov.model, "vtags_v2")
        if ckey in cache and not force:
            sem = cache[ckey]
        else:
            text = prov._chat(_PROMPT, [kf], schema=_SCHEMA, max_tokens=250)
            parsed = prov._parse_json(text) or {}
            try:
                conf = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            sem = {
                "emotion": _coerce(parsed.get("emotion"), EMOTIONS, "neutral"),
                "scene_type": _coerce(parsed.get("scene_type"), SCENE_TYPES, "establishing"),
                "description": (parsed.get("description") or "").strip()[:200],
                "confidence": max(0.0, min(1.0, conf)),
            }
            cache[ckey] = sem
            save_json(CACHE_PATH, cache)

        characters = matcher.match(kf)          # [] when ref-match disabled
        tags[sid] = {"characters": characters, **sem}
        done += 1
        # persist incrementally so the selector sees progress on long runs
        doc["tags"] = tags
        doc["_meta"] = {"model": prov.model, "generated": int(time.time()),
                        "n": len(tags), "refs": rstat["ref_ids"]}
        save_json(OUT_PATH, doc)
        print(f"  {sid}: {sem['emotion']}/{sem['scene_type']} "
              f"chars={characters} conf={sem['confidence']:.2f} :: {sem['description'][:60]}")

    print(f"\ntagged {done} shot(s) -> {OUT_PATH}  (total in file: {len(tags)})")
    return doc


def main():
    ap = argparse.ArgumentParser(
        prog="pipeline.vision_tags",
        description="Produce analysis/vision_tags.json (VISION_TAGS_CONTRACT.md)")
    ap.add_argument("--limit", type=int, default=None,
                    help="tag only the first N shots (default: all)")
    ap.add_argument("--shots", default=None,
                    help="comma-separated shot ids to tag, e.g. s00_003,s00_010")
    ap.add_argument("--provider", default="ollama", help="vision provider (ollama)")
    ap.add_argument("--force", action="store_true", help="ignore the tag cache")
    args = ap.parse_args()
    only = set(x.strip() for x in args.shots.split(",")) if args.shots else None
    try:
        run(args.limit, only, args.provider, args.force)
    except RuntimeError as e:
        raise SystemExit(f"\n{e}")


if __name__ == "__main__":
    main()
