"""
Chainsaw Man (Reze — the BOMB DEVIL)  ×  Måneskin "Beggin'"  — 30s hype hard-cut.

A short, high-energy showcase of Reze's Bomb Devil battle: sharp cuts on every
beat (no fades), drawn only from the fight act (her bomb fight), rendered at
24fps for smooth motion. The opposite pole from the come_home emotional edit —
same generic pipeline, a different profile + window + clip filter.

Run:  python -m pipeline.beggin_edit [build|render|all]
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import config, select as sel
from .io_util import load_json
from .chainsaw_edit import _filter_clips           # shared credit-region exclusion

AUDIO_FILE = "beggin.mp3"
AUDIO_ANALYSIS = "audio_beggin.json"
PROFILE = "bomb"
FPS = 24                                            # match 23.97 source -> smooth

# 45s with dynamics so it's readable: a verse/build (slower cuts) -> the drop at
# 109.5s -> the longest sustained chorus (109.5-131.3s) -> release -> build to the
# drop at 139.75s. The verse lead-in lets the eye settle before the chorus.
WINDOW = (95.5, 140.5)

# The Bomb Devil battle (Reze's fight act, from analysis/story.json): only draw
# footage from here so every shot is her bomb fight, not random earlier scenes.
FIGHT_SOURCE = (2788.0, 5008.0)

# One bold, on-beat hit on the drop — no lyric underlays (this sped-up "LatinHype"
# cut of Beggin doesn't match lrclib's synced timing, so lyrics would drift).
DROP_OUT = round(109.5 - WINDOW[0], 2)              # drop in output seconds
CAPTIONS = [
    {"at": DROP_OUT, "dur": 2.2, "style": "hit", "text": "BOMB DEVIL", "pos": "center"},
]


def _bomb_clips(clips_doc: dict) -> dict:
    """Credit-exclude, then keep only clips sourced from the Bomb Devil battle."""
    doc = _filter_clips(clips_doc)                  # returns a deep copy
    lo, hi = FIGHT_SOURCE
    n = 0
    for c in doc["clips"]:
        if c.get("usable") and not (lo <= c.get("start", 0.0) <= hi):
            c["usable"] = False
            n += 1
    doc["n_usable"] = sum(1 for c in doc["clips"] if c.get("usable"))
    print(f"restricted to bomb-fight window: dropped {n}; {doc['n_usable']} usable remain")
    return doc


def build(version: int = 2) -> dict:
    audio = load_json(config.ANALYSIS / AUDIO_ANALYSIS)
    if not audio:
        raise RuntimeError(f"missing {AUDIO_ANALYSIS} — analyse public/{AUDIO_FILE} first")
    clips_doc = _bomb_clips(load_json(config.ANALYSIS / "clips.json"))
    return sel.build_plan(
        profile_name=PROFILE, version=version, audio=audio, clips_doc=clips_doc,
        window=WINDOW, captions=CAPTIONS, audio_file=AUDIO_FILE, fps=FPS,
    )


def render_final(plan: dict, chunk: int = 200) -> Path:
    from . import render as render_mod
    plan_path = config.EDITS / f"edit_plan.v{plan['version']}.json"
    return render_mod.render_chunked(plan_path, config.OUT / "beggin_reze.mp4", chunk=chunk)


def _summary(p: dict) -> None:
    from collections import Counter
    m, cl = p["meta"], p["clips"]
    holds = [c["duration"] / m["fps"] for c in cl]
    print(f"beggin BOMB v{p['version']}: {m['durationInFrames']}f "
          f"({m['durationInFrames']/m['fps']:.1f}s) @ {m['fps']}fps {m['width']}x{m['height']}")
    print("clips:", len(cl), "| avg hold %.2fs" % (sum(holds)/len(holds)),
          "| min %.2fs max %.2fs" % (min(holds), max(holds)),
          "| drops:", sum(1 for c in cl if c["isDrop"]),
          "| flashes:", len(p["beatFlashes"]))
    print("transitions:", dict(Counter(c["transitionIn"]["type"] for c in cl)))
    print("levels:", dict(Counter(c["level"] for c in cl)))
    print("captions:", [(c["start"]/m["fps"], c["text"]) for c in p["captions"]])


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        _summary(build())
    elif cmd == "render":
        v = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        print("FINAL ->", render_final(load_json(config.EDITS / f"edit_plan.v{v}.json")))
    elif cmd == "all":
        p = build(); _summary(p)
        print("rendering (chunked, 24fps)..."); print("FINAL ->", render_final(p))
    else:
        raise SystemExit("usage: build | render | all")
