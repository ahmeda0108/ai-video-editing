"""
Project driver: Chainsaw Man — Reze Arc  ×  Jace June "Come Home".

A slower, narrative counterpart to the Can't Hold Us hype cut. Uses the "story"
profile: long breathing holds, CHRONOLOGICAL clip order so the arc unfolds,
slow-mo on impacts, a sparse few diegetic movie-audio hits (music ducks under
them), and only faint lyric UNDERLAYS at the hard-hitting parts — no overlays.

"Come Home" is a plea to a lover who's gone ("...she's never coming home"),
which maps onto Denji longing for Reze. The two underlays are the song's actual
repeated hook, placed in the chorus where it's sung.

Times are OUTPUT-relative seconds (0 = start of the cut).

Run:  python -m pipeline.come_home_edit [build|render|all]
"""
from __future__ import annotations

from pathlib import Path

from . import config
from . import select as sel
from .io_util import load_json
from .chainsaw_edit import _filter_clips   # same movie -> same credit exclusions

# Come Home structure (analysis/audio_come_home.json): drops 22.6/40.2/97.5/121/146.7;
# the sustained hook is 40.2-72.95s. Window catches build -> 40.2 drop -> full hook
# -> the 97.45 beat, ~90s total.
WINDOW = (9.85, 99.85)
AUDIO_FILE = "come_home.mp3"
AUDIO_ANALYSIS = "audio_come_home.json"

# Only real hook fragments, faint, in the chorus window where they're actually
# sung. drop@40.2 -> output ~30.35 ; hook runs to output ~63.
UNDERLAYS = [
    {"at": 31.0, "dur": 4.5, "style": "underlay", "text": "come home"},
    {"at": 60.0, "dur": 4.5, "style": "underlay", "text": "never coming home"},
]


def build(version: int = 1) -> dict:
    audio = load_json(config.ANALYSIS / AUDIO_ANALYSIS)
    if not audio:
        raise RuntimeError(f"missing {AUDIO_ANALYSIS}; run come_home audio analysis first")
    clips_doc = _filter_clips(load_json(config.ANALYSIS / "clips.json"))
    return sel.build_plan(
        profile_name="story",
        version=version,
        audio=audio,
        clips_doc=clips_doc,
        window=WINDOW,
        captions=UNDERLAYS,
        audio_file=AUDIO_FILE,
    )


def render_final(plan: dict, chunk: int = 200) -> Path:
    from . import render as render_mod
    plan_path = config.EDITS / f"edit_plan.v{plan['version']}.json"
    out = config.OUT / "come_home_reze.mp4"
    return render_mod.render_chunked(plan_path, out, chunk=chunk)


def _summarize(p: dict) -> None:
    from collections import Counter
    m, cl = p["meta"], p["clips"]
    print(f"come_home edit v{p['version']}: {m['durationInFrames']}f "
          f"({m['durationInFrames']/m['fps']:.1f}s) @ {m['width']}x{m['height']} "
          f"audio={m['audio']} start={m['audioStart']}s musicVol={m.get('musicVolume')}")
    print("clips:", len(cl), "unique:", len(set(c['clipId'] for c in cl)),
          "| avg hold: %.2fs" % (sum(c['duration'] for c in cl) / max(1, len(cl)) / m['fps']),
          "| slowmo:", sum(1 for c in cl if c['speed'] < 1.0),
          "| diegetic:", sum(1 for c in cl if c.get('diegetic')))
    print("levels:", dict(Counter(c['level'] for c in cl)),
          "| ducks:", p.get('audioDucks'))
    # chronological check: source time should trend upward across the edit
    src = [c['sourceStart'] for c in cl]
    asc = sum(1 for a, b in zip(src, src[1:]) if b >= a)
    print(f"chronological monotonicity: {asc}/{len(src)-1} steps forward "
          f"(src {min(src):.0f}s -> {max(src):.0f}s)")
    for cap in p["captions"]:
        print(f"  underlay f{cap['start']}+{cap['dur']} {cap['text']!r}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        _summarize(build(1))
    elif cmd == "render":
        v = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        p = load_json(config.EDITS / f"edit_plan.v{v}.json")
        print("FINAL ->", render_final(p))
    elif cmd == "all":
        p = build(1)
        _summarize(p)
        print("rendering final (chunked)...")
        print("FINAL ->", render_final(p))
    else:
        raise SystemExit(f"unknown command {cmd!r} (build|render|all)")
