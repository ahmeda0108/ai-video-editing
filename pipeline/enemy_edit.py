"""
Chainsaw Man (Reze) x Imagine Dragons feat. JID "Enemy" — tragic-action arc.

The footage (per the vision tags) is a tragic-action arc: a romance that
detonates into a city-levelling bomb battle and a grief aftermath. This maps
onto Enemy's verse -> "everybody wants to be my enemy" drop -> choruses ->
comedown. Built with the VISION-AWARE arc selector (emotion/scene tags place
each shot by content) but styled with hard on-beat cuts, no fades, at 24fps.

Run:  python -m pipeline.enemy_edit [build|render|all]
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config, story as story_mod, transcript as transcript_mod
from . import select_arc
from .io_util import load_json
from .chainsaw_edit import _filter_clips           # shared credit-region exclusion
from .come_home_arc import MOVIE                    # same source film

FOCUS = "reze"
WINDOW = (48.0, 113.2)                              # verse -> two drops -> comedown
AUDIO_FILE = "enemy.mp3"
AUDIO_ANALYSIS = "audio_enemy.json"
FPS = 24
FADE_IN, FADE_OUT = 0.0, 2.0                        # hard start, soft close on aftermath
PROFILE = "action_arc"

# Act boundaries in OUTPUT seconds, aligned to Enemy's drops (76.0, 101.0 minus
# the window start 48.0): setup(verse) -> the drop -> battle -> comedown.
ACT_BOUNDS = [("happy", 0.0, 28.0), ("fight", 28.0, 57.1), ("aftermath", 57.1, 65.2)]

# Kinetic underlays (premium: per-letter spring reveal + chromatic split + accent
# rule). Placed by OUTPUT seconds. The hook lands on the drop (song 76s -> out 28).
UNDERLAYS = [
    {"at": 26.8, "dur": 3.6, "style": "underlay", "text": "MY ENEMY",
     "sub": "everybody wants to be", "pos": "center"},
]


def _cut_off_ending(plan: dict, hold_sec: float = 2.8) -> dict:
    """End 'properly': decelerate into one long, held aftermath shot (the slow
    sad scene), then HARD-CUT to black+silence. fadeOutSec=0 makes the renderer
    stop picture and music dead on the final frame instead of gently fading."""
    fps = plan["meta"]["fps"]
    total = plan["meta"]["durationInFrames"]
    clips = plan["clips"]
    cut_at = total - round(hold_sec * fps)
    kept = [c for c in clips if c["trackStart"] < cut_at] or clips[:1]
    last = kept[-1]
    last["duration"] = total - last["trackStart"]        # long, lingering final hold
    last["speed"] = 1.0                                   # real-time (no judder)
    last["effects"] = [e for e in last.get("effects", []) if e != "shake"]
    last["transitionIn"] = {"type": "cut", "dur": 0}
    last["reason"] += " | FINAL held aftermath -> hard cut off"
    plan["clips"] = kept
    plan["meta"]["fadeOutSec"] = 0.0                      # picture + music stop dead
    from .io_util import save_json
    save_json(config.EDITS / f"edit_plan.v{plan['version']}.json", plan)
    return plan


def build(version: int = 3) -> dict:
    audio = load_json(config.ANALYSIS / AUDIO_ANALYSIS)
    if not audio:
        raise RuntimeError(f"missing {AUDIO_ANALYSIS}")
    if not load_json(config.ANALYSIS / "transcript.json"):
        os.environ.setdefault("MOVIE_SOURCE", MOVIE)
        transcript_mod.build(Path(MOVIE))
    transcript = load_json(config.ANALYSIS / "transcript.json")
    vision_tags = load_json(config.ANALYSIS / "vision_tags.json")
    vision_tags = vision_tags if isinstance(vision_tags, dict) else None
    story = story_mod.build_arc(FOCUS, transcript=transcript, vision_tags=vision_tags)

    clips_doc = _filter_clips(load_json(config.ANALYSIS / "clips.json"))
    plan = select_arc.build_arc_plan(
        version=version, audio=audio, clips_doc=clips_doc, story=story,
        window=WINDOW, act_song_bounds=ACT_BOUNDS, captions=UNDERLAYS,
        audio_file=AUDIO_FILE, profile_name=PROFILE,
        vision_tags=vision_tags, transcript=transcript,
        fade_in=FADE_IN, fade_out=FADE_OUT, fps=FPS,
    )
    return _cut_off_ending(plan)


def render_final(plan: dict, chunk: int = 200) -> Path:
    from . import render as render_mod
    plan_path = config.EDITS / f"edit_plan.v{plan['version']}.json"
    return render_mod.render_chunked(plan_path, config.OUT / "enemy_reze.mp4", chunk=chunk)


def _summary(p: dict) -> None:
    from collections import Counter
    m, cl = p["meta"], p["clips"]
    holds = [c["duration"] / m["fps"] for c in cl]
    acts = Counter(c["reason"].split(" act")[0] for c in cl)
    print(f"enemy ARC v{p['version']}: {m['durationInFrames']}f "
          f"({m['durationInFrames']/m['fps']:.1f}s) @ {m['fps']}fps {m['width']}x{m['height']} "
          f"fadeIn={m['fadeInSec']} fadeOut={m['fadeOutSec']}")
    print("clips:", len(cl), "| avg hold %.2fs" % (sum(holds)/len(holds)),
          "| min %.2f max %.2f" % (min(holds), max(holds)),
          "| transitions:", dict(Counter(c["transitionIn"]["type"] for c in cl)),
          "| diegetic:", sum(1 for c in cl if c.get("diegetic")))
    print("clips per act:", dict(acts))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        _summary(build())
    elif cmd == "render":
        v = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        print("FINAL ->", render_final(load_json(config.EDITS / f"edit_plan.v{v}.json")))
    elif cmd == "all":
        p = build(); _summary(p)
        print("rendering (chunked, 24fps)..."); print("FINAL ->", render_final(p))
    else:
        raise SystemExit("usage: build | render | all")
