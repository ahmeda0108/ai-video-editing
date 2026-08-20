"""
Project driver: Chainsaw Man — Reze Arc  ×  "Can't Hold Us".

A concrete, reproducible edit configuration for the generic pipeline. It pins the
90-second musical window (build -> chorus drop -> second drop -> high) and a
curated text track (title / story lines / drop hits / end card) that conveys the
arc and lands its beats on the music.

Times below are OUTPUT-relative seconds (0 = start of the cut). The window's
absolute-song offset is WINDOW[0]; build_plan handles the shift.

Run:  python -m pipeline.chainsaw_edit [version]
"""
from __future__ import annotations

from pathlib import Path

from . import config
from . import select as sel

# Absolute song times (see analysis/audio.json):
#   drops at 91.75 (chorus) and 123.15; section build 76.7, chorus 91.8-117.7.
# Window starts on the 47.5s section boundary where the build kicks in and runs
# 90s through both drops into the high section.
WINDOW = (47.5, 137.5)
AUDIO_FILE = "cant_hold_us.mp3"

# Source-time (seconds into the film) ranges to NEVER draw footage from: the
# opening staff-credit montage (~185-330s, credits burned into the frame) and
# the end-credit roll (after ~5760s). The motion detector loves the credited
# action teaser, so without this the impact clips on the drops carry Japanese
# staff text. Verified against source frames.
EXCLUDE_SOURCE_RANGES = [(185.0, 330.0), (5760.0, 1e9)]


def _filter_clips(clips_doc: dict) -> dict:
    """Return a copy of clips_doc with clips overlapping an excluded source
    range marked unusable, so selection can't pick credit-burned footage."""
    import copy
    doc = copy.deepcopy(clips_doc)
    n = 0
    for c in doc["clips"]:
        cs, ce = c.get("start", 0.0), c.get("end", 0.0)
        for lo, hi in EXCLUDE_SOURCE_RANGES:
            if cs < hi and ce > lo:            # overlap
                if c.get("usable"):
                    c["usable"] = False
                    n += 1
                break
    doc["n_usable"] = sum(1 for c in doc["clips"] if c.get("usable"))
    print(f"excluded {n} credit-region clips; {doc['n_usable']} usable remain")
    return doc

# Drops in output time: 91.75-47.5 = 44.25 ; 123.15-47.5 = 75.65
DROP1 = 44.25
DROP2 = 75.65

# Curated caption track. style: title | line | hit | end. Spoiler-aware but the
# Bomb Devil reveal *is* the arc — this is a "remember it" edit.
CAPTIONS = [
    {"at": 0.8,  "dur": 3.4, "style": "title", "text": "CHAINSAW MAN", "sub": "REZE ARC"},
    {"at": 10.0, "dur": 3.2, "style": "line",  "text": "he only wanted an ordinary life", "pos": "lower"},
    {"at": 19.5, "dur": 3.0, "style": "line",  "text": "then she walked in", "pos": "lower"},
    {"at": 31.0, "dur": 3.0, "style": "line",  "text": "she felt like summer", "pos": "lower"},
    # biggest chorus drop — the reveal hits
    {"at": DROP1,       "dur": 2.6, "style": "hit", "text": "THE BOMB DEVIL", "pos": "center"},
    {"at": 53.0, "dur": 3.0, "style": "line",  "text": "“run away with me”", "pos": "lower"},
    {"at": 63.5, "dur": 3.0, "style": "line",  "text": "there was never a normal for us", "pos": "lower"},
    # second drop — defiant, ties to the song
    {"at": DROP2,       "dur": 2.6, "style": "hit", "text": "CAN'T HOLD US", "pos": "center"},
    {"at": 84.0, "dur": 5.0, "style": "end", "text": "CHAINSAW MAN", "sub": "reze"},
]


def build(version: int = 1) -> dict:
    from .io_util import load_json
    clips_doc = _filter_clips(load_json(config.ANALYSIS / "clips.json"))
    plan = sel.build_plan(
        profile_name="amv",
        version=version,
        clips_doc=clips_doc,
        window=WINDOW,
        captions=CAPTIONS,
        audio_file=AUDIO_FILE,
    )
    return plan


def iterate(iters: int = 5) -> dict:
    """Build v1, then run the local structural critique/revise loop until the
    score plateaus. No per-iteration render (local critic reasons over the
    structured plan) -> cost-free and OOM-free. Returns the converged plan.
    """
    from . import critic_metrics, revise as revise_mod
    from .io_util import load_json, save_json

    audio = load_json(config.ANALYSIS / "audio.json")
    clips_doc = load_json(config.ANALYSIS / "clips.json")
    if not audio or not clips_doc:
        raise RuntimeError("run analysis (audio + clip library) first")
    clips_doc = _filter_clips(clips_doc)   # same exclusions the selector used

    plan = build(version=1)
    history = []
    prev = None
    best_plan, best_score = plan, -1
    for it in range(1, iters + 1):
        crit = critic_metrics.evaluate(plan, audio, clips_doc)
        save_json(config.EDITS / f"critique.v{plan['version']}.json", crit)
        history.append({"version": plan["version"], "score": crit["score"]})
        print(f"  v{plan['version']}: {crit['summary']}")
        if crit["score"] > best_score:
            best_plan, best_score = plan, crit["score"]
        if crit["score"] >= 96:
            print("  -> excellent; stopping.")
            break
        if prev is not None and (crit["score"] - prev) < 2 and it > 1:
            print("  -> improvement negligible; stopping.")
            break
        prev = crit["score"]
        new_plan = revise_mod.revise(plan, crit, audio, clips_doc)
        if not new_plan["notes"][-1]["applied"]:
            print("  -> no actionable changes; stopping.")
            break
        plan = new_plan

    # A revision can score worse than its parent; keep the best version, not the last.
    save_json(config.EDITS / "history.json",
              {"history": history, "final_version": best_plan["version"],
               "best_score": best_score})
    print("score history:", " -> ".join(f"v{h['version']}:{h['score']}" for h in history))
    print(f"best: v{best_plan['version']} @ {best_score}/100")
    return best_plan


def render_final(plan: dict, chunk: int = 200) -> "Path":
    """Chunked render of the converged plan to out/chainsaw_reze.mp4."""
    from . import render as render_mod
    plan_path = config.EDITS / f"edit_plan.v{plan['version']}.json"
    out = config.OUT / "chainsaw_reze.mp4"
    return render_mod.render_chunked(plan_path, out, chunk=chunk)


def _summarize(p: dict) -> None:
    from collections import Counter
    m, cl = p["meta"], p["clips"]
    print(f"chainsaw edit v{p['version']}: {m['durationInFrames']}f "
          f"({m['durationInFrames']/m['fps']:.1f}s) @ {m['width']}x{m['height']} "
          f"audio={m['audio']} start={m['audioStart']}s")
    print("clips:", len(cl), "unique:", len(set(c['clipId'] for c in cl)),
          "drops:", sum(1 for c in cl if c['isDrop']),
          "flashes:", len(p['beatFlashes']), "captions:", len(p['captions']))
    print("levels:", dict(Counter(c['level'] for c in cl)))
    for cap in p["captions"]:
        print(f"  cap f{cap['start']:>4}+{cap['dur']:>3} {cap['style']:>5} "
              f"{cap['text']!r}" + (f" / {cap['sub']!r}" if cap.get('sub') else ""))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        _summarize(build(1))
    elif cmd == "iterate":
        _summarize(iterate())
    elif cmd == "all":
        p = iterate()
        _summarize(p)
        print("rendering final (chunked)...")
        out = render_final(p)
        print("FINAL ->", out)
    elif cmd == "render":
        from .io_util import load_json
        v = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        p = load_json(config.EDITS / f"edit_plan.v{v}.json")
        print("FINAL ->", render_final(p))
    else:
        raise SystemExit(f"unknown command {cmd!r} (build|iterate|render|all)")
