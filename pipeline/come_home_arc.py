"""
Chainsaw Man (Reze) x Jace June "Come Home" — emotional-arc edit (v2).

The story-driven redo: transcript-derived acts (happy -> fight -> aftermath),
character-isolated + emotion-matched selection, lyric underlays aligned to the
exact sung word, a proper fade-in/out, and 24fps rendering for smoothness.

Run:  python -m pipeline.come_home_arc [build|render|all]
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config, story as story_mod, transcript as transcript_mod
from . import select_arc, lyrics as lyrics_mod
from .io_util import load_json
from .chainsaw_edit import _filter_clips           # shared credit-region exclusion

MOVIE = ("/c/Users/beast/Downloads/www.UIndex.org    -    Chainsaw Man - The Movie "
         "Reze Arc (2025) 2160p 4K WEB 5.1-WORLD/Chainsaw.Man.-.The.Movie.Reze.Arc."
         "2025.2160p.4K.WEB.x265.10bit.AAC5.1-WORLD.mkv")

FOCUS = "reze"
WINDOW = (1.0, 150.5)                                # opens on the hook, ends at outro
AUDIO_FILE = "come_home.mp3"
AUDIO_ANALYSIS = "audio_come_home.json"
FPS = 24                                             # match source 23.97 -> smooth
FADE_IN, FADE_OUT = 1.2, 3.5
ARTIST, TRACK = "Jace June", "Come Home"

# Act boundaries in OUTPUT seconds, aligned to the song's drops (40.2, 120.95;
# minus the window start 1.0): happy -> the drop -> fight -> the drop -> aftermath.
ACT_BOUNDS = [("happy", 0.0, 39.2), ("fight", 39.2, 119.95), ("aftermath", 119.95, 149.5)]

# Curated lyric underlays: emotionally-key lines only (no "la da da" filler),
# referenced by their real LRC song-time so they stay exactly aligned.
UNDERLAY_SONG_LINES = [
    (4.33,   "I need you to hold me"),
    (24.60,  "Never coming home"),
    (39.70,  "And she was an angel"),
    (75.50,  "So baby come home"),
    (125.31, "Baby come home"),
    (148.96, "Never coming home"),
]


def _underlays() -> list[dict]:
    w0 = WINDOW[0]
    caps = []
    for song_t, text in UNDERLAY_SONG_LINES:
        at = song_t - w0
        if 0 <= at <= (WINDOW[1] - w0):
            caps.append({"at": round(at, 2), "dur": 2.8, "style": "underlay", "text": text})
    return caps


def build(version: int = 1) -> dict:
    audio = load_json(config.ANALYSIS / AUDIO_ANALYSIS)
    if not audio:
        raise RuntimeError(f"missing {AUDIO_ANALYSIS}")
    # transcript (extract once if needed) + story arc
    if not load_json(config.ANALYSIS / "transcript.json"):
        os.environ.setdefault("MOVIE_SOURCE", MOVIE)
        transcript_mod.build(Path(MOVIE))
    transcript = load_json(config.ANALYSIS / "transcript.json")
    vision_tags = load_json(config.ANALYSIS / "vision_tags.json")
    vision_tags = vision_tags if isinstance(vision_tags, dict) else None
    story = story_mod.build_arc(FOCUS, transcript=transcript, vision_tags=vision_tags)

    lyrics_mod.fetch(ARTIST, TRACK)                  # cache LRC (alignment source)
    clips_doc = _filter_clips(load_json(config.ANALYSIS / "clips.json"))

    return select_arc.build_arc_plan(
        version=version, audio=audio, clips_doc=clips_doc, story=story,
        window=WINDOW, act_song_bounds=ACT_BOUNDS, captions=_underlays(),
        audio_file=AUDIO_FILE, profile_name="story",
        vision_tags=vision_tags, transcript=transcript,
        fade_in=FADE_IN, fade_out=FADE_OUT, fps=FPS,
    )


def render_final(plan: dict, chunk: int = 160) -> Path:
    from . import render as render_mod
    plan_path = config.EDITS / f"edit_plan.v{plan['version']}.json"
    return render_mod.render_chunked(plan_path, config.OUT / "come_home_arc.mp4", chunk=chunk)


def _summary(p: dict) -> None:
    from collections import Counter
    m, cl = p["meta"], p["clips"]
    print(f"come_home ARC v{p['version']}: {m['durationInFrames']}f "
          f"({m['durationInFrames']/m['fps']:.1f}s) @ {m['fps']}fps {m['width']}x{m['height']} "
          f"fadeIn={m['fadeInSec']} fadeOut={m['fadeOutSec']}")
    acts = Counter(c["reason"].split(" act")[0] for c in cl)
    print("clips:", len(cl), "| avg hold %.2fs" % (sum(c['duration'] for c in cl)/max(1,len(cl))/m['fps']),
          "| speed<1:", sum(1 for c in cl if c['speed'] < 1), "| diegetic:", sum(1 for c in cl if c.get('diegetic')))
    print("clips per act:", dict(acts))
    print("underlays:", [(c['start'], c['text']) for c in p['captions']])


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        _summary(build(1))
    elif cmd == "render":
        v = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print("FINAL ->", render_final(load_json(config.EDITS / f"edit_plan.v{v}.json")))
    elif cmd == "all":
        p = build(1); _summary(p)
        print("rendering (chunked, 24fps)..."); print("FINAL ->", render_final(p))
    else:
        raise SystemExit("usage: build | render | all")
