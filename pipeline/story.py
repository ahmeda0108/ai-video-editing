"""
Emotional arc builder — turns the transcript (+ optional vision tags) into an
ordered set of ACTS the selector maps onto the song.

An act is a movie time-range with a target emotion and character focus:

    {name, t0, t1, emotions[], characters[]}

Boundaries come from the TRANSCRIPT (always available), so this works before any
vision tag exists:
  * the focus character's speaking span bounds the whole arc,
  * impact sound-cues ([explosion]/[screams]/...) locate the fight,
  * everything before the fight is the "happy/together" act, everything after is
    the "aftermath".
Vision emotion tags, when present, only refine per-shot scoring in the selector;
they don't move act boundaries, so a sparse/empty tag file changes nothing here.

Output: analysis/story.json.

    python -m pipeline.story          # default: Reze, happy->fight->aftermath
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import config
from .io_util import load_json, save_json

# sound cues that mean "action/impact" (vs [sighs], [chuckles], ...)
IMPACT_CUES = {
    "explosion", "explosions", "blast", "blasts", "gunshot", "gunshots", "gunfire",
    "screams", "scream", "screaming", "shouts", "shout", "shouting", "crash",
    "crashing", "roars", "roar", "roaring", "grunts", "groans", "explosion stops",
    "gasps", "yells", "yelling", "rumbling", "explosions continue",
}

# Default 3-act template. emotions map to the vision emotion vocabulary
# (VISION_TAGS_CONTRACT.md): tender happy tense action fear grief neutral.
DEFAULT_TEMPLATE = [
    {"name": "happy", "emotions": ["tender", "happy"], "together": True},
    {"name": "fight", "emotions": ["action", "tense", "fear"], "together": False},
    {"name": "aftermath", "emotions": ["grief", "tender", "neutral"], "together": False},
]


def _speaker_times(transcript: dict, who: str) -> list[float]:
    return sorted(d["start"] for d in transcript["dialogue"] if d.get("speaker") == who)


def _impact_curve(transcript: dict, t0: float, t1: float, step: float = 30.0):
    """Smoothed impact-cue density over [t0, t1] at `step`-second resolution."""
    n = max(1, int((t1 - t0) / step) + 1)
    dens = np.zeros(n)
    for c in transcript["sound_cues"]:
        if c["cue"] in IMPACT_CUES and t0 <= c["t"] <= t1:
            dens[min(n - 1, int((c["t"] - t0) / step))] += 1
    if n >= 5:                                   # smooth with a small box filter
        k = np.ones(5) / 5
        dens = np.convolve(dens, k, mode="same")
    return dens


def _fight_window(transcript: dict, t0: float, t1: float) -> tuple[float, float]:
    """Locate the climax fight = sustained high-impact-density region. Searched
    only in the mid-to-late arc so an early skirmish during the romance can't be
    mistaken for the war, and leaving a tail for the aftermath."""
    span = t1 - t0
    lo = t0 + 0.30 * span                        # a happy front third is reserved
    hi = t1 - 0.06 * span                        # a small aftermath tail is reserved
    step = 30.0
    dens = _impact_curve(transcript, lo, hi, step)
    if dens.max() <= 0:                          # no impact cues -> proportional
        return t0 + 0.40 * span, t0 + 0.85 * span
    # the battle = the full span of elevated activity (first..last hot bin), so
    # the build-up isn't left behind in "happy". A modest threshold ignores lone
    # stray cues while keeping the sustained war together.
    thresh = max(dens.mean() * 0.8, dens.max() * 0.25)
    hot = np.where(dens >= thresh)[0]
    fs = lo + hot[0] * step
    fe = lo + (hot[-1] + 1) * step
    return fs, fe


def build_arc(focus: str = "reze", template: Optional[list] = None,
              transcript: Optional[dict] = None,
              vision_tags: Optional[dict] = None) -> dict:
    transcript = transcript or load_json(config.ANALYSIS / "transcript.json")
    if not transcript:
        raise RuntimeError("run `python -m pipeline.transcript <movie>` first")
    template = template or DEFAULT_TEMPLATE

    ftimes = _speaker_times(transcript, focus)
    if ftimes:
        arc0 = max(0.0, ftimes[0] - 20.0)        # small lead-in before first line
        arc1 = ftimes[-1] + 20.0
    else:                                         # focus never labeled -> whole film
        allt = [d["start"] for d in transcript["dialogue"]]
        arc0, arc1 = (min(allt), max(allt)) if allt else (0.0, 0.0)

    fs, fe = _fight_window(transcript, arc0, arc1)
    ranges = {
        "happy": (round(arc0, 1), round(fs, 1)),
        "fight": (round(fs, 1), round(fe, 1)),
        "aftermath": (round(fe, 1), round(arc1, 1)),
    }
    partner = "denji" if focus != "denji" else "reze"
    acts = []
    for spec in template:
        t0, t1 = ranges[spec["name"]]
        chars = [focus, partner] if spec.get("together") else [focus]
        acts.append({
            "name": spec["name"], "t0": t0, "t1": t1,
            "emotions": spec["emotions"], "characters": chars,
        })

    result = {
        "focus": focus,
        "arc": [round(arc0, 1), round(arc1, 1)],
        "fight": [round(fs, 1), round(fe, 1)],
        "acts": acts,
        "tagged": bool(vision_tags),
    }
    save_json(config.ANALYSIS / "story.json", result)
    return result


if __name__ == "__main__":
    import sys
    focus = sys.argv[1] if len(sys.argv) > 1 else "reze"
    vt = load_json(config.ANALYSIS / "vision_tags.json")
    r = build_arc(focus, vision_tags=vt if isinstance(vt, dict) else None)
    print(f"focus={r['focus']}  arc={r['arc'][0]/60:.1f}-{r['arc'][1]/60:.1f}min "
          f"(vision tags: {'yes' if r['tagged'] else 'none yet'})")
    for a in r["acts"]:
        print(f"  {a['name']:9s} {a['t0']/60:5.1f}-{a['t1']/60:5.1f}min "
              f"emotions={a['emotions']} chars={a['characters']}")
