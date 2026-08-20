"""
Emotional-arc selection — the story-driven alternative to energy/motion cutting.

Each beat-segment belongs to an ACT (happy / fight / aftermath) decided by its
position in the song. Candidate shots for a segment are drawn from that act's
MOVIE time-range (from story.py) and scored by:

  emotion match (vision tag, when present)  >  character presence (vision, else
  transcript speaker map)  >  quality  >  intensity-fits-the-music  >  variety

Everything degrades gracefully: no vision tag for a shot -> fall back to motion
as an emotion proxy and the transcript for character presence; `characters == []`
-> skip the identity bonus. A shot is never dropped just for being untagged.

Smoothness: no aggressive slow-mo here (that judders); pacing comes from long
holds + gentle dissolves. Rendered at the source fps by the driver.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .io_util import load_json, save_json
from .profiles import get_profile, ORIENTATION_DIMS
from .select import (
    build_cut_grid, rel_energy, _cos_hist, _pick_source, _trans_dur,
    _assign_diegetic, _captions_to_frames,
)


# --- character presence from the transcript (fallback when vision is sparse) --
def _presence_intervals(transcript: dict, pad: float = 9.0) -> dict[str, list]:
    """Per character -> merged [start,end] intervals around their spoken lines.
    A shot overlapping an interval is 'probably that character's scene'."""
    raw: dict[str, list] = {}
    for d in transcript.get("dialogue", []):
        sp = d.get("speaker")
        if sp:
            raw.setdefault(sp, []).append([d["start"] - pad, d["end"] + pad])
    merged: dict[str, list] = {}
    for sp, ivs in raw.items():
        ivs.sort()
        out = [ivs[0][:]]
        for a, b in ivs[1:]:
            if a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        merged[sp] = out
    return merged


def _present(intervals: dict, chars: list, t0: float, t1: float) -> bool:
    for ch in chars:
        for a, b in intervals.get(ch, []):
            if t0 <= b and t1 >= a:
                return True
    return False


def _act_of(t_out: float, bounds: list) -> str:
    for name, s0, s1 in bounds:
        if s0 <= t_out < s1:
            return name
    return bounds[-1][0]


def _emotion_from_motion(clip: dict) -> str:
    """Proxy emotion when a shot has no vision tag: motion -> action/tense/calm."""
    if clip.get("action") or clip["intensity"] > 0.6:
        return "action"
    if clip["intensity"] > 0.35:
        return "tense"
    return "tender"


def _assign(segments, clips, profile, audio, story, bounds, tags, presence):
    usable = [c for c in clips if c["usable"]]
    if not usable:
        raise RuntimeError("no usable clips")
    acts = {a["name"]: a for a in story["acts"]}
    by_id = {c["id"]: c for c in usable}
    max_reuse = profile["max_reuse"]
    no_rep = profile["no_repeat_within"]

    uses = {c["id"]: 0 for c in usable}
    last_at: dict[str, int] = {}
    assigned: dict[int, dict] = {}

    for si, seg in enumerate(segments):
        act = acts[_act_of(seg["start"], bounds)]
        pool = [c for c in usable if act["t0"] <= c["start"] <= act["t1"]] or usable
        target = rel_energy(audio, seg.get("t_abs", seg["start"]))
        slot = seg["end"] - seg["start"]
        prev = by_id.get(assigned[si - 1]["clipId"]) if si - 1 in assigned else None
        want_action = act["name"] == "fight"

        best, best_s = None, -1e9
        for c in pool:
            if c["id"] in last_at and (si - last_at[c["id"]]) < no_rep:
                continue
            s = 0.9 * c["quality"]
            s += 0.9 * (1.0 - abs(c["intensity"] - target))
            # emotion: fight wants energetic shots, happy/aftermath want calm
            s += (1.0 if want_action else -0.6) * c["intensity"]
            tag = tags.get(c["id"])
            if tag:                                   # vision refines (reliable)
                if tag.get("emotion") in act["emotions"]:
                    s += 1.6
                if want_action and tag.get("scene_type") == "action":
                    s += 0.7
                if not want_action and tag.get("scene_type") in ("dialogue", "establishing", "reaction"):
                    s += 0.6
                ch = tag.get("characters") or []
                if ch:
                    if set(ch) & set(act["characters"]):
                        s += 1.3                      # our character, on-story
                    if "crowd" in ch or (ch and not set(ch) & set(act["characters"])):
                        s -= 1.2                      # random people -> penalise
            else:                                     # motion proxy when untagged
                if _emotion_from_motion(c) in act["emotions"]:
                    s += 0.5
            if _present(presence, act["characters"], c["start"], c["end"]):
                s += 0.7                              # transcript character presence
            if prev is not None:
                if c.get("cluster") == prev.get("cluster"):
                    s -= 2.5
                s -= 1.4 * _cos_hist(c["color_hist"], prev["color_hist"])
            # Smoothness: strongly prefer clips long enough to fill the slot near
            # 1x. Non-interpolated slow-mo (a short clip stretched) is what makes
            # the picture judder, so short clips are heavily penalised when longer
            # ones are available in the act's pool.
            fill = c["dur"] / max(1e-6, slot)
            if fill < 0.9:
                s -= 3.2 * (0.9 - min(fill, 0.9)) / 0.9
            s -= 0.6 * uses[c["id"]]
            if uses[c["id"]] >= max_reuse:
                s -= 4.0
            if s > best_s:
                best_s, best = s, c
        if best is None:
            best = max(pool, key=lambda c: c["quality"] - 0.5 * uses[c["id"]])

        is_action = best.get("action") or best["intensity"] > 0.55
        sstart, speed = _pick_source(best, slot, is_action)   # no forced slow-mo
        uses[best["id"]] += 1
        last_at[best["id"]] = si
        assigned[si] = {
            "clipId": best["id"], "source": best.get("source_static") or "footage.mp4",
            "sourceStart": sstart, "speed": speed, "is_action": bool(is_action),
            "intensity": best["intensity"], "impact": best["impact"],
            "quality": best["quality"], "cluster": best.get("cluster"),
            "act": act["name"],
        }
    return assigned


def _finalize(segments, assigned, profile, fps: int) -> list[dict]:
    trans = profile["transitions"]
    out = []
    for si, seg in enumerate(segments):
        a = assigned[si]
        start_f = round(seg["start"] * fps)
        dur_f = max(1, round(seg["end"] * fps) - start_f)
        pool = trans["drop"] if seg["is_drop"] else trans[seg["level"]]
        tin = pool[si % len(pool)]
        effects = []
        if not a["is_action"] and profile.get("kenburns_on_static"):
            effects.append("kenburns")
        out.append({
            "id": f"c{si:03d}", "clipId": a["clipId"], "source": a["source"],
            "sourceStart": a["sourceStart"], "trackStart": start_f, "duration": dur_f,
            "speed": a["speed"], "transitionIn": {"type": tin, "dur": _trans_dur(tin, profile)},
            "effects": effects, "level": seg["level"], "isDrop": seg["is_drop"],
            "intensity": round(a["intensity"], 3), "impact": round(a["impact"], 3),
            "reason": f"{a['act']} act; {seg['level']} energy -> intensity {a['intensity']:.2f}",
        })
    return out


def build_arc_plan(version: int, audio: dict, clips_doc: dict, story: dict,
                   window, act_song_bounds: list, captions: Optional[list] = None,
                   audio_file: Optional[str] = None, profile_name: str = "story",
                   vision_tags: Optional[dict] = None, transcript: Optional[dict] = None,
                   fade_in: float = 0.0, fade_out: float = 0.0, fps: int = 24) -> dict:
    profile = get_profile(profile_name)
    w, h = ORIENTATION_DIMS[profile["orientation"]]
    w0, w1 = window
    duration_frames = round((w1 - w0) * fps)

    segments = build_cut_grid(audio, profile, window=window)
    tags = (vision_tags or {}).get("tags", {}) if vision_tags else {}
    presence = _presence_intervals(transcript) if transcript else {}
    assigned = _assign(segments, clips_doc["clips"], profile, audio, story,
                       act_song_bounds, tags, presence)
    clips = _finalize(segments, assigned, profile, fps)
    audio_ducks = _assign_diegetic(clips, profile, fps, audio=audio, w0=w0)

    plan = {
        "version": version, "profile": profile_name,
        "meta": {
            "fps": int(fps), "width": w, "height": h, "durationInFrames": duration_frames,
            "audio": audio_file or "track.mp3", "audioStart": round(w0, 3),
            "musicVolume": round(float(profile.get("music_volume", 1.0)), 2),
            "fadeInSec": round(fade_in, 2), "fadeOutSec": round(fade_out, 2),
            "tempo_bpm": audio.get("tempo_bpm", 0),
        },
        "grade": profile["grade"], "clips": clips, "beatFlashes": [],
        "audioDucks": audio_ducks,
        "captions": _captions_to_frames(captions, fps, duration_frames),
        "audioInfo": {"window": [round(w0, 3), round(w1, 3)],
                      "acts": story["acts"], "act_song_bounds": act_song_bounds},
        "notes": [],
    }
    save_json(config.EDITS / f"edit_plan.v{version}.json", plan)
    return plan
