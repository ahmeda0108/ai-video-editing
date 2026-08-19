"""
Clip selection + edit-plan generation — the editorial brain.

Turns musical structure (beats / energy / sections / drops) + the clip library
into a concrete, deterministic edit plan that Remotion renders. The logic is
generic; a `profile` (amv / hype / montage / ...) tunes it.

Editorial principles encoded here:
  * cut ON beats; density scales with musical energy (fast on drops, slow in
    quiet sections) -> pacing follows the music
  * reserve high-impact footage for high-energy sections and drops -> buildup
    -> payoff, and strong footage isn't wasted in weak musical sections
  * match clip visual intensity to musical energy -> no busy shot in a calm
    passage, no static shot on a beat drop
  * never place visually similar / same-cluster shots back to back -> shot
    variety, no repetition
  * ken-burns on static shots, punch/flash/shake on impacts -> motion where the
    music wants it

Output: edits/edit_plan.vN.json (versioned, inspectable) — the render contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .io_util import load_json, save_json
from .profiles import get_profile, ORIENTATION_DIMS


# ---------------------------------------------------------------------------
def _cos_hist(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
    return float(np.dot(va, vb) / denom)


def _energy_at(audio: dict, t: float) -> float:
    hz = audio.get("energy_hz", 20.0)
    e = audio["energy"]
    i = min(int(t * hz), len(e) - 1)
    return float(e[i])


def _level(e: float, lo: float, hi: float) -> str:
    if e >= hi:
        return "high"
    if e <= lo:
        return "low"
    return "mid"


# ---------------------------------------------------------------------------
def build_cut_grid(audio: dict, profile: dict) -> list[dict]:
    """Beat-aligned segments whose length scales with local musical energy."""
    beats = list(audio["beats"])
    duration = audio["duration"]
    if not beats or beats[-1] < duration - 0.2:
        beats = beats + [duration]
    energies = np.array(audio["energy"])
    lo_q, hi_q = float(np.quantile(energies, 0.4)), float(np.quantile(energies, 0.72))
    drops = set(round(d, 1) for d in audio.get("drops", []))
    bpc = profile["beats_per_clip"]

    segs: list[dict] = []
    i = 0
    while i < len(beats) - 1:
        t = beats[i]
        e = _energy_at(audio, t)
        lvl = _level(e, lo_q, hi_q)
        stride = bpc[lvl]
        j = min(i + stride, len(beats) - 1)
        start, end = beats[i], beats[j]

        # a drop must fall on a cut: if a drop lands inside this segment, end here
        is_drop = False
        for k in range(i + 1, j + 1):
            if round(beats[k], 1) in drops or any(abs(beats[k] - d) < 0.12 for d in audio.get("drops", [])):
                j = k
                end = beats[k]
                is_drop = True
                break
        # half-beat densification right at a drop for a punchy hit
        if is_drop and profile.get("allow_half_beat_on_drop") and (end - start) > 0.35:
            mid = round((start + end) / 2, 3)
            segs.append({"start": round(start, 3), "end": mid, "level": "high", "energy": e, "is_drop": False})
            segs.append({"start": mid, "end": round(end, 3), "level": "high", "energy": e, "is_drop": True})
        else:
            segs.append({"start": round(start, 3), "end": round(end, 3), "level": lvl, "energy": e, "is_drop": is_drop})
        i = j
    # make sure we cover to the very end
    if segs and segs[-1]["end"] < duration - 0.05:
        segs.append({"start": segs[-1]["end"], "end": round(duration, 3),
                     "level": "low", "energy": _energy_at(audio, segs[-1]["end"]), "is_drop": False})
    return segs


# ---------------------------------------------------------------------------
_TRANS_DUR = {"cut": 0, "fade": None, "swell": None, "punch": 8, "whip": 10,
              "slide": 10, "flash": 7, "glitch": 9}


def _trans_dur(kind: str, profile: dict) -> int:
    if kind in ("fade", "swell"):
        return profile["crossfade_frames"]
    return _TRANS_DUR.get(kind, 0)


def _pick_source(clip: dict, seg_dur: float, is_action: bool):
    """Choose sourceStart + playbackRate to fill seg_dur seconds of timeline."""
    src_lo, src_hi = clip["start"], clip["end"]
    avail = clip["dur"]
    speed = 1.0
    need = seg_dur
    if avail < seg_dur:                       # too short -> gentle slow-mo to fill
        speed = max(0.5, round(avail / seg_dur, 3))
        need = seg_dur * speed
    need = min(need, avail)
    if is_action and clip.get("motion_peaks"):
        centre = clip["motion_peaks"][0]
        sstart = centre - need / 2
    else:
        sstart = src_lo + (avail - need) * 0.3
    sstart = max(src_lo, min(sstart, max(src_lo, src_hi - need)))
    return round(sstart, 3), speed


def assign_clips(segments: list[dict], clips: list[dict], profile: dict, audio: dict) -> list[dict]:
    usable = [c for c in clips if c["usable"]]
    if not usable:
        raise RuntimeError("no usable clips to edit with")
    by_id = {c["id"]: c for c in usable}

    n = len(segments)
    order = list(range(n))
    # Two-pass reservation: assign peak segments (drops / high energy) first from
    # the strongest impact footage so it isn't spent early in quiet passages.
    if profile.get("reserve_impact_for_peaks"):
        peak = [i for i in order if segments[i]["is_drop"] or segments[i]["level"] == "high"]
        rest = [i for i in order if i not in set(peak)]
        order = peak + rest

    uses: dict[str, int] = {c["id"]: 0 for c in usable}
    last_used_at: dict[str, int] = {}
    assigned: dict[int, dict] = {}
    max_reuse = profile["max_reuse"]
    no_rep = profile["no_repeat_within"]

    def prev_clip(i: int):
        if i - 1 >= 0 and (i - 1) in assigned:
            return by_id.get(assigned[i - 1]["clipId"])
        return None

    for si in order:
        seg = segments[si]
        t = seg["energy"]
        target = float(np.clip((t - 0.15) / 0.6, 0.0, 1.0))  # musical energy -> desired clip intensity
        is_peak = seg["is_drop"] or seg["level"] == "high"
        prev = prev_clip(si)

        best, best_s = None, -1e9
        for c in usable:
            if c["id"] in last_used_at and (si - last_used_at[c["id"]]) < no_rep:
                continue
            s = 0.9 * c["quality"]
            s += 1.2 * (1.0 - abs(c["intensity"] - target))
            if is_peak:
                s += 1.3 * c["impact"] + 0.4 * c["intensity"]
            else:
                s += 0.3 * (1.0 - c["intensity"])          # calmer shots in calm music
            if prev is not None:
                if c.get("cluster") == prev.get("cluster"):
                    s -= 2.5
                s -= 1.6 * _cos_hist(c["color_hist"], prev["color_hist"])
            s -= 0.5 * uses[c["id"]]
            if uses[c["id"]] >= max_reuse:
                s -= 4.0
            if s > best_s:
                best_s, best = s, c
        if best is None:                                    # pool exhausted -> relax no-repeat
            best = max(usable, key=lambda c: c["quality"] - 0.5 * uses[c["id"]])

        seg_dur = seg["end"] - seg["start"]
        is_action = best.get("action") or best["intensity"] > 0.55
        sstart, speed = _pick_source(best, seg_dur, is_action)
        uses[best["id"]] += 1
        last_used_at[best["id"]] = si
        assigned[si] = {
            "clipId": best["id"],
            "source": best.get("source_static") or "footage.mp4",
            "sourceStart": sstart,
            "speed": speed,
            "is_action": bool(is_action),
            "intensity": best["intensity"],
            "impact": best["impact"],
            "quality": best["quality"],
            "cluster": best.get("cluster"),
        }

    # ---- second sweep in timeline order: transitions, effects, framing ------
    fps = config.FPS
    out: list[dict] = []
    trans = profile["transitions"]
    for si in range(n):
        seg = segments[si]
        a = assigned[si]
        start_f = round(seg["start"] * fps)
        end_f = round(seg["end"] * fps)
        dur_f = max(1, end_f - start_f)

        if seg["is_drop"]:
            kind_pool = trans["drop"]
        else:
            kind_pool = trans[seg["level"]]
        tin = kind_pool[si % len(kind_pool)]

        effects = []
        if not a["is_action"] and profile.get("kenburns_on_static"):
            effects.append("kenburns")
        if a["is_action"] and profile.get("shake_on_action") and (seg["level"] == "high" or seg["is_drop"]):
            effects.append("shake")
        if seg["is_drop"] and profile.get("flash_on_drop"):
            effects.append("flash")

        reason = (
            f"{'DROP ' if seg['is_drop'] else ''}{seg['level']} energy "
            f"({seg['energy']:.2f}) -> intensity {a['intensity']:.2f}, "
            f"quality {a['quality']:.2f}"
        )

        out.append({
            "id": f"c{si:03d}",
            "clipId": a["clipId"],
            "source": a["source"],
            "sourceStart": a["sourceStart"],
            "trackStart": start_f,
            "duration": dur_f,
            "speed": a["speed"],
            "transitionIn": {"type": tin, "dur": _trans_dur(tin, profile)},
            "effects": effects,
            "level": seg["level"],
            "isDrop": seg["is_drop"],
            "intensity": round(a["intensity"], 3),
            "impact": round(a["impact"], 3),
            "reason": reason,
        })
    return out


# ---------------------------------------------------------------------------
def build_plan(profile_name: str = "amv", version: int = 1,
               audio: Optional[dict] = None, clips_doc: Optional[dict] = None) -> dict:
    profile = get_profile(profile_name)
    audio = audio or load_json(config.ANALYSIS / "audio.json")
    clips_doc = clips_doc or load_json(config.ANALYSIS / "clips.json")
    if not audio or not clips_doc:
        raise RuntimeError("run audio + library analysis first")

    fps = config.FPS
    w, h = ORIENTATION_DIMS[profile["orientation"]]
    duration_frames = round(audio["duration"] * fps)

    segments = build_cut_grid(audio, profile)
    clips = assign_clips(segments, clips_doc["clips"], profile, audio)

    # beat flashes / impact accents on strong downbeats + drops
    energies = np.array(audio["energy"])
    hi_q = float(np.quantile(energies, 0.72))
    beat_flashes = []
    for d in audio.get("downbeats", []):
        if _energy_at(audio, d) >= hi_q:
            beat_flashes.append(round(d * fps))
    for d in audio.get("drops", []):
        beat_flashes.append(round(d * fps))
    beat_flashes = sorted(set(beat_flashes))

    plan = {
        "version": version,
        "profile": profile_name,
        "meta": {
            "fps": fps, "width": w, "height": h,
            "durationInFrames": duration_frames,
            "audio": Path(audio["track"]).name,
            "tempo_bpm": audio["tempo_bpm"],
        },
        "grade": profile["grade"],
        "clips": clips,
        "beatFlashes": beat_flashes,
        "audioInfo": {
            "drops": audio.get("drops", []),
            "sections": audio.get("sections", []),
            "beats": audio.get("beats", []),
        },
        "notes": [],
    }
    save_json(config.EDITS / f"edit_plan.v{version}.json", plan)
    return plan


if __name__ == "__main__":
    import sys
    prof = sys.argv[1] if len(sys.argv) > 1 else "amv"
    p = build_plan(prof, version=1)
    cl = p["clips"]
    from collections import Counter
    lv = Counter(c["level"] for c in cl)
    print(f"profile={prof} clips={len(cl)} duration={p['meta']['durationInFrames']}f "
          f"({p['meta']['durationInFrames']/config.FPS:.1f}s)")
    print("levels:", dict(lv), "drops:", sum(1 for c in cl if c['isDrop']),
          "flashes:", len(p["beatFlashes"]))
    uniq = len(set(c["clipId"] for c in cl))
    print(f"unique clips used: {uniq}/{len(cl)} cuts")
    durs = [c["duration"] for c in cl]
    print(f"clip len frames min/med/max: {min(durs)}/{sorted(durs)[len(durs)//2]}/{max(durs)}")
    print("first 6 cuts:")
    for c in cl[:6]:
        print(f"  {c['id']} f{c['trackStart']:>3}+{c['duration']:>2} {c['clipId']} "
              f"src@{c['sourceStart']:.1f}s x{c['speed']} {c['transitionIn']['type']:>5} "
              f"{','.join(c['effects']) or '-'} | {c['reason']}")
