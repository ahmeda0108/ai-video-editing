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


def rel_energy(audio: dict, t: float) -> float:
    """Energy at t as a percentile RANK within the track (0..1).

    Absolute loudness varies wildly track to track; the desired clip intensity
    should track how loud this moment is *relative to the rest of the song*, so
    a chorus maps to ~1 even if its absolute RMS is modest.
    """
    e = np.asarray(audio["energy"])
    se = np.sort(e)
    hz = audio.get("energy_hz", 20.0)
    v = e[min(int(t * hz), len(e) - 1)]
    return float(np.searchsorted(se, v) / max(1, len(se)))


# ---------------------------------------------------------------------------
def build_cut_grid(audio: dict, profile: dict, window=None) -> list[dict]:
    """Beat-aligned segments whose length scales with local musical energy.

    `window` = (start_sec, end_sec) edits only that slice of the track; the
    output timeline is 0-based but each segment keeps `t_abs` (absolute song
    time) for energy lookups.
    """
    w0, w1 = window if window else (0.0, audio["duration"])
    # Grid points: DOWNBEATS (bar lines) when the profile asks for it, so cuts
    # land on the "1" of a bar; otherwise every detected beat.
    grid = audio.get("downbeats") if (profile.get("cut_on") == "downbeats"
                                      and audio.get("downbeats")) else audio["beats"]
    beats = [b for b in grid if w0 - 1e-6 <= b <= w1 + 1e-6]
    if not beats or beats[0] > w0 + 0.05:
        beats = [w0] + beats
    if beats[-1] < w1 - 0.2:
        beats = beats + [w1]
    duration = w1
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

    # Pacing floor: for slow/narrative profiles, merge any cut shorter than
    # min_hold_sec into the previous one so shots breathe (no choppy micro-cuts).
    min_hold = profile.get("min_hold_sec", 0.0)
    if min_hold > 0 and len(segs) > 1:
        merged = [dict(segs[0])]
        for s in segs[1:]:
            if (s["end"] - s["start"]) < min_hold:
                merged[-1]["end"] = s["end"]
                if s.get("is_drop"):
                    merged[-1]["is_drop"] = True
                if s["level"] == "high" or merged[-1]["level"] == "high":
                    merged[-1]["level"] = "high"
            else:
                merged.append(dict(s))
        segs = merged

    # Shift to an output-relative timeline (0-based) but keep t_abs = absolute
    # song time so energy/rel-energy lookups still index the real track.
    for s in segs:
        s["t_abs"] = s["start"]
        s["start"] = round(s["start"] - w0, 3)
        s["end"] = round(s["end"] - w0, 3)
    return segs


# ---------------------------------------------------------------------------
_TRANS_DUR = {"cut": 0, "fade": None, "swell": None, "punch": 8, "whip": 10,
              "slide": 10, "flash": 7, "glitch": 9}


def _trans_dur(kind: str, profile: dict) -> int:
    if kind in ("fade", "swell"):
        return profile["crossfade_frames"]
    return _TRANS_DUR.get(kind, 0)


def _pick_source(clip: dict, seg_dur: float, is_action: bool, force_speed=None):
    """Choose sourceStart + playbackRate to fill seg_dur seconds of timeline.

    `force_speed` (e.g. 0.65) requests deliberate slow-mo; if the clip is too
    short to fill at that rate we fall back to whatever slower rate fills it.
    """
    src_lo, src_hi = clip["start"], clip["end"]
    avail = clip["dur"]
    speed = 1.0 if force_speed is None else float(force_speed)
    need = seg_dur * speed
    if need > avail:                          # not enough source -> slow to fill
        speed = max(0.5, round(avail / seg_dur, 3))   # floor: never crawl < 0.5x
        need = seg_dur * speed
    need = min(need, avail)
    if is_action and clip.get("motion_peaks"):
        centre = clip["motion_peaks"][0]
        sstart = centre - need / 2
    else:
        sstart = src_lo + (avail - need) * 0.3
    sstart = max(src_lo, min(sstart, max(src_lo, src_hi - need)))
    return round(sstart, 3), speed


def assign_clips(segments: list[dict], clips: list[dict], profile: dict, audio: dict,
                 fps: Optional[int] = None) -> list[dict]:
    usable = [c for c in clips if c["usable"]]
    if not usable:
        raise RuntimeError("no usable clips to edit with")
    by_id = {c["id"]: c for c in usable}

    # Narrative mode: pull selection toward CHRONOLOGICAL movie order so the
    # story unfolds. Segment i (fraction p of the edit) prefers clips whose
    # source time is near p through the usable footage span.
    narrative = profile.get("narrative")
    nw = profile.get("narrative_weight", 2.0)
    src_times = [c["start"] for c in usable]
    src_lo, src_hi = (min(src_times), max(src_times)) if src_times else (0.0, 1.0)
    src_span = max(1e-6, src_hi - src_lo)
    total_out = max(1e-6, max(seg["end"] for seg in segments))

    n = len(segments)
    order = list(range(n))
    # Two-pass reservation: assign peak segments (drops / high energy) first from
    # the strongest impact footage so it isn't spent early in quiet passages.
    # Narrative mode skips this (chronology must be assigned in timeline order).
    if profile.get("reserve_impact_for_peaks") and not narrative:
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
        # desired clip intensity = how loud this moment is relative to the song
        target = rel_energy(audio, seg.get("t_abs", seg["start"]))
        is_peak = seg["is_drop"] or seg["level"] == "high"
        prev = prev_clip(si)

        target_src = src_lo + (seg["start"] / total_out) * src_span if narrative else None
        slot_dur = seg["end"] - seg["start"]

        best, best_s = None, -1e9
        for c in usable:
            if c["id"] in last_used_at and (si - last_used_at[c["id"]]) < no_rep:
                continue
            s = 0.9 * c["quality"]
            s += 1.2 * (1.0 - abs(c["intensity"] - target))
            # avoid stretching a too-short clip into a long slot (extreme slow-mo
            # looks laggy). Penalise clips that couldn't fill the slot above ~0.55x.
            if c["dur"] < slot_dur * 0.55:
                s -= 1.0 * (1.0 - c["dur"] / max(1e-6, slot_dur * 0.55))
            if is_peak:
                s += 1.3 * c["impact"] + 0.4 * c["intensity"]
            else:
                s += 0.3 * (1.0 - c["intensity"])          # calmer shots in calm music
            if prev is not None:
                if c.get("cluster") == prev.get("cluster"):
                    s -= 2.5
                s -= 1.6 * _cos_hist(c["color_hist"], prev["color_hist"])
            if narrative:                                  # reward chronological order
                s -= nw * abs(c["start"] - target_src) / src_span
            s -= 0.5 * uses[c["id"]]
            if uses[c["id"]] >= max_reuse:
                s -= 4.0
            if s > best_s:
                best_s, best = s, c
        if best is None:                                    # pool exhausted -> relax no-repeat
            best = max(usable, key=lambda c: c["quality"] - 0.5 * uses[c["id"]])

        seg_dur = seg["end"] - seg["start"]
        is_action = best.get("action") or best["intensity"] > 0.55
        slowmo = profile.get("slowmo_on_impact")
        force = slowmo if (slowmo and is_peak) else None    # linger on impacts
        sstart, speed = _pick_source(best, seg_dur, is_action, force_speed=force)
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
    fps = fps or config.FPS
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
def _captions_to_frames(specs: Optional[list], fps: int, dur_frames: int) -> list[dict]:
    """Convert caption specs (output-relative seconds) to frame-based Captions.

    Spec: {text, sub?, at (sec), dur (sec), style, pos?}. Anything landing past
    the window end is clamped/dropped so it can't run past the composition.
    """
    if not specs:
        return []
    out = []
    for c in specs:
        start = round(float(c["at"]) * fps)
        d = round(float(c.get("dur", 2.0)) * fps)
        if start >= dur_frames:
            continue
        d = min(d, dur_frames - start)
        if d <= 0:
            continue
        cap = {
            "text": c["text"],
            "start": start,
            "dur": d,
            "style": c.get("style", "line"),
        }
        if c.get("sub"):
            cap["sub"] = c["sub"]
        if c.get("pos"):
            cap["pos"] = c["pos"]
        out.append(cap)
    return out


def _probe_loudness(source_static: str, sstart: float, dur: float) -> float:
    """Mean loudness (dBFS) of a source clip's native audio over [sstart, +dur].
    Used to pick diegetic moments that are ACTUALLY loud, not just visually busy.
    Returns a very low value if the clip has no audio / can't be read.
    """
    import re
    import subprocess
    from . import ffmpeg_util
    src = config.PUBLIC / source_static
    if not src.exists():
        return -99.0
    proc = subprocess.run(
        [ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin",
         "-ss", f"{max(0.0, sstart):.3f}", "-t", f"{max(0.4, dur):.3f}",
         "-i", str(src), "-vn", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"mean_volume:\s*(-?[0-9.]+) dB", proc.stderr)
    return float(m.group(1)) if m else -99.0


def _song_peak_windows(audio: Optional[dict], w0: float, fps: int) -> list[tuple]:
    """Output-frame windows covering the song's loud MAIN parts (its `high`
    sections). Diegetic ducks must avoid these so the song's hook/chorus plays
    clean — native movie audio only punches through in the quieter verses."""
    if not audio:
        return []
    out = []
    for s in audio.get("sections", []):
        if s.get("level") == "high":
            a = (s["start"] - w0) * fps
            b = (s["end"] - w0) * fps
            out.append((a, b))
    return out


def _assign_diegetic(clips: list[dict], profile: dict, fps: int,
                     audio: Optional[dict] = None, w0: float = 0.0) -> list[dict]:
    """Pick a capped, well-spaced few hard-hitting clips to keep their NATIVE
    movie audio; mark them and return music-duck windows.

    Two-stage: shortlist by visual impact (explosions/hits move a lot), then
    RE-RANK by actually probing each candidate's native-audio loudness, so a
    ducked-music window always has a genuinely loud hit under it. Kept sparse so
    the mix never clutters. When `protect_song_peaks`, no duck may overlap a
    loud section of the song — only VERY important audio, only over the verses.
    """
    if not profile.get("diegetic"):
        return []
    cap = int(profile.get("diegetic_max", 3))
    gap = float(profile.get("diegetic_min_gap_sec", 12.0)) * fps
    duck_to = float(profile.get("duck_to", 0.3))
    dvol = float(profile.get("diegetic_volume", 0.9))
    min_impact = float(profile.get("diegetic_min_impact", 0.0))
    min_len = 0.7 * fps                    # long enough for the audio to land

    peaks = _song_peak_windows(audio, w0, fps) if profile.get("protect_song_peaks") else []

    def _hits_peak(c) -> bool:
        s, e = c["trackStart"], c["trackStart"] + c["duration"]
        return any(s < pb and e > pa for pa, pb in peaks)

    intro_guard = 1.5 * fps                # let the song open before any duck
    shortlist = [c for c in sorted(
        clips, key=lambda c: (c["isDrop"], c["impact"], c["intensity"]),
        reverse=True)
        if c["duration"] >= min_len and c["trackStart"] >= intro_guard
        and c["impact"] >= min_impact and not _hits_peak(c)][:16]
    # probe native loudness over each clip's actual source window
    for c in shortlist:
        c["_loud"] = _probe_loudness(
            c["source"], c["sourceStart"], (c["duration"] / fps) * c.get("speed", 1.0))
    # prefer loud + impactful; greedily pick spaced-out winners
    shortlist.sort(key=lambda c: c["_loud"] + 6.0 * c["impact"], reverse=True)
    chosen: list[dict] = []
    for c in shortlist:
        if len(chosen) >= cap:
            break
        if c["_loud"] < -40.0:             # effectively silent -> skip
            continue
        if any(abs(c["trackStart"] - o["trackStart"]) < gap for o in chosen):
            continue
        chosen.append(c)

    ducks = []
    for c in chosen:
        c["diegetic"] = True
        c["diegeticVolume"] = round(dvol, 2)
        ducks.append({"start": int(c["trackStart"]), "dur": int(c["duration"]),
                      "to": round(duck_to, 2)})
    ducks.sort(key=lambda d: d["start"])
    return ducks


def build_plan(profile_name: str = "amv", version: int = 1,
               audio: Optional[dict] = None, clips_doc: Optional[dict] = None,
               window=None, captions: Optional[list] = None,
               audio_file: Optional[str] = None, fps: Optional[int] = None) -> dict:
    profile = get_profile(profile_name)
    audio = audio or load_json(config.ANALYSIS / "audio.json")
    clips_doc = clips_doc or load_json(config.ANALYSIS / "clips.json")
    if not audio or not clips_doc:
        raise RuntimeError("run audio + library analysis first")

    fps = fps or config.FPS
    w, h = ORIENTATION_DIMS[profile["orientation"]]
    w0, w1 = window if window else (0.0, audio["duration"])
    duration_frames = round((w1 - w0) * fps)

    segments = build_cut_grid(audio, profile, window=window)
    clips = assign_clips(segments, clips_doc["clips"], profile, audio, fps=fps)

    # beat flashes / impact accents on strong downbeats + drops, restricted to
    # the window and shifted to the output-relative timeline.
    energies = np.array(audio["energy"])
    hi_q = float(np.quantile(energies, 0.72))
    beat_flashes = []
    if profile.get("flash_on_downbeats", True):
        for d in audio.get("downbeats", []):
            if w0 <= d <= w1 and _energy_at(audio, d) >= hi_q:
                beat_flashes.append(round((d - w0) * fps))
    for d in audio.get("drops", []):
        if w0 <= d <= w1:
            beat_flashes.append(round((d - w0) * fps))
    beat_flashes = sorted(f for f in set(beat_flashes) if 0 <= f < duration_frames)
    if not profile.get("flash_on_drop"):     # slow/cinematic profiles: no strobe
        beat_flashes = []

    # diegetic audio: mark a sparse few clips to keep native movie audio + get
    # the music-duck windows that let them punch through.
    audio_ducks = _assign_diegetic(clips, profile, fps)

    plan = {
        "version": version,
        "profile": profile_name,
        "meta": {
            "fps": fps, "width": w, "height": h,
            "durationInFrames": duration_frames,
            "audio": audio_file or Path(audio["track"]).name,
            "audioStart": round(w0, 3),
            "musicVolume": round(float(profile.get("music_volume", 1.0)), 2),
            "tempo_bpm": audio["tempo_bpm"],
        },
        "grade": profile["grade"],
        "clips": clips,
        "beatFlashes": beat_flashes,
        "audioDucks": audio_ducks,
        "captions": _captions_to_frames(captions, fps, duration_frames),
        "audioInfo": {
            "window": [round(w0, 3), round(w1, 3)],
            "drops": [round(d - w0, 3) for d in audio.get("drops", []) if w0 <= d <= w1],
            "sections": audio.get("sections", []),
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
