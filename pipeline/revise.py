"""
Deterministic plan revision.

Applies a critique's structured `action`s to the current edit plan to produce
the next version. This is the "don't blindly rewrite everything" step: we mutate
a structured timeline, change a bounded number of cuts per pass (worst issues
first), and keep the beat grid intact. Clip-choice fixes (replace / inject /
diversify) never move cut boundaries; pacing fixes (split / merge) preserve the
outer boundaries so the edit stays locked to the music.

revise(plan, critique) -> new plan (version + 1), saved to edits/.
"""
from __future__ import annotations

import numpy as np

from . import config
from .io_util import save_json
from .select import _pick_source, _trans_dur
from .profiles import get_profile

MAX_CHANGES_PER_PASS = 8


def _energy_at(audio: dict, t: float) -> float:
    hz = audio.get("energy_hz", 20.0)
    e = audio["energy"]
    return float(e[min(int(t * hz), len(e) - 1)])


def _snap_to_beat(frame: int, audio: dict, fps: int) -> int:
    beats = audio.get("beats", [])
    if not beats:
        return frame
    t = frame / fps
    nearest = min(beats, key=lambda b: abs(b - t))
    return int(round(nearest * fps))


def _usage(clips: list[dict]) -> dict:
    u: dict[str, int] = {}
    for c in clips:
        u[c["clipId"]] = u.get(c["clipId"], 0) + 1
    return u


def _pick_replacement(lib_usable, target_intensity, prefer, exclude_clusters,
                      exclude_ids, usage, max_reuse):
    best, best_s = None, -1e9
    for c in lib_usable:
        if c["id"] in exclude_ids:
            continue
        if c.get("cluster") in exclude_clusters:
            continue
        s = 0.9 * c["quality"]
        s += 1.2 * (1.0 - abs(c["intensity"] - target_intensity))
        if prefer == "impact":
            s += 1.5 * c["impact"] + 0.3 * c["intensity"]
        elif prefer == "calm":
            s += 0.8 * (1.0 - c["intensity"])
        s -= 0.6 * usage.get(c["id"], 0)
        if usage.get(c["id"], 0) >= max_reuse:
            s -= 4.0
        if s > best_s:
            best_s, best = s, c
    return best


def _apply_clip(slot: dict, lib_clip: dict, profile: dict):
    """Rewrite a plan slot to use lib_clip (re-pick source, effects)."""
    fps = config.FPS
    seg_dur = slot["duration"] / fps
    is_action = lib_clip.get("action") or lib_clip["intensity"] > 0.55
    sstart, speed = _pick_source(lib_clip, seg_dur, is_action)
    slot["clipId"] = lib_clip["id"]
    slot["source"] = lib_clip.get("source_static") or "footage.mp4"
    slot["sourceStart"] = sstart
    slot["speed"] = speed
    slot["intensity"] = round(lib_clip["intensity"], 3)
    slot["impact"] = round(lib_clip["impact"], 3)
    # refresh effects
    eff = []
    if not is_action and profile.get("kenburns_on_static"):
        eff.append("kenburns")
    if is_action and profile.get("shake_on_action") and (slot["level"] == "high" or slot["isDrop"]):
        eff.append("shake")
    if slot["isDrop"] and profile.get("flash_on_drop"):
        eff.append("flash")
    slot["effects"] = eff


def revise(plan: dict, critique: dict, audio: dict, clips_doc: dict) -> dict:
    profile = get_profile(plan["profile"])
    fps = plan["meta"]["fps"]
    lib = {c["id"]: c for c in clips_doc["clips"]}
    lib_usable = [c for c in clips_doc["clips"] if c["usable"]]
    clips = [dict(c) for c in plan["clips"]]
    by_index = {c["id"]: i for i, c in enumerate(clips)}
    max_reuse = profile["max_reuse"]

    applied: list[str] = []
    issues = sorted(critique.get("issues", []), key=lambda i: -i.get("severity", 1))

    def neighbors_clusters(i):
        cl = set()
        for j in (i - 1, i + 1):
            if 0 <= j < len(clips):
                lc = lib.get(clips[j]["clipId"])
                if lc:
                    cl.add(lc.get("cluster"))
        return cl

    for issue in issues:
        if len(applied) >= MAX_CHANGES_PER_PASS:
            break
        action = issue.get("action") or {}
        if isinstance(action, str):
            action = {"action": action}
        kind = action.get("action")
        cid = issue.get("clipId") or action.get("clipId")

        if kind == "replace" and cid in by_index:
            i = by_index[cid]
            slot = clips[i]
            usage = _usage(clips)
            excl_ids = {slot["clipId"]}
            for j in (i - 1, i + 1):
                if 0 <= j < len(clips):
                    excl_ids.add(clips[j]["clipId"])
            new = _pick_replacement(
                lib_usable,
                target_intensity=action.get("match_intensity", slot.get("intensity", 0.5)),
                prefer=action.get("prefer"),
                exclude_clusters=neighbors_clusters(i),
                exclude_ids=excl_ids, usage=usage, max_reuse=max_reuse)
            if new and new["id"] != slot["clipId"]:
                _apply_clip(slot, new, profile)
                applied.append(f"replace {cid}: -> {new['id']} ({issue['type']})")

        elif kind == "inject":
            lib_id = action.get("libClipId")
            new = lib.get(lib_id)
            if not new:
                continue
            # find the weakest high-energy slot to overwrite
            cands = sorted(
                [c for c in clips if c["level"] == "high" or c["isDrop"]],
                key=lambda c: c.get("impact", 0))
            if cands:
                slot = cands[0]
                _apply_clip(slot, new, profile)
                applied.append(f"inject {lib_id} -> {slot['id']} ({issue['type']})")

        elif kind == "diversify":
            usage = _usage(clips)
            over = [cid_ for cid_, n in usage.items() if n >= 2]
            unused = [c for c in lib_usable if c["id"] not in usage]
            k = 0
            for slot in clips:
                if slot["clipId"] in over and unused:
                    new = unused.pop(0)
                    _apply_clip(slot, new, profile)
                    k += 1
                    if k >= 3:
                        break
            if k:
                applied.append(f"diversify: replaced {k} duplicate cuts")

        elif kind == "shorten" and cid in by_index:
            i = by_index[cid]
            slot = clips[i]
            to = int(action.get("toFrames", slot["duration"] // 2))
            mid = _snap_to_beat(slot["trackStart"] + to, audio, fps)
            if slot["trackStart"] + 4 < mid < slot["trackStart"] + slot["duration"] - 4:
                # split: shrink current, insert a new contrasting cut after it
                new_dur = slot["trackStart"] + slot["duration"] - mid
                slot["duration"] = mid - slot["trackStart"]
                usage = _usage(clips)
                fill = _pick_replacement(
                    lib_usable, target_intensity=slot.get("intensity", 0.6),
                    prefer="impact", exclude_clusters=neighbors_clusters(i),
                    exclude_ids={slot["clipId"]}, usage=usage, max_reuse=max_reuse)
                if fill:
                    new_slot = dict(slot)
                    new_slot["id"] = slot["id"] + "b"
                    new_slot["trackStart"] = mid
                    new_slot["duration"] = new_dur
                    new_slot["transitionIn"] = {"type": "cut", "dur": 0}
                    _apply_clip(new_slot, fill, profile)
                    clips.insert(i + 1, new_slot)
                    by_index = {c["id"]: k for k, c in enumerate(clips)}
                    applied.append(f"split {cid} at {mid}f -> +{new_slot['id']}")

        elif kind == "lengthen" and cid in by_index:
            i = by_index[cid]
            if i + 1 < len(clips):
                nxt = clips[i + 1]
                clips[i]["duration"] = (nxt["trackStart"] + nxt["duration"]) - clips[i]["trackStart"]
                clips.pop(i + 1)
                by_index = {c["id"]: k for k, c in enumerate(clips)}
                applied.append(f"merge {cid} + next")

    # renumber ids for cleanliness and refresh reasons
    for k, c in enumerate(clips):
        c["id"] = f"c{k:03d}"

    new_version = plan["version"] + 1
    new_plan = dict(plan)
    new_plan["version"] = new_version
    new_plan["clips"] = clips
    notes = list(plan.get("notes", []))
    notes.append({
        "version": new_version,
        "from_score": critique.get("score"),
        "applied": applied,
    })
    new_plan["notes"] = notes
    save_json(config.EDITS / f"edit_plan.v{new_version}.json", new_plan)
    return new_plan


if __name__ == "__main__":
    from .io_util import load_json
    plan = load_json(config.EDITS / "edit_plan.v1.json")
    audio = load_json(config.ANALYSIS / "audio.json")
    clips_doc = load_json(config.ANALYSIS / "clips.json")
    from . import critic_metrics
    crit = critic_metrics.evaluate(plan, audio, clips_doc)
    print("v1 critique:", crit["summary"])
    np_ = revise(plan, crit, audio, clips_doc)
    crit2 = critic_metrics.evaluate(np_, audio, clips_doc)
    print("v2 applied:", np_["notes"][-1]["applied"])
    print("v2 critique:", crit2["summary"])
