"""
Metrics critic — evaluate an edit plan WITHOUT looking at pixels.

Operates purely on the plan + audio analysis + clip library, and produces a
score plus structured, timestamped, machine-actionable issues. This is:
  * the always-available critic when there is no vision API key, and
  * the objective backbone that the vision critic's feedback is merged into.

Each issue carries an `action` dict that revise.py can apply deterministically,
so critique -> revision never requires a blind full rewrite.
"""
from __future__ import annotations

import numpy as np

from . import config


def _cos(a, b) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9))


def _energy_at(audio: dict, t: float) -> float:
    hz = audio.get("energy_hz", 20.0)
    e = audio["energy"]
    return float(e[min(int(t * hz), len(e) - 1)])


def evaluate(plan: dict, audio: dict, clips_doc: dict) -> dict:
    fps = plan["meta"]["fps"]
    lib = {c["id"]: c for c in clips_doc["clips"]}
    clips = plan["clips"]
    energies = np.array(audio["energy"])
    hi_q = float(np.quantile(energies, 0.72))
    beats = audio.get("beats", [])

    issues: list[dict] = []

    def add(sev, typ, cid, note, suggestion, action, t):
        issues.append({
            "severity": sev, "type": typ, "clipId": cid,
            "t": round(t, 2), "frame": int(round(t * fps)),
            "note": note, "suggestion": suggestion, "action": action,
        })

    # ---- 1. repetition: same cluster / near-identical palette back to back ---
    for a, b in zip(clips, clips[1:]):
        la, lb = lib.get(a["clipId"]), lib.get(b["clipId"])
        if not la or not lb:
            continue
        same_cluster = la.get("cluster") is not None and la.get("cluster") == lb.get("cluster")
        pal = _cos(la["color_hist"], lb["color_hist"])
        if same_cluster or pal > 0.92:
            add(2, "repetition", b["id"],
                f"{b['id']} looks like the previous shot ({a['id']}) "
                f"({'same cluster' if same_cluster else f'palette sim {pal:.2f}'})",
                "replace with a visually different clip",
                {"action": "replace", "clipId": b["id"], "avoid_cluster": la.get("cluster"),
                 "match_intensity": b.get("intensity")},
                b["trackStart"] / fps)

    # ---- 2. intensity mismatch vs music --------------------------------------
    for c in clips:
        t = c["trackStart"] / fps
        e = _energy_at(audio, t)
        target = float(np.clip((e - 0.15) / 0.6, 0.0, 1.0))
        inten = c.get("intensity", 0.5)
        if e >= hi_q and inten < 0.4:
            add(3, "weak_in_strong", c["id"],
                f"{c['id']} is a low-intensity shot ({inten:.2f}) on a high-energy "
                f"beat (energy {e:.2f}) — wasted strong musical moment",
                "use a high-motion / high-impact clip here",
                {"action": "replace", "clipId": c["id"], "prefer": "impact",
                 "match_intensity": max(0.7, target)}, t)
        elif e < 0.22 and inten > 0.7:
            add(1, "busy_in_calm", c["id"],
                f"{c['id']} is a busy shot ({inten:.2f}) in a calm passage "
                f"(energy {e:.2f})",
                "use a calmer / scenic clip here",
                {"action": "replace", "clipId": c["id"], "prefer": "calm",
                 "match_intensity": target}, t)

    # ---- 3. weak drops -------------------------------------------------------
    for d in audio.get("drops", []):
        # find the clip covering the drop
        cover = None
        for c in clips:
            s = c["trackStart"] / fps
            e = (c["trackStart"] + c["duration"]) / fps
            if s - 0.15 <= d <= e:
                cover = c
                break
        if cover and cover.get("impact", 0) < 0.6:
            add(3, "weak_drop", cover["id"],
                f"drop at {d:.1f}s landed on {cover['id']} with low impact "
                f"({cover.get('impact', 0):.2f})",
                "put the highest-impact clip on the drop",
                {"action": "replace", "clipId": cover["id"], "prefer": "impact",
                 "match_intensity": 0.9}, d)

    # ---- 4. pacing: under/over-editing --------------------------------------
    for c in clips:
        t = c["trackStart"] / fps
        e = _energy_at(audio, t)
        if e >= hi_q and c["duration"] > int(1.4 * fps) and not c["isDrop"]:
            add(1, "under_editing", c["id"],
                f"{c['id']} runs {c['duration'] / fps:.1f}s in a high-energy part — too long",
                "shorten to keep pace with the music",
                {"action": "shorten", "clipId": c["id"], "toFrames": int(0.8 * fps)}, t)
    # over-editing: 4+ consecutive ultra-short non-drop clips
    run = 0
    for c in clips:
        if c["duration"] < int(0.3 * fps) and not c["isDrop"]:
            run += 1
            if run >= 4:
                add(1, "over_editing", c["id"],
                    "several ultra-short cuts in a row — risks feeling frantic",
                    "merge or lengthen some of these cuts",
                    {"action": "lengthen", "clipId": c["id"], "toFrames": int(0.5 * fps)},
                    c["trackStart"] / fps)
        else:
            run = 0

    # ---- 5. wasted strong footage -------------------------------------------
    used = set(c["clipId"] for c in clips)
    strong_unused = [c for c in clips_doc["clips"]
                     if c["usable"] and c.get("impact", 0) > 0.8 and c["id"] not in used]
    for su in strong_unused[:3]:
        add(2, "wasted_footage", None,
            f"high-impact clip {su['id']} (impact {su['impact']:.2f}) is never used",
            "swap it into a drop or high-energy beat",
            {"action": "inject", "libClipId": su["id"], "prefer": "impact"}, 0.0)

    # ---- 6. shot variety -----------------------------------------------------
    clusters = [lib[c["clipId"]].get("cluster") for c in clips if c["clipId"] in lib]
    variety = len(set(clusters)) / max(1, len(clusters))
    if variety < 0.55:
        add(2, "low_variety", None,
            f"only {len(set(clusters))} distinct shots across {len(clusters)} cuts "
            f"(variety {variety:.2f})",
            "increase shot variety; avoid reusing the same footage",
            {"action": "diversify"}, 0.0)

    # ---- score ---------------------------------------------------------------
    penalty = sum({1: 2, 2: 5, 3: 9}[i["severity"]] for i in issues)
    score = int(max(0, 100 - penalty))
    issues.sort(key=lambda i: (-i["severity"], i["t"]))

    return {
        "score": score,
        "n_issues": len(issues),
        "variety": round(variety, 3),
        "issues": issues,
        "summary": _summarize(score, issues),
    }


def _summarize(score: int, issues: list[dict]) -> str:
    from collections import Counter
    kinds = Counter(i["type"] for i in issues)
    top = ", ".join(f"{k}×{v}" for k, v in kinds.most_common(4))
    return f"score {score}/100; {len(issues)} issues" + (f" ({top})" if top else "")
