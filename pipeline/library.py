"""
Clip library assembly — the searchable index the editor selects from.

Merges shot geometry (ingest) + visual embeddings (visual) into one record per
clip, and adds the derived fields selection actually queries:

  * quality   0..1  composite of sharpness, colorfulness, good exposure
  * intensity 0..1  normalised motion — used to match clip energy to the music
  * impact    0..1  peak motion — "does something hit here"
  * usable    bool  black / blown / flat / too-short clips are gated OUT

Semantic tags (subject, camera, mood, ...) from the vision provider are merged
in later by vision.tagging; this module leaves a `tags` slot for them.

Output: analysis/clips.json
"""
from __future__ import annotations

import numpy as np

from . import config
from .io_util import load_json, save_json


def _exposure_score(b: float) -> float:
    """1.0 for well-exposed mid frames, decaying toward black/blown."""
    if b < 0.06 or b > 0.97:
        return 0.0
    # peak around 0.45, gentle falloff
    return float(np.clip(1.0 - abs(b - 0.45) / 0.55, 0.0, 1.0))


def build(force: bool = False) -> dict:
    shots_doc = load_json(config.ANALYSIS / "shots.json")
    emb_doc = load_json(config.ANALYSIS / "embeddings.json")
    if not shots_doc or not emb_doc:
        raise RuntimeError("run ingest + visual first")

    emb_by_id = {e["id"]: e for e in emb_doc["shots"]}

    # normalise motion across the library for intensity/impact
    motions = np.array([s["motion_mean"] for s in shots_doc["shots"]]) if shots_doc["shots"] else np.array([0.0])
    maxes = np.array([s["motion_max"] for s in shots_doc["shots"]]) if shots_doc["shots"] else np.array([0.0])
    m_lo, m_hi = np.quantile(motions, 0.05), np.quantile(motions, 0.95)
    x_lo, x_hi = np.quantile(maxes, 0.05), np.quantile(maxes, 0.95)

    def norm(v, lo, hi):
        return float(np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    clips = []
    for s in shots_doc["shots"]:
        e = emb_by_id.get(s["id"])
        if not e:
            continue
        exposure = _exposure_score(s["brightness"])
        sharp = e.get("sharpness_n", 0.0)
        colorf = e.get("colorfulness_n", 0.0)
        intensity = norm(s["motion_mean"], m_lo, m_hi)
        impact = norm(s["motion_max"], x_lo, x_hi)

        # usability gate
        reject = None
        if s["dur"] < config.MIN_SHOT_SEC:
            reject = "too_short"
        elif s["brightness"] < 0.06:
            reject = "black"
        elif s["brightness"] > 0.97:
            reject = "blown_out"
        elif sharp < 0.02 and colorf < 0.05:
            reject = "flat"
        usable = reject is None

        quality = round(float(
            0.42 * sharp + 0.28 * colorf + 0.30 * exposure
        ), 4)

        clips.append({
            "id": s["id"],
            "index": s["index"],
            "source": s.get("source"),
            "source_static": s.get("source_static"),
            "source_fps": s.get("source_fps"),
            "start": s["start"],
            "end": s["end"],
            "dur": s["dur"],
            "motion_mean": s["motion_mean"],
            "motion_peaks": s.get("motion_peaks", []),
            "brightness": s["brightness"],
            "action": s["action"],
            "cluster": e.get("cluster"),
            "phash": e["phash"],
            "color_hist": e["color_hist"],
            "sharpness_n": sharp,
            "colorfulness_n": colorf,
            "exposure": round(exposure, 4),
            "quality": quality,
            "intensity": round(intensity, 4),
            "impact": round(impact, 4),
            "usable": usable,
            "reject_reason": reject,
            "keyframes": [kf["path"] for kf in s["keyframes"]],
            "tags": {},   # filled by vision.tagging
        })

    n_usable = sum(1 for c in clips if c["usable"])
    result = {
        "_input_hash": shots_doc["_input_hash"],
        "sources": shots_doc.get("sources", []),
        "total_duration": shots_doc.get("total_duration"),
        "n_clips": len(clips),
        "n_usable": n_usable,
        "clips": clips,
    }
    save_json(config.ANALYSIS / "clips.json", result)
    return result


def load() -> dict:
    doc = load_json(config.ANALYSIS / "clips.json")
    if not doc:
        raise RuntimeError("clips.json missing — run library.build()")
    return doc


if __name__ == "__main__":
    r = build(force=True)
    print(f"clips={r['n_clips']} usable={r['n_usable']}")
    rejects = [(c['id'], c['reject_reason']) for c in r['clips'] if not c['usable']]
    print("rejected:", rejects)
    top = sorted([c for c in r['clips'] if c['usable']], key=lambda c: -c['quality'])[:5]
    print("top quality:", [(c['id'], c['quality'], 'act' if c['action'] else '-') for c in top])
    action = sorted([c for c in r['clips'] if c['usable'] and c['action']], key=lambda c: -c['intensity'])[:5]
    print("top intensity (action):", [(c['id'], c['intensity'], c['impact']) for c in action])
