"""
Visual embeddings, similarity + redundancy detection (fully local, no API).

Per shot we compute, from its keyframes:
  * pHash        -> 64-bit perceptual hash (near-duplicate detection)
  * color_hist   -> coarse HSV histogram (mood / palette similarity)
  * sharpness    -> variance-of-Laplacian proxy (focus/quality)
  * colorfulness -> Hasler-Susstrunk colorfulness (visual interest)

Then we cluster visually redundant shots (same location/framing that would read
as repetition if cut back-to-back). Everything here is deterministic and cheap
so it re-runs freely.

Output: analysis/embeddings.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import config
from .io_util import file_sig, load_json, save_json

# 32x32 DCT basis for pHash, precomputed once.
_N = 32
_k = np.arange(_N)
_DCT = np.cos(np.pi * (2 * _k[:, None] + 1) * _k[None, :] / (2 * _N)).astype("float32")


def _phash(gray32: np.ndarray) -> int:
    """64-bit perceptual hash from a 32x32 grayscale array."""
    d = _DCT @ gray32 @ _DCT.T          # 2D DCT-II
    low = d[:8, :8].copy()
    low[0, 0] = 0.0                     # drop DC term
    med = np.median(low)
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _color_hist(rgb: np.ndarray) -> list[float]:
    """Coarse HSV histogram (8 hue x 3 sat x 3 val = 72 bins), L1-normalised."""
    import colorsys
    small = rgb[::4, ::4, :].reshape(-1, 3) / 255.0
    # vectorised rgb->hsv is verbose; use max/min formulation
    r, g, b = small[:, 0], small[:, 1], small[:, 2]
    mx = small.max(axis=1)
    mn = small.min(axis=1)
    diff = mx - mn + 1e-9
    hue = np.zeros_like(mx)
    mask = mx == r
    hue[mask] = ((g - b) / diff)[mask] % 6
    mask = mx == g
    hue[mask] = ((b - r) / diff)[mask] + 2
    mask = mx == b
    hue[mask] = ((r - g) / diff)[mask] + 4
    hue = (hue / 6.0) % 1.0
    sat = diff / (mx + 1e-9)
    val = mx
    hb = np.clip((hue * 8).astype(int), 0, 7)
    sb = np.clip((sat * 3).astype(int), 0, 2)
    vb = np.clip((val * 3).astype(int), 0, 2)
    idx = hb * 9 + sb * 3 + vb
    hist = np.bincount(idx, minlength=72).astype("float32")
    hist /= hist.sum() + 1e-9
    return [round(float(x), 5) for x in hist]


def _sharpness(gray: np.ndarray) -> float:
    """Variance of a Laplacian-ish filter — higher = sharper/more detail."""
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.var(gx) + np.var(gy))


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler-Susstrunk colorfulness metric (normalised roughly to 0..1)."""
    r, g, b = rgb[..., 0].astype("float32"), rgb[..., 1].astype("float32"), rgb[..., 2].astype("float32")
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float((std + 0.3 * mean) / 255.0)


def _load(path: Path):
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    gray = np.asarray(img.convert("L"), dtype="float32")
    gray32 = np.asarray(img.convert("L").resize((_N, _N)), dtype="float32")
    return rgb, gray, gray32


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def compute(force: bool = False) -> dict:
    shots_doc = load_json(config.ANALYSIS / "shots.json")
    if not shots_doc:
        raise RuntimeError("run ingest first (no shots.json)")

    out_path = config.ANALYSIS / "embeddings.json"
    ih = shots_doc["_input_hash"]
    cached = load_json(out_path)
    if not force and isinstance(cached, dict) and cached.get("_input_hash") == ih:
        return cached

    entries = []
    for shot in shots_doc["shots"]:
        kfs = shot["keyframes"]
        if not kfs:
            continue
        phashes, hists, sharps, colorfs = [], [], [], []
        for kf in kfs:
            p = config.ROOT / kf["path"]
            if not p.exists():
                continue
            rgb, gray, gray32 = _load(p)
            phashes.append(_phash(gray32))
            hists.append(_color_hist(rgb))
            sharps.append(_sharpness(gray / 255.0))
            colorfs.append(_colorfulness(rgb))
        if not phashes:
            continue
        # representative = median keyframe (middle by index after sort is fine)
        rep = len(phashes) // 2
        mean_hist = [round(float(x), 5) for x in np.mean(hists, axis=0)]
        entries.append({
            "id": shot["id"],
            "index": shot["index"],
            "phash": phashes[rep],
            "phash_all": phashes,
            "color_hist": mean_hist,
            "sharpness": round(float(np.mean(sharps)), 5),
            "colorfulness": round(float(np.mean(colorfs)), 5),
        })

    # ---- redundancy clustering (near-duplicate by pHash hamming) ----------
    n = len(entries)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if _hamming(entries[i]["phash"], entries[j]["phash"]) <= config.DEDUP_HAMMING:
                union(i, j)
    cluster_ids = {}
    for i in range(n):
        r = find(i)
        cluster_ids.setdefault(r, len(cluster_ids))
        entries[i]["cluster"] = cluster_ids[r]

    # normalise sharpness/colorfulness to 0..1 across the library for scoring
    if entries:
        sh = np.array([e["sharpness"] for e in entries])
        cf = np.array([e["colorfulness"] for e in entries])
        sh_n = (sh - sh.min()) / (np.ptp(sh) + 1e-9)
        cf_n = (cf - cf.min()) / (np.ptp(cf) + 1e-9)
        for e, a, b in zip(entries, sh_n, cf_n):
            e["sharpness_n"] = round(float(a), 4)
            e["colorfulness_n"] = round(float(b), 4)
            del e["phash_all"]   # keep the artifact compact

    result = {
        "_input_hash": ih,
        "n_shots": n,
        "n_clusters": len(cluster_ids),
        "dedup_hamming": config.DEDUP_HAMMING,
        "shots": entries,
    }
    save_json(out_path, result)
    return result


if __name__ == "__main__":
    r = compute(force=True)
    print(f"embedded {r['n_shots']} shots -> {r['n_clusters']} visual clusters")
    # show clusters with >1 member (redundant groups)
    from collections import defaultdict
    groups = defaultdict(list)
    for e in r["shots"]:
        groups[e["cluster"]].append(e["id"])
    dupes = {c: ids for c, ids in groups.items() if len(ids) > 1}
    print(f"redundant clusters: {len(dupes)}")
    for c, ids in list(dupes.items())[:6]:
        print(f"  cluster {c}: {ids}")
