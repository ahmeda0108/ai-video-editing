"""
Global motion + brightness time-series for the whole source video.

We decode the ENTIRE video once as tiny grayscale frames (128x72) at a fixed
analysis fps, then compute frame-to-frame difference energy. This single cheap
pass gives us:

  * motion[t]      -> how much is moving (fight/impact detector)
  * brightness[t]  -> exposure (dark scene vs bright flash)

Crucially, the analysis fps here is high enough (12fps) that fast action still
registers strong motion. We do NOT use this low-res pass to *pick* impact
frames directly — instead we locate motion peaks and later extract full-res
keyframes exactly at those timestamps (see ingest.py). That's how fast fight
impacts survive instead of being averaged away by naive low-fps sampling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from . import ffmpeg_util
from .io_util import file_sig, load_json, save_json

ANALYSIS_FPS = 12.0
W, H = 128, 72


def _cache_path(footage: Path) -> Path:
    from .io_util import sha1
    return config.CACHE / f"motion_{sha1(file_sig(footage))[:16]}.json"


def compute(footage: Optional[Path] = None, force: bool = False) -> dict:
    footage = Path(footage) if footage else config.DEFAULT_FOOTAGE
    out_path = _cache_path(footage)          # per-file cache (multi-source safe)
    ih = file_sig(footage)
    if not force:
        cached = load_json(out_path)
        if isinstance(cached, dict) and cached.get("_input_hash") == ih:
            return cached

    # binary decode (can't use text=True helper for raw bytes)
    import subprocess
    proc = subprocess.run(
        [
            ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin",
            "-i", str(footage),
            "-vf", f"fps={ANALYSIS_FPS},scale={W}:{H},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gray decode failed: {proc.stderr.decode(errors='ignore')[-400:]}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(buf) // (W * H)
    frames = buf[: n * W * H].reshape(n, W * H)   # (n, px) uint8, kept small

    # Block-process so a 100-min film doesn't blow up memory (upcasting every
    # frame to float32 at once would be gigabytes). uint8 stays; only a block
    # is promoted to int16 at a time.
    brightness = np.empty(n, dtype="float32")
    motion = np.zeros(n, dtype="float32")
    B = 4000
    for s in range(0, n, B):
        e = min(s + B, n)
        brightness[s:e] = frames[s:e].mean(axis=1) / 255.0
        d0 = max(1, s)                              # motion[0] stays 0
        cur = frames[d0:e].astype(np.int16)
        prev = frames[d0 - 1:e - 1].astype(np.int16)
        motion[d0:e] = np.abs(cur - prev).mean(axis=1) / 255.0

    result = {
        "_input_hash": ih,
        "footage": footage.name,
        "fps": ANALYSIS_FPS,
        "n": int(n),
        "brightness": [round(float(b), 4) for b in brightness],
        "motion": [round(float(m), 5) for m in motion],
    }
    save_json(out_path, result)
    return result


def time_to_index(t: float) -> int:
    return int(round(t * ANALYSIS_FPS))


if __name__ == "__main__":
    r = compute(force=True)
    m = np.array(r["motion"])
    print(f"frames={r['n']} @ {r['fps']}fps  motion mean={m.mean():.4f} "
          f"p50={np.percentile(m,50):.4f} p90={np.percentile(m,90):.4f} max={m.max():.4f}")
