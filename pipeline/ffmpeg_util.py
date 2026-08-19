"""
ffmpeg access + small subprocess helpers.

We deliberately do NOT rely on a system ffmpeg (there isn't one on this
machine). Instead we use the static binary bundled by `imageio-ffmpeg`, so the
pipeline is self-contained and reproducible across machines.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg with the given args (ffmpeg binary is prepended)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", *args]
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )


def probe_media(path: Path) -> dict:
    """
    Return {duration, fps, width, height, has_audio} using ffmpeg's stderr
    (imageio-ffmpeg does not ship ffprobe, so we parse the -i banner).
    """
    p = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    txt = p.stderr
    out: dict = {"duration": None, "fps": None, "width": None, "height": None, "has_audio": False}

    # Duration: 00:04:20.43
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            hms = line.split("Duration:")[1].split(",")[0].strip()
            try:
                h, m, s = hms.split(":")
                out["duration"] = int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                pass
        if "Video:" in line:
            # resolution like 1920x1080
            import re
            mres = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if mres:
                out["width"], out["height"] = int(mres.group(1)), int(mres.group(2))
            mfps = re.search(r"([\d.]+)\s+fps", line)
            if mfps:
                out["fps"] = float(mfps.group(1))
        if "Audio:" in line:
            out["has_audio"] = True
    return out


def decode_audio_mono(path: Path, sr: int) -> "list[float]":
    """
    Decode any audio/video's audio track to mono float32 PCM at `sr` Hz.
    Returns a numpy array (import numpy lazily to keep import cheap).
    """
    import numpy as np

    p = subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-nostdin",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", str(sr),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "pipe:1",
        ],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {p.stderr.decode(errors='ignore')[-500:]}")
    return np.frombuffer(p.stdout, dtype="<f4").astype("float32")
