"""
Footage ingestion: shot detection + adaptive keyframe extraction.

Accepts RAW footage — a single file, or a folder / list of files — and turns it
into a clip library. For each source video:
  1. Detect scene cuts (ffmpeg scene score) -> shot boundaries.
  2. Pull motion/brightness stats from the global motion pass (motion.py),
     skipping the cut frame itself.
  3. Extract ADAPTIVE keyframes: more frames for longer / higher-motion shots;
     for high-motion shots the keyframes are taken at motion PEAKS (impact
     moments) rather than uniformly — so fast fight impacts are preserved.

Every shot carries its `source` (so a folder of many clips merges into one
library) and a `source_static` path (relative to public/) so Remotion can load
it with staticFile().

Output: analysis/shots.json  + jpg keyframes under analysis/keyframes/.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import config, ffmpeg_util, motion as motion_mod
from .io_util import file_sig, load_json, save_json, sha1

_PTS = re.compile(r"pts_time:([0-9.]+)")
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg"}


# ---------------------------------------------------------------------------
def discover_sources(sources: Union[Path, str, list, None]) -> list[Path]:
    """Resolve a file / folder / list into a sorted list of video files."""
    if sources is None:
        return [config.DEFAULT_FOOTAGE]
    if isinstance(sources, (str, Path)):
        p = Path(sources)
        if p.is_dir():
            return sorted(f for f in p.iterdir() if f.suffix.lower() in _VIDEO_EXT)
        return [p]
    out: list[Path] = []
    for s in sources:
        out.extend(discover_sources(s))
    return out


def _static_path(footage: Path) -> Optional[str]:
    """Path relative to public/ (for Remotion staticFile), or None if outside."""
    try:
        return str(footage.resolve().relative_to(config.PUBLIC.resolve())).replace("\\", "/")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
def _scene_cuts(footage: Path, threshold: float) -> list[float]:
    proc = subprocess.run(
        [
            ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin",
            "-i", str(footage),
            "-filter:v", f"scale=320:-1,select='gt(scene,{threshold})',showinfo",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    cuts = []
    for line in proc.stderr.splitlines():
        if "showinfo" in line and "pts_time" in line:
            m = _PTS.search(line)
            if m:
                cuts.append(float(m.group(1)))
    return sorted(set(cuts))


def _build_shots(cuts: list[float], duration: float) -> list[dict]:
    bounds = [0.0] + [c for c in cuts if 0.05 < c < duration - 0.05] + [duration]
    bounds = sorted(set(round(b, 3) for b in bounds))
    shots = [{"start": a, "end": b, "dur": round(b - a, 3)} for a, b in zip(bounds, bounds[1:])]
    merged: list[dict] = []
    for s in shots:
        if merged and s["dur"] < config.MIN_SHOT_SEC:
            merged[-1]["end"] = s["end"]
            merged[-1]["dur"] = round(merged[-1]["end"] - merged[-1]["start"], 3)
        else:
            merged.append(dict(s))
    if len(merged) >= 2 and merged[-1]["dur"] < config.MIN_SHOT_SEC:
        merged[-2]["end"] = merged[-1]["end"]
        merged[-2]["dur"] = round(merged[-2]["end"] - merged[-2]["start"], 3)
        merged.pop()
    return merged


def _shot_motion(shot: dict, motion: np.ndarray, bright: np.ndarray, fps: float):
    i0 = int(round(shot["start"] * fps)) + 1     # +1 skips the cut spike
    i1 = max(i0 + 1, int(round(shot["end"] * fps)))
    i0 = min(i0, len(motion) - 1)
    i1 = min(i1, len(motion))
    seg = motion[i0:i1]
    bseg = bright[i0:i1]
    if len(seg) == 0:
        return 0.0, 0.0, 0.5, []
    m_mean = float(seg.mean())
    m_max = float(seg.max())
    b_mean = float(bseg.mean()) if len(bseg) else 0.5
    peaks = []
    if len(seg) >= 3:
        for k in range(1, len(seg) - 1):
            if seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1] and seg[k] > m_mean:
                peaks.append((seg[k], (i0 + k) / fps))
        peaks.sort(reverse=True)
    peak_times = [round(t, 3) for _, t in peaks]
    return m_mean, m_max, b_mean, peak_times


def _adaptive_kf_times(shot: dict, peak_times: list[float], motion_hi: bool) -> list[float]:
    dur = shot["dur"]
    n = 1 + int(dur // 2.0)
    if motion_hi:
        n += 2
    n = max(config.KF_MIN_PER_SHOT, min(config.KF_MAX_PER_SHOT, n))
    times: list[float] = []
    if motion_hi and peak_times:
        for t in peak_times:
            if all(abs(t - u) > 0.25 for u in times):
                times.append(t)
            if len(times) >= n:
                break
    if len(times) < n:
        lo, hi = shot["start"] + dur * 0.12, shot["end"] - dur * 0.12
        if hi <= lo:
            lo, hi = shot["start"], shot["end"]
        extra = np.linspace(lo, hi, n - len(times) + 2)[1:-1] if n - len(times) > 0 else []
        for f in extra:
            times.append(round(float(f), 3))
    if not times:
        times = [round(shot["start"] + dur / 2, 3)]
    return sorted(times)[:n]


def _extract_keyframe(footage: Path, t: float, dest: Path) -> Optional[dict]:
    long_edge = config.KF_LONG_EDGE
    vf = f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    proc = subprocess.run(
        [
            ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin",
            "-ss", f"{t:.3f}", "-i", str(footage),
            "-frames:v", "1", "-vf", vf, "-q:v", "3", "-y", str(dest),
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not dest.exists():
        return None
    return {"t": round(t, 3), "path": str(dest.relative_to(config.ROOT)).replace("\\", "/")}


# ---------------------------------------------------------------------------
def _analyze_one(footage: Path, src_idx: int, force: bool) -> list[dict]:
    info = ffmpeg_util.probe_media(footage)
    duration = info["duration"] or 0.0
    static = _static_path(footage)

    mo = motion_mod.compute(footage, force=force)
    motion = np.array(mo["motion"], dtype="float32")
    bright = np.array(mo["brightness"], dtype="float32")
    afps = mo["fps"]

    cuts = _scene_cuts(footage, config.SCENE_THRESHOLD / 100.0 * 3.0)
    shots = _build_shots(cuts, duration)

    per_shot_mean = [_shot_motion(s, motion, bright, afps)[0] for s in shots]
    hi_thresh = float(np.quantile(per_shot_mean, 0.72)) if per_shot_mean else 0.0

    out_shots = []
    for idx, s in enumerate(shots):
        m_mean, m_max, b_mean, peaks = _shot_motion(s, motion, bright, afps)
        motion_hi = m_mean >= hi_thresh and m_mean > 0.006
        kf_times = _adaptive_kf_times(s, peaks, motion_hi)
        keyframes = []
        for j, t in enumerate(kf_times):
            dest = config.KEYFRAMES / f"src{src_idx:02d}_shot{idx:03d}_{j}.jpg"
            kf = _extract_keyframe(footage, t, dest)
            if kf:
                keyframes.append(kf)
        out_shots.append({
            "id": f"s{src_idx:02d}_{idx:03d}",
            "src_idx": src_idx,
            "index": idx,
            "source": str(footage).replace("\\", "/"),
            "source_static": static,
            "source_fps": info["fps"],
            "source_duration": round(duration, 3),
            "start": s["start"],
            "end": s["end"],
            "dur": s["dur"],
            "motion_mean": round(m_mean, 5),
            "motion_max": round(m_max, 5),
            "brightness": round(b_mean, 4),
            "action": bool(motion_hi),
            "motion_peaks": peaks[:6],
            "keyframes": keyframes,
        })
    return out_shots


def analyze(sources: Union[Path, str, list, None] = None, force: bool = False) -> dict:
    files = discover_sources(sources)
    files = [f for f in files if f.exists()]
    if not files:
        raise RuntimeError(f"no source video files found for {sources!r}")

    combined_sig = sha1(*[file_sig(f) for f in files])
    out_path = config.ANALYSIS / "shots.json"
    cached = load_json(out_path)
    if not force and isinstance(cached, dict) and cached.get("_input_hash") == combined_sig:
        return cached

    # fresh keyframes
    for f in config.KEYFRAMES.glob("*.jpg"):
        f.unlink()

    all_shots: list[dict] = []
    for si, footage in enumerate(files):
        all_shots.extend(_analyze_one(footage, si, force))

    result = {
        "_input_hash": combined_sig,
        "sources": [str(f).replace("\\", "/") for f in files],
        "n_sources": len(files),
        "total_duration": round(sum(s["source_duration"] for s in all_shots[:1]) if all_shots else 0, 3),
        "n_shots": len(all_shots),
        "shots": all_shots,
    }
    # recompute total duration properly (sum of unique source durations)
    seen = {}
    for s in all_shots:
        seen[s["source"]] = s["source_duration"]
    result["total_duration"] = round(sum(seen.values()), 3)
    save_json(out_path, result)
    return result


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    r = analyze(src, force=True)
    shots = r["shots"]
    n_action = sum(1 for s in shots if s["action"])
    durs = [s["dur"] for s in shots]
    n_kf = sum(len(s["keyframes"]) for s in shots)
    print(f"sources={r['n_sources']} shots={r['n_shots']} action={n_action} keyframes={n_kf}")
    print(f"dur min/med/max={min(durs):.2f}/{sorted(durs)[len(durs)//2]:.2f}/{max(durs):.2f}")
