"""
Render an edit plan to mp4 via Remotion.

Invokes the Remotion CLI on the AutoEdit composition, passing the plan JSON as
input props. Remotion's calculateMetadata reads dimensions/duration from the
plan, so the same composition renders landscape AMVs or vertical hype reels.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from . import config
from . import ffmpeg_util

# On Windows the npm shims are .cmd files; bare "npx" isn't directly executable.
NPX = "npx.cmd" if os.name == "nt" else "npx"


def render(plan_path: Path, out_path: Optional[Path] = None,
           concurrency: Optional[int] = None, verbose: bool = True) -> Path:
    plan_path = Path(plan_path)
    if out_path is None:
        out_path = config.OUT / (plan_path.stem.replace("edit_plan", "draft") + ".mp4")
    out_path = Path(out_path)

    cmd = [
        NPX, "remotion", "render",
        "src/index.ts", config.COMPOSITION_ID,
        out_path.as_posix(),
        f"--props={plan_path.as_posix()}",
        "--timeout=120000",
    ]
    if concurrency:
        cmd.append(f"--concurrency={concurrency}")
    if not verbose:
        cmd.append("--log=error")

    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(config.ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"remotion render failed (exit {proc.returncode})")
    if not out_path.exists():
        raise RuntimeError(f"render reported success but {out_path} is missing")
    return out_path


def render_chunk(plan_path: Path, start: int, end: int, out_path: Path,
                 concurrency: int = 4, timeout_ms: int = 120000) -> Path:
    """Render only frames [start, end] (inclusive). Used for chunked rendering
    to stay under environment per-process limits, then concatenated.

    `timeout_ms` raises the per-frame delayRender timeout: seeking deep into a
    long, high-bitrate source video for OffthreadVideo extraction can exceed the
    30s default, so we give each frame more headroom."""
    cmd = [
        NPX, "remotion", "render",
        "src/index.ts", config.COMPOSITION_ID,
        Path(out_path).as_posix(),
        f"--props={Path(plan_path).as_posix()}",
        f"--frames={start}-{end}",
        f"--concurrency={concurrency}",
        f"--timeout={timeout_ms}",
        "--log=error",
    ]
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(config.ROOT))
    if proc.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError(f"chunk render {start}-{end} failed")
    return Path(out_path)


def concat(parts: list[Path], out_path: Path) -> Path:
    """Concatenate rendered mp4 chunks (same codec params) into one file.

    The concat demuxer resolves list entries relative to the list file, so we
    write basenames and run ffmpeg from the chunks' directory — absolute
    Windows paths (with a `C:` drive) confuse the demuxer.
    """
    parts = [Path(p) for p in parts]
    workdir = parts[0].parent
    out_path = Path(out_path).resolve()
    listfile = workdir / "_concat.txt"
    listfile.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    proc = subprocess.run([
        ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", "_concat.txt",
        "-c", "copy", str(out_path),
    ], cwd=str(workdir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"concat failed: {proc.stderr[-500:]}")
    return out_path


def render_chunked(plan_path: Path, out_path: Path, chunk: int = 250,
                   concurrency: int = 4) -> Path:
    """Render a plan in memory-safe chunks and concatenate to out_path."""
    from .io_util import load_json
    total = load_json(plan_path)["meta"]["durationInFrames"]
    parts = []
    i = 0
    idx = 0
    while i < total:
        end = min(i + chunk - 1, total - 1)
        part = config.OUT / f"_part{idx:02d}.mp4"
        print(f"[chunk {idx}] frames {i}-{end}")
        render_chunk(plan_path, i, end, part, concurrency=concurrency)
        parts.append(part)
        i = end + 1
        idx += 1
    out = concat(parts, out_path)
    for p in parts:
        p.unlink(missing_ok=True)
    (config.OUT / "_concat.txt").unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "chunk":
        # chunk <plan> <start> <end> <out>
        render_chunk(Path(args[1]), int(args[2]), int(args[3]), Path(args[4]))
    elif args and args[0] == "concat":
        # concat <out> <part> <part> ...
        concat([Path(p) for p in args[2:]], Path(args[1]))
        print("concatenated ->", args[1])
    elif args and args[0] == "chunked":
        # chunked <plan> <out> [chunkSize]
        cs = int(args[3]) if len(args) > 3 else 250
        out = render_chunked(Path(args[1]), Path(args[2]), chunk=cs)
        print("rendered ->", out)
    else:
        plan = Path(args[0]) if args else config.EDITS / "edit_plan.v1.json"
        out = render(plan)
        print("rendered ->", out)
