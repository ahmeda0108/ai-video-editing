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


if __name__ == "__main__":
    import sys
    plan = Path(sys.argv[1]) if len(sys.argv) > 1 else config.EDITS / "edit_plan.v1.json"
    out = render(plan)
    print("rendered ->", out)
