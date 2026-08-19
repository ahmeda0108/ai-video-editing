"""
Central configuration + paths for the autonomous editing pipeline.

Everything the pipeline produces is written under the repo so it stays
inspectable. Paths are resolved relative to the repo root (parent of this
file's `pipeline/` dir), never the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- repo layout ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # project root
PIPELINE = ROOT / "pipeline"
PUBLIC = ROOT / "public"                                # Remotion staticFile() root
ANALYSIS = ROOT / "analysis"                            # cached intermediate data
KEYFRAMES = ANALYSIS / "keyframes"                      # extracted jpgs per shot
EDITS = ROOT / "edits"                                  # versioned edit plans + critiques
OUT = ROOT / "out"                                      # rendered draft mp4s
CACHE = ANALYSIS / "cache"                              # vision-provider response cache

for _d in (ANALYSIS, KEYFRAMES, EDITS, OUT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ---- default media --------------------------------------------------------
# These are the repo's sample assets. Override on the CLI for other projects.
DEFAULT_FOOTAGE = PUBLIC / "footage.mp4"
DEFAULT_TRACK = PUBLIC / "track_v1.wav"

# ---- render target --------------------------------------------------------
FPS = 30
WIDTH = 1920
HEIGHT = 1080
COMPOSITION_ID = "AutoEdit"

# ---- analysis knobs -------------------------------------------------------
# Scene detection sensitivity (ffmpeg scdet threshold, 0..100). Lower = more cuts.
SCENE_THRESHOLD = 8.0
MIN_SHOT_SEC = 0.4          # discard/merge shots shorter than this
MAX_SHOT_SEC = 10.0         # very long shots get sub-sampled for keyframes

# Adaptive keyframe sampling. Fast/high-motion shots must not be under-sampled,
# or fight impacts get lost. We sample MORE frames when motion is high.
KF_MIN_PER_SHOT = 1
KF_MAX_PER_SHOT = 6
KF_LONG_EDGE = 384          # keyframe jpeg long-edge px (small = cheap vision calls)

# Audio
AUDIO_SR = 22050
HOP = 512                   # onset-envelope hop -> ~43 fps analysis rate

# Similarity / dedup
DEDUP_HAMMING = 8           # phash hamming distance under which shots are "similar"

# ---- environment / .env ---------------------------------------------------
def load_dotenv() -> None:
    """Minimal .env loader (no external dependency). Never logs values."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


load_dotenv()

# Which vision provider to use for semantic tagging + critique.
#   "auto"      -> anthropic if ANTHROPIC_API_KEY present, else local
#   "anthropic" -> force Anthropic multimodal
#   "local"     -> force local metrics-only heuristics (no network, no key)
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "auto")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Vision model id (multimodal). Swappable; kept current. Defaults to the most
# capable Claude vision model; set ANTHROPIC_MODEL=claude-haiku-4-5 (or
# claude-sonnet-5) in .env to trade some quality for lower API cost.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")


def has_anthropic() -> bool:
    return bool(ANTHROPIC_API_KEY)


def active_vision_provider() -> str:
    if VISION_PROVIDER == "auto":
        return "anthropic" if has_anthropic() else "local"
    return VISION_PROVIDER
