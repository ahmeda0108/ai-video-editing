"""
Reference-image character matching (100% local, zero cost).

Names in `characters` come from matching each shot's keyframe against a small
set of user-supplied reference images -- NOT from the VLM guessing identities
(a 3B VLM can't reliably name franchise characters). Identity is only as good as
the reference set, and the whole thing degrades gracefully:

    refs/<char_id>/*.jpg     -> labeled references you drop in
    (no refs / no embedder)  -> match() returns []  (contract stays valid)

Embedding backend: an ONNX CLIP-style image encoder run via onnxruntime (CPU,
no torch). Put the model at `refs/_model/clip_image.onnx` (see VISION_LOCAL.md).
If onnxruntime or the model is missing, matching is disabled (empty result) and
the pipeline still produces valid tags -- identity just turns on later.

Everything here is pure-local: nothing is uploaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import config

# Closed character-id vocabulary (must match VISION_TAGS_CONTRACT.md).
CHAR_IDS = {"denji", "reze", "makima", "aki", "power", "pochita", "crowd", "other"}

REF_DIR = config.ROOT / "refs"
MODEL_PATH = REF_DIR / "_model" / "clip_image.onnx"
INDEX_PATH = config.CACHE / "ref_index.json"

# cosine similarity above which a reference id is considered present
THRESHOLD = float(__import__("os").environ.get("REF_THRESHOLD", "0.75"))
_INPUT = 224  # CLIP ViT-B/32 input size


# ---------------------------------------------------------------------------
def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32"); b = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_matches(vec, index: dict, threshold: float = THRESHOLD) -> list[str]:
    """Given a shot embedding and {id: [ref_vecs]}, return ids above threshold.

    Pure numpy; fully testable without any model. `crowd` isn't matched here
    (it's a scene attribute, not an identity) -- the producer sets it separately.
    """
    scores = {}
    for cid, vecs in index.items():
        if cid not in CHAR_IDS:
            continue
        s = max((_cosine(vec, v) for v in vecs), default=0.0)
        if s >= threshold:
            scores[cid] = s
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------------------
class _ClipEmbedder:
    """ONNX CLIP image encoder (CPU). Lazily loaded; None if unavailable."""
    _session = None
    _ok: Optional[bool] = None

    @classmethod
    def available(cls) -> bool:
        if cls._ok is not None:
            return cls._ok
        try:
            import onnxruntime  # noqa: F401
            cls._ok = MODEL_PATH.exists()
        except ImportError:
            cls._ok = False
        return cls._ok

    @classmethod
    def _sess(cls):
        if cls._session is None:
            import onnxruntime
            cls._session = onnxruntime.InferenceSession(
                str(MODEL_PATH), providers=["CPUExecutionProvider"])
        return cls._session

    @classmethod
    def embed(cls, image_path: Path):
        import numpy as np
        from PIL import Image
        im = Image.open(image_path).convert("RGB").resize((_INPUT, _INPUT))
        arr = np.asarray(im, dtype="float32") / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype="float32")
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype="float32")
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
        sess = cls._sess()
        out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        return np.asarray(out, dtype="float32").reshape(-1)


# ---------------------------------------------------------------------------
def build_index(force: bool = False) -> dict:
    """Embed all refs/<id>/*.jpg into {id: [vectors]}, cached to disk."""
    if not force and INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    index: dict = {}
    if not REF_DIR.exists() or not _ClipEmbedder.available():
        return index
    for cid_dir in sorted(REF_DIR.iterdir()):
        if not cid_dir.is_dir() or cid_dir.name.startswith("_"):
            continue
        cid = cid_dir.name.lower()
        if cid not in CHAR_IDS:
            continue
        vecs = []
        for img in sorted(cid_dir.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                try:
                    vecs.append(_ClipEmbedder.embed(img).tolist())
                except Exception:
                    continue
        if vecs:
            index[cid] = vecs
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index), encoding="utf-8")
    return index


def status() -> dict:
    """Human-facing readiness snapshot for the producer / docs."""
    ref_ids = []
    if REF_DIR.exists():
        ref_ids = sorted(d.name.lower() for d in REF_DIR.iterdir()
                         if d.is_dir() and not d.name.startswith("_")
                         and d.name.lower() in CHAR_IDS)
    return {
        "refs_dir": str(REF_DIR),
        "ref_ids": ref_ids,
        "embedder_available": _ClipEmbedder.available(),
        "model_path": str(MODEL_PATH),
        "enabled": bool(ref_ids) and _ClipEmbedder.available(),
    }


class Matcher:
    """Reusable matcher: builds the ref index once, matches many frames."""
    def __init__(self, force_index: bool = False):
        self.index = build_index(force=force_index)
        self.enabled = bool(self.index) and _ClipEmbedder.available()

    def match(self, image_path: Path) -> list[str]:
        if not self.enabled:
            return []
        try:
            vec = _ClipEmbedder.embed(Path(image_path))
        except Exception:
            return []
        return best_matches(vec, self.index)
