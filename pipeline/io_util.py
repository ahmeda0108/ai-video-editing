"""Small helpers: content hashing + cached JSON artifacts.

Every expensive stage writes a JSON artifact that embeds the hash of its
inputs. On re-run, if the input hash matches, we reuse the cached artifact
instead of recomputing — this is how we avoid re-analyzing / re-sending
unchanged footage (a hard rule of the pipeline).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def file_sig(path: Path) -> str:
    """Cheap file signature: size + mtime + name. Good enough for a local cache."""
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def save_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_if_fresh(path: Path, input_hash: str) -> Optional[Any]:
    """Return cached artifact iff it exists and its stored _input_hash matches."""
    data = load_json(path)
    if isinstance(data, dict) and data.get("_input_hash") == input_hash:
        return data
    return None
