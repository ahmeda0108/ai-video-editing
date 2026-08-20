"""
Synced-lyrics fetch + parse, so lyric underlays land on the exact sung word.

Pulls a time-stamped LRC from lrclib.net (free, no key), caches it, and exposes
the lines as {t_sec, text}. The selector/driver picks which lines to show and
converts their absolute song time into the edit's output-relative time.

    python -m pipeline.lyrics "Jace June" "Come Home"
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional

from . import config

_LRC = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]\s*(.*)")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def fetch(artist: str, track: str, force: bool = False) -> Optional[str]:
    """Return the raw synced-LRC text (cached), or None if unavailable."""
    cache = config.ANALYSIS / "lyrics"
    cache.mkdir(exist_ok=True)
    path = cache / f"{_slug(artist)}__{_slug(track)}.lrc"
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    q = urllib.parse.urlencode({"artist_name": artist, "track_name": track})
    try:
        req = urllib.request.Request("https://lrclib.net/api/get?" + q,
                                     headers={"User-Agent": "autoedit/1.0"})
        d = json.load(urllib.request.urlopen(req, timeout=15))
        synced = d.get("syncedLyrics") or ""
    except Exception:
        synced = ""
    if not synced:
        return None
    path.write_text(synced, encoding="utf-8")
    return synced


def parse_lrc(text: str) -> list[dict]:
    """LRC -> [{t, text}] sorted by time, blank lines dropped."""
    out = []
    for line in text.splitlines():
        m = _LRC.match(line)
        if not m:
            continue
        mm, ss, frac, body = m.groups()
        t = int(mm) * 60 + int(ss) + (int(frac.ljust(3, "0")) / 1000 if frac else 0.0)
        body = body.strip()
        if body:
            out.append({"t": round(t, 2), "text": body})
    return out


def lines(artist: str, track: str) -> list[dict]:
    txt = fetch(artist, track)
    return parse_lrc(txt) if txt else []


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "Jace June"
    t = sys.argv[2] if len(sys.argv) > 2 else "Come Home"
    ls = lines(a, t)
    print(f"{a} - {t}: {len(ls)} synced lines")
    for l in ls[:20]:
        print(f"  {l['t']:6.2f}s  {l['text']}")
