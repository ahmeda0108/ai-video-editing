"""
Movie transcript ingestion — the story spine.

Extracts embedded subtitle tracks from the source film and parses them into a
timestamped transcript. Two products, both keyed to the movie's own timeline
(which matches clip `sourceStart` seconds, so lines correlate to shots):

  * dialogue[]    -> {start, end, text}         (what's said, when)
  * sound_cues[]  -> {t, cue}                    ([explosion], [scream], ...)

Dialogue gives emotional temperature + character-name mentions for story.py;
sound cues are a free map of the film's hard-hitting audio moments (great for
choosing diegetic hits). A movie usually ships several subtitle tracks — a
"forced" signs/songs track, a clean dialogue track, and an SDH track that also
brackets sound effects. We auto-classify them.

Output: analysis/transcript.json (cached by content hash).

    python -m pipeline.transcript "<movie.mkv>"
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from . import config, ffmpeg_util
from .io_util import load_json, save_json, sha1

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_CUE = re.compile(r"\[([^\]]{1,40})\]")            # [explosion], [footsteps], ...
_LEAD = re.compile(r"^\[([^\]]{1,40})\]\s*")       # a leading [label]
_TAG = re.compile(r"<[^>]+>")                       # <i> ... </i>

# Contract character ids (VISION_TAGS_CONTRACT.md). SDH speaker labels are
# Capitalized names ([Denji]); sound effects are lowercase verbs ([gasps]).
_CHAR_IDS = {"denji", "reze", "makima", "aki", "power", "pochita"}


def _norm_speaker(label: str) -> Optional[str]:
    """A leading [label] -> normalized character id, 'other' for another named
    speaker (Capitalized), or None if it's a sound effect (lowercase)."""
    w = label.strip()
    if not w or not w[0].isupper():
        return None                       # lowercase -> sound effect, not a speaker
    key = re.sub(r"[^a-z]", "", w.lower())
    return key if key in _CHAR_IDS else "other"


def _movie() -> Path:
    # MOVIE_SOURCE env (or .env, loaded by config) points at the subtitled film.
    src = os.environ.get("MOVIE_SOURCE", "")
    if not src:
        raise RuntimeError("no movie source; set MOVIE_SOURCE env or pass a path")
    return Path(src)


def _sub_tracks(movie: Path) -> list[dict]:
    """List subtitle streams: [{index, lang, forced}] via ffmpeg -i parsing."""
    proc = subprocess.run([ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-i", str(movie)],
                          capture_output=True, text=True)
    tracks = []
    for line in proc.stderr.splitlines():
        m = re.search(r"Stream #0:(\d+)\(?(\w+)?\)?: Subtitle", line)
        if m:
            tracks.append({"index": int(m.group(1)), "lang": m.group(2) or "und",
                           "forced": "forced" in line.lower()})
    return tracks


def _extract(movie: Path, index: int, dest: Path) -> str:
    subprocess.run([ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
                    "-i", str(movie), "-map", f"0:{index}", str(dest)],
                   capture_output=True, text=True)
    return dest.read_text(encoding="utf-8", errors="ignore") if dest.exists() else ""


def _sec(ts: str) -> Optional[float]:
    m = _TS.match(ts)
    if not m:
        return None
    h, mm, s, ms = map(int, m.groups())
    return h * 3600 + mm * 60 + s + ms / 1000.0


def parse_srt(text: str) -> list[dict]:
    """SRT text -> [{start, end, text}] (tags stripped, blanks dropped)."""
    out = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        arrow = next((l for l in lines if "-->" in l), None)
        if not arrow:
            continue
        a, _, b = arrow.partition("-->")
        start, end = _sec(a.strip()), _sec(b.strip())
        if start is None or end is None:
            continue
        body_lines = lines[lines.index(arrow) + 1:]
        body = _TAG.sub("", " ".join(body_lines)).strip()
        if body:
            out.append({"start": round(start, 3), "end": round(end, 3), "text": body})
    return out


def _classify(parsed: dict[int, list[dict]]) -> tuple[int, Optional[int]]:
    """Pick (dialogue_track, sdh_track): dialogue = most non-cue lines; sdh =
    the track with the most bracketed sound cues (may be the same or None)."""
    def dialogue_count(lines):
        return sum(1 for l in lines if not (l["text"].startswith("[") and l["text"].endswith("]")))

    def cue_count(lines):
        return sum(1 for l in lines if _CUE.search(l["text"]))

    dialogue = max(parsed, key=lambda i: dialogue_count(parsed[i]))
    sdh_candidates = [(i, cue_count(parsed[i])) for i in parsed]
    sdh = max(sdh_candidates, key=lambda t: t[1])
    return dialogue, (sdh[0] if sdh[1] > 5 else None)


def build(movie: Optional[Path] = None, force: bool = False) -> dict:
    movie = Path(movie) if movie else _movie()
    tracks = _sub_tracks(movie)
    if not tracks:
        raise RuntimeError(f"no subtitle tracks in {movie.name}")

    tmp = config.CACHE / "subs"
    tmp.mkdir(exist_ok=True)
    parsed: dict[int, list[dict]] = {}
    for t in tracks:
        txt = _extract(movie, t["index"], tmp / f"t{t['index']}.srt")
        p = parse_srt(txt)
        if p:
            parsed[t["index"]] = p
    if not parsed:
        raise RuntimeError("subtitle tracks present but none parsed")

    sig = sha1(movie.name, str(sorted((i, len(v)) for i, v in parsed.items())))
    out_path = config.ANALYSIS / "transcript.json"
    cached = load_json(out_path)
    if not force and isinstance(cached, dict) and cached.get("_input_hash") == sig:
        return cached

    di, si = _classify(parsed)
    primary = si if si is not None else di       # SDH has speakers + cues; prefer it

    dialogue: list[dict] = []
    sound_cues: list[dict] = []
    for l in parsed[primary]:
        text = l["text"]
        speaker = None
        lead = _LEAD.match(text)
        if lead:
            sp = _norm_speaker(lead.group(1))
            if sp:                               # leading [Name] -> speaker
                speaker = sp
            else:                                # leading [sfx] -> sound cue
                sound_cues.append({"t": l["start"], "cue": lead.group(1).lower().strip()})
            text = _LEAD.sub("", text, count=1)
        for m in _CUE.finditer(text):            # any remaining bracket -> sound cue
            sound_cues.append({"t": l["start"], "cue": m.group(1).lower().strip()})
        text = _CUE.sub("", text).strip()
        if text:
            dialogue.append({"start": l["start"], "end": l["end"],
                             "text": text, "speaker": speaker})

    # per-character speaking time-ranges (transcript-based presence signal)
    speakers: dict[str, list] = {}
    for d in dialogue:
        if d["speaker"]:
            speakers.setdefault(d["speaker"], []).append([d["start"], d["end"]])

    result = {
        "_input_hash": sig,
        "movie": movie.name,
        "dialogue_track": di,
        "sdh_track": si,
        "n_dialogue": len(dialogue),
        "n_cues": len(sound_cues),
        "speaker_lines": {k: len(v) for k, v in speakers.items()},
        "dialogue": dialogue,
        "sound_cues": sound_cues,
    }
    save_json(out_path, result)
    return result


if __name__ == "__main__":
    import sys
    mv = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    r = build(mv, force=True)
    print(f"movie={r['movie']}  dialogue_track={r['dialogue_track']} "
          f"sdh_track={r['sdh_track']}")
    print(f"dialogue lines={r['n_dialogue']}  sound cues={r['n_cues']}")
    print("speaker lines:", r["speaker_lines"])
    print("first labeled dialogue:")
    shown = 0
    for l in r["dialogue"]:
        if l["speaker"]:
            print(f"  {l['start']:7.1f}  [{l['speaker']}] {l['text'][:60]}")
            shown += 1
        if shown >= 5:
            break
    from collections import Counter
    cc = Counter(c["cue"] for c in r["sound_cues"])
    print("top sound cues:", dict(cc.most_common(10)))
