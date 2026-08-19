"""
Audio analysis — the musical structure the editor cuts to.

Pure numpy DSP (no librosa/torch) so it installs anywhere. From the track we
derive everything the editor needs to feel musical:

  * tempo (BPM) via onset-envelope autocorrelation
  * a phase-aligned BEAT grid (each beat snapped to the nearest real onset)
  * DOWNBEATS (assumed 4/4) — the strong structural pulse we cut hardest on
  * an ENERGY envelope (loudness over time)
  * SECTIONS (low / mid / high energy spans) — verse vs chorus vs drop
  * DROPS (sharp low->high energy transitions = the payoff moments)
  * BUILDUPS (rising energy spans that precede a drop)

Output cached to analysis/audio.json (inspectable, regenerated on input change).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from . import ffmpeg_util
from .io_util import file_sig, load_if_fresh, save_json

SR = config.AUDIO_SR
HOP = config.HOP
WIN = 1024


# ---------------------------------------------------------------------------
# Low-level DSP
# ---------------------------------------------------------------------------
def _stft_mag(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, shape (n_frames, n_bins). Hann window."""
    n = len(y)
    if n < WIN:
        y = np.pad(y, (0, WIN - n))
        n = len(y)
    n_frames = 1 + (n - WIN) // HOP
    window = np.hanning(WIN).astype("float32")
    # framed view
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = y[idx] * window
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).astype("float32")


def _onset_envelope(mag: np.ndarray) -> np.ndarray:
    """Spectral flux: summed positive frame-to-frame magnitude increase."""
    diff = np.diff(mag, axis=0)
    flux = np.maximum(diff, 0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    # normalise + light smoothing
    if flux.max() > 0:
        flux = flux / flux.max()
    kernel = np.hanning(5)
    kernel /= kernel.sum()
    flux = np.convolve(flux, kernel, mode="same")
    return flux.astype("float32")


def _frame_times(n_frames: int) -> np.ndarray:
    return (np.arange(n_frames) * HOP / SR).astype("float32")


def _estimate_tempo(onset: np.ndarray, fps: float) -> float:
    """Tempo (BPM) from the dominant periodicity of the onset envelope."""
    oe = onset - onset.mean()
    corr = np.correlate(oe, oe, mode="full")[len(oe) - 1:]
    # candidate lags for 50..200 BPM
    min_bpm, max_bpm = 60.0, 190.0
    min_lag = int(fps * 60.0 / max_bpm)
    max_lag = int(fps * 60.0 / min_bpm)
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return 120.0
    seg = corr[min_lag:max_lag]
    lag = min_lag + int(np.argmax(seg))
    bpm = 60.0 * fps / lag
    # fold into a musical range
    while bpm < 80:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return float(round(bpm, 2))


def _beat_grid(onset: np.ndarray, times: np.ndarray, bpm: float, duration: float):
    """
    Build a beat grid at `bpm`, choose the phase that best lines up with real
    onsets, then snap each beat to the strongest onset in a small window.
    """
    period = 60.0 / bpm                       # seconds per beat
    fps = SR / HOP
    period_f = period * fps
    # choose phase (0..period) maximising onset energy on the grid
    best_phase, best_score = 0.0, -1.0
    for ph in np.linspace(0, period, 24, endpoint=False):
        grid = np.arange(ph, duration, period)
        gi = np.clip((grid * fps).astype(int), 0, len(onset) - 1)
        score = float(onset[gi].sum())
        if score > best_score:
            best_score, best_phase = score, ph
    beats = np.arange(best_phase, duration, period)
    # snap each beat to the nearest onset peak within a TIGHT +/- 0.14 beat
    # window. One output beat per grid position (no dedupe) so the grid stays
    # regular and count-stable — cutting drifts badly otherwise.
    win = max(1, int(period_f * 0.14))
    out = []
    for b in beats:
        c = int(b * fps)
        lo, hi = max(0, c - win), min(len(onset), c + win + 1)
        if hi > lo:
            local = lo + int(np.argmax(onset[lo:hi]))
            out.append(round(float(times[min(local, len(times) - 1)]), 3))
        else:
            out.append(round(float(b), 3))
    return out


def _energy_envelope(y: np.ndarray, hz: float = 20.0):
    """RMS loudness resampled to `hz` samples/sec, plus its time axis."""
    win = int(SR / hz)
    if win < 1:
        win = 1
    n = len(y) // win
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    trimmed = y[: n * win].reshape(n, win)
    rms = np.sqrt((trimmed ** 2).mean(axis=1) + 1e-9)
    rms = rms / (rms.max() + 1e-9)
    t = np.arange(n) / hz
    return rms.astype("float32"), t.astype("float32")


def _sections(rms: np.ndarray, t: np.ndarray):
    """Segment the song into low/mid/high energy spans via smoothed loudness."""
    # smooth ~1s
    k = max(3, int(len(rms) * 0.03))
    ker = np.ones(k) / k
    sm = np.convolve(rms, ker, mode="same")
    lo_q, hi_q = np.quantile(sm, 0.4), np.quantile(sm, 0.72)

    def level(v: float) -> str:
        if v >= hi_q:
            return "high"
        if v <= lo_q:
            return "low"
        return "mid"

    labels = [level(v) for v in sm]
    sections = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            sections.append({
                "start": round(float(t[start]), 3),
                "end": round(float(t[i]), 3),
                "level": labels[start],
                "energy": round(float(sm[start:i].mean()), 4),
            })
            start = i
    sections.append({
        "start": round(float(t[start]), 3),
        "end": round(float(t[-1]), 3),
        "level": labels[start],
        "energy": round(float(sm[start:].mean()), 4),
    })
    # merge tiny (<1.2s) sections into the previous
    merged = []
    for s in sections:
        if merged and (s["end"] - s["start"]) < 1.2:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    return merged, sm


def _drops(sm: np.ndarray, t: np.ndarray, sections: list[dict]):
    """
    Detect drops = the moment energy jumps from lower to sustained-high.

    Two complementary signals, unioned:
      1) section transitions INTO a "high" span (structural, robust)
      2) a local sliding-window low->high jump (catches drops inside a section)
    """
    hz = 1.0 / (t[1] - t[0]) if len(t) > 1 else 20.0
    hi_q = np.quantile(sm, 0.66)
    cand: list[float] = []

    # (1) structural: entering a high section from a non-high one
    for prev, cur in zip(sections, sections[1:]):
        if cur["level"] == "high" and prev["level"] != "high":
            cand.append(round(float(cur["start"]), 3))

    # (2) local sliding jump
    look = max(1, int(hz * 0.6))
    for i in range(look, len(sm) - look):
        before = sm[i - look:i].mean()
        after = sm[i:i + look].mean()
        if after > hi_q and (after - before) > 0.14:
            cand.append(round(float(t[i]), 3))

    cand.sort()
    drops: list[float] = []
    for tt in cand:
        if not drops or tt - drops[-1] > 2.0:
            drops.append(tt)
    return drops


def _buildups(sm: np.ndarray, t: np.ndarray, drops: list[float]):
    """A buildup is the rising-energy span in the ~4s before each drop."""
    hz = 1.0 / (t[1] - t[0]) if len(t) > 1 else 20.0
    spans = []
    for d in drops:
        di = int(d * hz)
        s0 = max(0, di - int(hz * 4))
        if di - s0 < 2:
            continue
        seg = sm[s0:di]
        # rising if end noticeably higher than start
        if seg[-1] - seg[0] > 0.1:
            spans.append({"start": round(float(t[s0]), 3), "end": round(d, 3)})
    return spans


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def analyze(track: Optional[Path] = None, force: bool = False) -> dict:
    track = Path(track) if track else config.DEFAULT_TRACK
    out_path = config.ANALYSIS / "audio.json"
    ih = file_sig(track)
    if not force:
        cached = load_if_fresh(out_path, ih)
        if cached:
            return cached

    y = ffmpeg_util.decode_audio_mono(track, SR)
    duration = len(y) / SR
    mag = _stft_mag(y)
    onset = _onset_envelope(mag)
    times = _frame_times(len(onset))
    fps = SR / HOP

    bpm = _estimate_tempo(onset, fps)
    beats = _beat_grid(onset, times, bpm, duration)
    downbeats = beats[::4]                       # assume 4/4

    rms, tE = _energy_envelope(y, hz=20.0)
    sections, sm = _sections(rms, tE)
    drops = _drops(sm, tE, sections)
    buildups = _buildups(sm, tE, drops)

    # per-beat energy (so the editor can match clip intensity to the music)
    beat_energy = []
    for b in beats:
        i = min(int(b * 20.0), len(sm) - 1)
        beat_energy.append(round(float(sm[i]), 4))

    result = {
        "_input_hash": ih,
        "track": track.name,
        "duration": round(duration, 3),
        "tempo_bpm": bpm,
        "beats": beats,
        "downbeats": downbeats,
        "beat_energy": beat_energy,
        "energy_hz": 20.0,
        "energy": [round(float(v), 4) for v in sm],
        "sections": sections,
        "drops": drops,
        "buildups": buildups,
    }
    save_json(out_path, result)
    return result


if __name__ == "__main__":
    import json
    r = analyze(force=True)
    print(json.dumps({
        "duration": r["duration"],
        "tempo_bpm": r["tempo_bpm"],
        "n_beats": len(r["beats"]),
        "first_beats": r["beats"][:8],
        "n_downbeats": len(r["downbeats"]),
        "sections": r["sections"],
        "drops": r["drops"],
        "buildups": r["buildups"],
    }, indent=2))
