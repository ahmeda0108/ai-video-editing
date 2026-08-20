"""
Edit profiles — the "required purpose" knob.

The selection + timeline algorithm is generic; a profile only parametrises it.
This is how the same pipeline produces an AMV, a hype reel, a calm montage, or
a trailer from the same footage + audio, WITHOUT hard-coding any one anime,
song, or style.

Each profile describes:
  * how cut density responds to musical energy (beats-per-clip by level)
  * the transition vocabulary for low / mid / high energy and for drops
  * effect rules (ken-burns on static shots, punch/flash on impacts)
  * grade preset + framing (landscape vs vertical)
"""
from __future__ import annotations

# Transition types understood by the Remotion AutoEdit composition:
#   cut | fade | punch | whip | slide | flash | glitch | swell
PROFILES: dict[str, dict] = {
    # -------- music video / AMV: beat-cut, punchy, saves impact for drops -----
    "amv": {
        "label": "Anime / music video — beat-cut, punchy, impact on drops",
        "orientation": "landscape",
        "beats_per_clip": {"low": 3, "mid": 2, "high": 1},
        "allow_half_beat_on_drop": True,      # denser cutting right at drops
        "transitions": {
            "low": ["fade"],
            "mid": ["cut", "fade"],
            "high": ["cut", "punch", "whip"],
            "drop": ["flash", "punch"],
        },
        "crossfade_frames": 9,
        "hardcut_frames": 0,
        "kenburns_on_static": True,
        "shake_on_action": True,
        "flash_on_drop": True,
        "grade": {
            "preset": "neonDusk", "saturation": 0.9, "contrast": 1.08,
            "brightness": 0.98, "vignette": 0.45, "grain": 0.05, "letterbox": 90,
            "tint": "#243056", "tintOpacity": 0.16,
        },
        "max_reuse": 3,             # a clip may appear at most this many times
        "no_repeat_within": 6,      # not the same clip within N cuts
        "reserve_impact_for_peaks": True,
    },
    # -------- hype hard-cut: sharp beat cuts, NO fades (Bomb Devil / Reze) -----
    "bomb": {
        "label": "Hype hard-cut — Bomb Devil, sharp on-beat cuts, no fades",
        "orientation": "landscape",
        # readable hype: each shot registers. Choruses cut every 2 beats (~0.9s),
        # verses breathe over 3-4 beats — hard-cut but comprehensible, not machine-gun.
        "beats_per_clip": {"low": 4, "mid": 3, "high": 2},
        "allow_half_beat_on_drop": False,   # keep EVERY cut ON a beat (99% on-beat)
        "transitions": {                    # HARD CUTS ONLY — no slow dissolves
            "low": ["cut"],
            "mid": ["cut"],
            "high": ["cut"],
            "drop": ["cut"],
        },
        "crossfade_frames": 0,
        "hardcut_frames": 0,
        "kenburns_on_static": False,        # no slow drift — stay punchy
        "shake_on_action": False,           # no jitter (avoid the "cracky" feel)
        "flash_on_drop": True,              # quick white flash on the DROP clips only
        "flash_on_downbeats": False,        # NOT on every loud downbeat — avoids strobe
        "grade": {
            "preset": "neonDusk", "saturation": 0.95, "contrast": 1.13,
            "brightness": 0.98, "vignette": 0.42, "grain": 0.06, "letterbox": 90,
            "tint": "#2a1030", "tintOpacity": 0.14,
        },
        "max_reuse": 3,
        "no_repeat_within": 5,
        "reserve_impact_for_peaks": True,
    },
    # -------- vision-aware ARC with hard on-beat cuts (tragic action) ----------
    # Driven by select_arc (acts + vision tags + character presence) but styled
    # like the bomb profile: hard cuts on every beat, no fades, readable holds.
    "action_arc": {
        "label": "Vision-aware 3-act arc, hard on-beat cuts (tragic action)",
        "orientation": "landscape",
        "beats_per_clip": {"low": 4, "mid": 3, "high": 2},   # readable, on every beat
        # NB: no cut_on -> cuts on ALL beats (denser/on-beat), not just downbeats
        "allow_half_beat_on_drop": False,
        "transitions": {                    # HARD CUTS ONLY
            "low": ["cut"], "mid": ["cut"], "high": ["cut"], "drop": ["cut"],
        },
        "crossfade_frames": 0,
        "hardcut_frames": 0,
        "kenburns_on_static": False,
        "shake_on_action": False,
        "flash_on_drop": False,
        "grade": {
            "preset": "cinemaCool", "saturation": 0.9, "contrast": 1.1,
            "brightness": 0.98, "vignette": 0.5, "grain": 0.05, "letterbox": 120,
            "tint": "#101826", "tintOpacity": 0.13,
        },
        "max_reuse": 2,
        "no_repeat_within": 6,
        "reserve_impact_for_peaks": False,
        # ---- arc/selection extensions (read by select_arc) ----
        # keep below the fastest segment (2 beats ~0.79s @152bpm) so chorus cuts
        # aren't merged into one long hold; beats_per_clip controls pacing.
        "min_hold_sec": 0.6,
        "diegetic": True,
        "diegetic_max": 2,
        "diegetic_min_gap_sec": 16.0,
        "diegetic_min_impact": 0.85,
        "protect_song_peaks": True,
        "diegetic_volume": 1.3,
        "duck_to": 0.28,
        "music_volume": 0.9,
    },

    # -------- hype reel: fastest, vertical, hard cuts + flashes ---------------
    "hype": {
        "label": "Vertical hype reel — fastest, hard cuts, flashes",
        "orientation": "portrait",
        "beats_per_clip": {"low": 2, "mid": 1, "high": 1},
        "allow_half_beat_on_drop": True,
        "transitions": {
            "low": ["cut", "fade"],
            "mid": ["cut", "punch"],
            "high": ["cut", "punch", "whip", "glitch"],
            "drop": ["flash", "glitch"],
        },
        "crossfade_frames": 6,
        "hardcut_frames": 0,
        "kenburns_on_static": True,
        "shake_on_action": True,
        "flash_on_drop": True,
        "grade": {
            "preset": "punch", "saturation": 1.05, "contrast": 1.12,
            "brightness": 1.0, "vignette": 0.35, "grain": 0.04, "letterbox": 0,
            "tint": "#101018", "tintOpacity": 0.1,
        },
        "max_reuse": 3,
        "no_repeat_within": 5,
        "reserve_impact_for_peaks": True,
    },

    # -------- story: slow narrative cut, diegetic audio, lyric underlays -----
    # Built for emotional/cinematic edits (slower songs). Cuts are long and
    # breathe; clips play in ROUGHLY CHRONOLOGICAL movie order so the story
    # unfolds instead of jumping around; a capped few hard-hitting scenes keep
    # their native movie audio (music ducks under them); impacts get slow-mo.
    "story": {
        "label": "Narrative cinematic — slow holds, chronological, diegetic audio",
        "orientation": "landscape",
        # Cut on DOWNBEATS (bar lines) so every scene-switch lands on the "1" of
        # a bar — the switch is felt with the music instead of on a weak off-beat.
        # With cut_on="downbeats", beats_per_clip counts BARS: 2 bars (~5s) when
        # calm, 1 bar (~2.6s) when intense.
        "cut_on": "downbeats",
        "beats_per_clip": {"low": 2, "mid": 2, "high": 1},   # bars-per-clip here
        "allow_half_beat_on_drop": False,
        "transitions": {
            "low": ["fade"],
            "mid": ["fade"],
            "high": ["fade", "cut"],
            "drop": ["swell"],
        },
        "crossfade_frames": 18,          # slow 0.6s dissolves
        "hardcut_frames": 0,
        "kenburns_on_static": True,      # gentle drift on held shots
        "shake_on_action": False,        # no jitter — let it be composed
        "flash_on_drop": False,          # no strobe
        "grade": {
            "preset": "cinemaCool", "saturation": 0.92, "contrast": 1.05,
            "brightness": 0.99, "vignette": 0.55, "grain": 0.05, "letterbox": 140,
            "tint": "#101a2e", "tintOpacity": 0.14,
        },
        "max_reuse": 1,                  # every cut a different shot
        "no_repeat_within": 10,
        "reserve_impact_for_peaks": False,
        # ---- story-mode extensions (read by select.py) ----
        "narrative": True,               # chronological source-time sweep
        "narrative_weight": 2.4,         # how hard to pull toward chronological order
        "min_hold_sec": 1.6,             # pacing floor: never cut faster than this
        "slowmo_on_impact": 0.65,        # slow-mo factor for impact/drop clips
        "diegetic": True,                # let native movie audio punch through
        "diegetic_max": 3,               # sparse: only a few VERY important hits
        "diegetic_min_gap_sec": 16.0,    # keep them spaced out
        "diegetic_min_impact": 0.85,     # only the hardest-hitting movie audio
        "protect_song_peaks": True,      # never duck during the song's main part
        "diegetic_volume": 1.4,          # amplify native audio so impacts hit hard
        "duck_to": 0.24,                 # music dips well under a diegetic hit
        "music_volume": 0.85,            # base music level
    },

    # -------- montage: calm, longer clips, crossfades ------------------------
    "montage": {
        "label": "Calm montage — longer clips, gentle crossfades",
        "orientation": "landscape",
        "beats_per_clip": {"low": 6, "mid": 4, "high": 3},
        "allow_half_beat_on_drop": False,
        "transitions": {
            "low": ["fade"],
            "mid": ["fade"],
            "high": ["fade", "cut"],
            "drop": ["swell"],
        },
        "crossfade_frames": 16,
        "hardcut_frames": 0,
        "kenburns_on_static": True,
        "shake_on_action": False,
        "flash_on_drop": False,
        "grade": {
            "preset": "warm", "saturation": 0.95, "contrast": 1.03,
            "brightness": 1.0, "vignette": 0.5, "grain": 0.06, "letterbox": 120,
            "tint": "#2a1f14", "tintOpacity": 0.14,
        },
        "max_reuse": 2,
        "no_repeat_within": 8,
        "reserve_impact_for_peaks": False,
    },
}

ORIENTATION_DIMS = {
    "landscape": (1920, 1080),
    "portrait": (1080, 1920),
    "square": (1080, 1080),
}


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choices: {list(PROFILES)}")
    return PROFILES[name]
