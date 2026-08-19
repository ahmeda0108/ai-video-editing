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
