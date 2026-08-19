"""
Vision provider abstraction.

Two responsibilities, behind one swappable interface:
  1. tag_clips()  -> semantic tags per clip (subject, setting, camera, mood...)
  2. critique()   -> structured, timestamped feedback on a rendered draft

Providers:
  * LocalProvider     — no network, no key. Tags from CV metrics; critique is
    the metrics critic (critic_metrics). Always available, so the whole
    build -> render -> critique -> revise loop runs today without credentials.
  * AnthropicProvider — true multimodal understanding via the Claude API.
    Tags keyframes and critiques sampled frames of the rendered video. Enabled
    automatically when ANTHROPIC_API_KEY is in .env.

Both cache results by content hash so unchanged footage / unchanged drafts are
never re-sent to the API (a hard cost rule).
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Optional

from . import config, ffmpeg_util
from . import critic_metrics
from .io_util import load_json, save_json, sha1, file_sig


# ---------------------------------------------------------------------------
class VisionProvider:
    name = "base"

    def tag_clips(self, clips_doc: dict) -> dict:
        raise NotImplementedError

    def critique(self, draft_path: Path, plan: dict, audio: dict, clips_doc: dict) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class LocalProvider(VisionProvider):
    """Metrics-only: derives tags from CV features; critiques via critic_metrics."""
    name = "local"

    _HUE = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]

    def _palette(self, hist: list[float]) -> str:
        # 72-bin HSV hist: 8 hue x 3 sat x 3 val. Sum over sat/val per hue.
        import numpy as np
        h = np.array(hist).reshape(8, 3, 3).sum(axis=(1, 2))
        return self._HUE[int(h.argmax())]

    def tag_clips(self, clips_doc: dict) -> dict:
        tags = {}
        for c in clips_doc["clips"]:
            if not c["usable"]:
                continue
            energy = "high" if c["intensity"] > 0.6 else ("low" if c["intensity"] < 0.25 else "mid")
            brightness = ("dark" if c["brightness"] < 0.3 else
                          "bright" if c["brightness"] > 0.6 else "mid")
            tags[c["id"]] = {
                "kind": "action" if c.get("action") else "scenic",
                "energy": energy,
                "brightness": brightness,
                "palette": self._palette(c["color_hist"]),
                "is_impact": bool(c["impact"] > 0.7),
                "source": "local-metrics",
            }
        return tags

    def critique(self, draft_path: Path, plan: dict, audio: dict, clips_doc: dict) -> dict:
        res = critic_metrics.evaluate(plan, audio, clips_doc)
        res["provider"] = "local"
        return res


# ---------------------------------------------------------------------------
CRITIQUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer"},
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "t": {"type": "number"},
                    "clipId": {"type": "string"},
                    "type": {"type": "string"},
                    "severity": {"type": "integer"},
                    "note": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["t", "type", "severity", "note", "suggestion", "action"],
            },
        },
    },
    "required": ["score", "summary", "issues"],
}


class AnthropicProvider(VisionProvider):
    """Claude multimodal: real semantic tagging + rendered-draft critique."""
    name = "anthropic"

    def __init__(self):
        self.model = config.ANTHROPIC_MODEL
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic SDK not installed. `pip install anthropic` "
                    "or set VISION_PROVIDER=local."
                ) from e
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        return self._client

    @staticmethod
    def _img_block(path: Path) -> dict:
        data = base64.standard_b64encode(path.read_bytes()).decode()
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

    # ---- semantic keyframe tagging ---------------------------------------
    def tag_clips(self, clips_doc: dict) -> dict:
        cache_path = config.CACHE / "tags_anthropic.json"
        cache = load_json(cache_path) or {}
        out = {}
        client = self._client_lazy()
        for c in clips_doc["clips"]:
            if not c["usable"] or not c["keyframes"]:
                continue
            kf = config.ROOT / c["keyframes"][len(c["keyframes"]) // 2]
            if not kf.exists():
                continue
            key = sha1(file_sig(kf))
            if key in cache:                          # never re-send unchanged frames
                out[c["id"]] = cache[key]
                continue
            msg = client.messages.create(
                model=self.model,
                max_tokens=400,
                system=("You tag anime footage frames for a video editor. Reply with "
                        "compact JSON: {subject, setting, camera, mood, action_type, "
                        "composition, is_impact(bool)}. One or two words per field."),
                messages=[{"role": "user", "content": [
                    self._img_block(kf),
                    {"type": "text", "text": "Tag this frame as JSON."},
                ]}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            try:
                tag = json.loads(text[text.find("{"): text.rfind("}") + 1])
            except (json.JSONDecodeError, ValueError):
                tag = {"raw": text[:200]}
            tag["source"] = "anthropic"
            cache[key] = tag
            out[c["id"]] = tag
        save_json(cache_path, cache)
        return out

    # ---- rendered-draft critique -----------------------------------------
    def _sample_frames(self, draft_path: Path, plan: dict) -> list[tuple[float, Path]]:
        """Extract frames at each cut start + each drop for the model to judge."""
        fps = plan["meta"]["fps"]
        times = sorted(set(
            [round(c["trackStart"] / fps, 2) for c in plan["clips"]] +
            [round(d, 2) for d in plan.get("audioInfo", {}).get("drops", [])]
        ))
        # cap to ~24 frames to bound cost
        if len(times) > 24:
            step = len(times) / 24
            times = [times[int(i * step)] for i in range(24)]
        frames = []
        tmp = config.CACHE / "draft_frames"
        tmp.mkdir(exist_ok=True)
        for i, t in enumerate(times):
            dest = tmp / f"f{i:02d}.jpg"
            subprocess.run([
                ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin",
                "-ss", f"{t:.3f}", "-i", str(draft_path),
                "-frames:v", "1", "-vf", "scale=384:-2", "-q:v", "4", "-y", str(dest),
            ], capture_output=True)
            if dest.exists():
                frames.append((t, dest))
        return frames

    def critique(self, draft_path: Path, plan: dict, audio: dict, clips_doc: dict) -> dict:
        # start from the objective metrics critique, then enrich with vision
        base = critic_metrics.evaluate(plan, audio, clips_doc)

        cache_path = config.CACHE / "critique_anthropic.json"
        cache = load_json(cache_path) or {}
        key = sha1(file_sig(draft_path), json.dumps(plan["clips"])[:4000])
        if key in cache:
            merged = cache[key]
            merged["provider"] = "anthropic(cached)"
            return merged

        client = self._client_lazy()
        frames = self._sample_frames(draft_path, plan)
        content: list[dict] = []
        for t, p in frames:
            content.append({"type": "text", "text": f"t={t}s"})
            content.append(self._img_block(p))
        sections = plan.get("audioInfo", {}).get("sections", [])
        drops = plan.get("audioInfo", {}).get("drops", [])
        content.append({"type": "text", "text":
            f"Music: {plan['meta']['tempo_bpm']} BPM, drops at {drops}s, "
            f"sections {sections}. These frames are consecutive cuts of an AMV. "
            f"Objective metric notes: {base['summary']}. "
            "Critique the EDIT as a professional AMV editor: pacing, buildup->payoff, "
            "beat-drop emphasis, shot variety, repetition, weak/strong clip placement, "
            "composition. For each problem give t (seconds), type, severity 1-3, a note, "
            "a concrete suggestion, and an action tag (one of: replace, shorten, lengthen, "
            "inject, diversify, reorder). Also give an overall score 0-100."})

        msg = client.messages.create(
            model=self.model,
            max_tokens=2000,
            output_config={"format": {"type": "json_schema", "schema": CRITIQUE_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            vision = json.loads(text)
        except json.JSONDecodeError:
            vision = {"score": base["score"], "summary": text[:300], "issues": []}

        # merge: keep objective issues, append vision issues, average scores
        vissues = vision.get("issues", [])
        for vi in vissues:
            vi.setdefault("action", {"action": vi.get("action", "replace")})
            if isinstance(vi.get("action"), str):
                vi["action"] = {"action": vi["action"]}
        merged = {
            "score": int(round(0.5 * base["score"] + 0.5 * vision.get("score", base["score"]))),
            "summary": f"metrics: {base['summary']} | vision: {vision.get('summary', '')[:160]}",
            "variety": base["variety"],
            "issues": base["issues"] + vissues,
            "provider": "anthropic",
        }
        merged["n_issues"] = len(merged["issues"])
        cache[key] = merged
        save_json(cache_path, cache)
        return merged


# ---------------------------------------------------------------------------
def get_provider(name: Optional[str] = None) -> VisionProvider:
    name = name or config.active_vision_provider()
    if name == "anthropic":
        return AnthropicProvider()
    return LocalProvider()
