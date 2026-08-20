"""
Vision provider abstraction.

Three responsibilities, behind one swappable interface:
  1. tag_clips()     -> semantic tags per clip (subject, setting, camera, mood...)
  2. critique()      -> structured, timestamped feedback on a rendered draft
  3. analyze_image() -> general-purpose structured analysis of one image/frame

Providers:
  * LocalProvider     - no network, no key. Tags from CV metrics; critique is
    the metrics critic (critic_metrics). Always available, so the whole
    build -> render -> critique -> revise loop runs today without credentials.
    Metrics-only: no analyze_image (raises, rather than hallucinating).
  * OllamaProvider    - 100% local, zero-cost multimodal via an Ollama vision
    model (e.g. qwen2.5vl:3b) over Ollama's local HTTP API. No key, no network
    egress, no per-call cost; stdlib-only (adds no pip dependency). This is the
    default whenever a local Ollama server is reachable and no key is set.
  * AnthropicProvider - true multimodal understanding via the Claude API
    (optional, paid). Enabled automatically when ANTHROPIC_API_KEY is in .env.

Every provider caches results by content hash so unchanged footage / unchanged
drafts are never re-analyzed (a hard cost + latency rule).
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

    # True for providers that genuinely "look" at pixels (VLM / multimodal).
    # The orchestrator renders every iteration for semantic providers so the
    # critique judges the actual rendered frames, not just the plan metrics.
    semantic = False

    def tag_clips(self, clips_doc: dict) -> dict:
        raise NotImplementedError

    def critique(self, draft_path: Path, plan: dict, audio: dict, clips_doc: dict) -> dict:
        raise NotImplementedError

    def analyze_image(self, image_path: Path) -> dict:
        """General-purpose single-image analysis (see GENERAL_SCHEMA).

        Metrics-only providers cannot do this; the base implementation makes
        that explicit rather than returning a hallucinated structure.
        """
        raise NotImplementedError(
            f"provider '{self.name}' has no semantic vision. "
            "Use a local VLM (VISION_PROVIDER=ollama) or set ANTHROPIC_API_KEY."
        )


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
def sample_draft_frames(draft_path: Path, plan: dict) -> "list[tuple[float, Path]]":
    """Extract frames at each cut start + each drop for a VLM to judge.

    Shared by every semantic provider (Anthropic, Ollama) so the rendered-draft
    critique always looks at the same, bounded set of representative frames.
    """
    fps = plan["meta"]["fps"]
    times = sorted(set(
        [round(c["trackStart"] / fps, 2) for c in plan["clips"]] +
        [round(d, 2) for d in plan.get("audioInfo", {}).get("drops", [])]
    ))
    # cap to ~24 frames to bound compute (local VLM time / remote cost)
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


# General-purpose single-image / single-frame analysis schema. This is the
# "arbitrary image + video" contract from the design: it is media-agnostic
# (anime, film, gameplay, photos, screen recordings...) and every field is
# allowed to be "unknown"/null rather than guessed. `observed` vs `inferred`
# keeps the model honest about what is actually visible vs deduced.
GENERAL_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "people_count": {"type": ["integer", "null"]},
        "actions": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "emotion": {"type": "array", "items": {"type": "string"}},
        "clothing": {"type": "array", "items": {"type": "string"}},
        "environment": {"type": "string"},
        "setting_kind": {"type": "string"},   # indoor/outdoor/anime/game/... or unknown
        "camera": {
            "type": "object",
            "properties": {
                "shot_type": {"type": "string"},   # close-up/medium/wide/... or unknown
                "movement": {"type": "string"},    # static/pan/handheld/... or unknown
                "angle": {"type": "string"},       # eye-level/high/low/... or unknown
            },
        },
        "composition": {"type": "string"},
        "lighting": {"type": "string"},
        "color_palette": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string"},
        "text": {"type": "string"},               # OCR; "" if none
        "observed": {"type": "array", "items": {"type": "string"}},
        "inferred": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},         # 0..1
    },
    "required": ["scene", "confidence"],
}

def _normalize_general(d: dict) -> dict:
    """Guarantee a complete, consistently-shaped record.

    Small models often OMIT fields they're unsure about. We backfill omissions
    with honest defaults ("unknown" / [] / null) so downstream code always sees
    every key — this is the opposite of hallucinating: absent == unknown.
    """
    out = dict(d)
    for k in ("subjects", "actions", "objects", "emotion", "clothing",
              "color_palette", "observed", "inferred"):
        v = out.get(k)
        out[k] = v if isinstance(v, list) else ([] if v in (None, "") else [v])
    for k in ("scene", "environment", "setting_kind", "composition",
              "lighting", "mood"):
        out[k] = out.get(k) or "unknown"
    out["text"] = out.get("text") or ""
    cam = out.get("camera") if isinstance(out.get("camera"), dict) else {}
    out["camera"] = {"shot_type": cam.get("shot_type") or "unknown",
                     "movement": cam.get("movement") or "unknown",
                     "angle": cam.get("angle") or "unknown"}
    if "people_count" not in out:
        out["people_count"] = None
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out


_GENERAL_PROMPT = (
    "You are a precise, general-purpose vision analyst for a video editor. "
    "The image may be anime, film, TV, real footage, gameplay, a screen "
    "recording, a photo, or animation. Describe ONLY what is visible. "
    "Fill the JSON schema. Use \"unknown\" (or null / empty array) for anything "
    "you cannot reliably determine - never guess. Put things you literally see "
    "in `observed` and reasonable deductions in `inferred`. `text` is OCR of any "
    "on-screen text (\"\" if none). `confidence` is your overall 0..1 certainty."
)


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
    semantic = True

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
        return sample_draft_frames(draft_path, plan)

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
class OllamaProvider(VisionProvider):
    """100% local, zero-cost multimodal via an Ollama vision model.

    Same capabilities as AnthropicProvider (semantic keyframe tags + rendered-
    draft critique + general image analysis) but runs entirely on this machine
    through Ollama's local HTTP API. No API key, no network egress, no per-call
    cost. Uses only the Python stdlib (urllib) so it adds no pip dependency.

    Model is swappable via VISION_MODEL (default a small CPU-friendly VLM).
    Results are cached by content hash exactly like the other providers.
    """
    name = "ollama"
    semantic = True

    def __init__(self):
        self.model = config.VISION_MODEL
        self.host = config.OLLAMA_HOST.rstrip("/")

    # ---- low-level local inference ---------------------------------------
    @staticmethod
    def _b64(path: Path) -> str:
        return base64.standard_b64encode(path.read_bytes()).decode()

    def _chat(self, prompt: str, images: "list[Path]", *,
              schema: Optional[dict] = None, max_tokens: int = 800,
              timeout: float = 180.0) -> str:
        """One local multimodal turn. Returns the raw model text (JSON string).

        Raises a clear, actionable RuntimeError if Ollama isn't running or the
        model isn't pulled - and NEVER silently falls back to a cloud provider.
        """
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [self._b64(p) for p in images],
            }],
            # ask for JSON; pass the schema when the model/runtime supports it
            "format": schema if schema is not None else "json",
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            if e.code == 404 or "not found" in detail.lower():
                raise RuntimeError(
                    f"Ollama model '{self.model}' is not installed.\n"
                    f"  Pull it with:  ollama pull {self.model}\n"
                    f"  Or run:        python -m pipeline.run setup --pull"
                ) from e
            raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach local Ollama at {self.host} ({e.reason}).\n"
                "  Is it installed and running?  Check:  python -m pipeline.run doctor\n"
                "  Install:  winget install Ollama.Ollama   (then it runs as a service)\n"
                "  This pipeline never falls back to a cloud API - vision stays local."
            ) from e
        return (payload.get("message") or {}).get("content", "") or ""

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Tolerant JSON extraction (small local models sometimes add prose)."""
        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    # ---- general-purpose image analysis ----------------------------------
    def analyze_image(self, image_path: Path) -> dict:
        image_path = Path(image_path)
        cache_path = config.CACHE / "analyze_ollama.json"
        cache = load_json(cache_path) or {}
        key = sha1(file_sig(image_path), self.model)
        if key in cache:
            out = dict(cache[key]); out["cached"] = True
            return out
        text = self._chat(_GENERAL_PROMPT, [image_path],
                          schema=GENERAL_SCHEMA, max_tokens=900)
        parsed = self._parse_json(text) or {
            "scene": text[:300], "confidence": 0.0,
            "note": "model did not return valid JSON",
        }
        parsed = _normalize_general(parsed)
        parsed["source"] = f"ollama:{self.model}"
        cache[key] = parsed
        save_json(cache_path, cache)
        return parsed

    # ---- semantic keyframe tagging ---------------------------------------
    def tag_clips(self, clips_doc: dict) -> dict:
        cache_path = config.CACHE / "tags_ollama.json"
        cache = load_json(cache_path) or {}
        out = {}
        tag_prompt = (
            "Tag this footage frame for a video editor. Reply as compact JSON: "
            "{subject, setting, camera, mood, action_type, composition, "
            "is_impact(bool)}. One or two words per field; use \"unknown\" if "
            "not determinable. Do not guess."
        )
        for c in clips_doc["clips"]:
            if not c["usable"] or not c["keyframes"]:
                continue
            kf = config.ROOT / c["keyframes"][len(c["keyframes"]) // 2]
            if not kf.exists():
                continue
            key = sha1(file_sig(kf), self.model)
            if key in cache:                          # never re-analyze unchanged frames
                out[c["id"]] = cache[key]
                continue
            text = self._chat(tag_prompt, [kf], max_tokens=300)
            tag = self._parse_json(text) or {"raw": text[:200]}
            tag["source"] = f"ollama:{self.model}"
            cache[key] = tag
            out[c["id"]] = tag
        save_json(cache_path, cache)
        return out

    # ---- rendered-draft critique -----------------------------------------
    def critique(self, draft_path: Path, plan: dict, audio: dict, clips_doc: dict) -> dict:
        base = critic_metrics.evaluate(plan, audio, clips_doc)

        cache_path = config.CACHE / "critique_ollama.json"
        cache = load_json(cache_path) or {}
        key = sha1(file_sig(draft_path), self.model, json.dumps(plan["clips"])[:4000])
        if key in cache:
            merged = dict(cache[key]); merged["provider"] = "ollama(cached)"
            return merged

        frames = sample_draft_frames(draft_path, plan)
        sections = plan.get("audioInfo", {}).get("sections", [])
        drops = plan.get("audioInfo", {}).get("drops", [])
        times_str = ", ".join(f"{t}s" for t, _ in frames)
        prompt = (
            f"These {len(frames)} frames are consecutive cuts of an edited video "
            f"at times [{times_str}]. Music: {plan['meta']['tempo_bpm']} BPM, "
            f"drops at {drops}s, sections {sections}. Objective metric notes: "
            f"{base['summary']}. Critique the EDIT as a professional editor: "
            "pacing, buildup->payoff, beat-drop emphasis, shot variety, "
            "repetition, weak/strong placement, composition. Reply as JSON: "
            "{score (0-100 int), summary (str), issues: [{t (sec), type, "
            "severity (1-3 int), note, suggestion, action (one of replace, "
            "shorten, lengthen, inject, diversify, reorder)}]}."
        )
        # Critique sends many frames in one local pass; on CPU that can take
        # far longer than a single-image call. Scale the timeout with the frame
        # count so a slow-but-working local run doesn't spuriously "time out".
        timeout = max(300.0, 60.0 * len(frames))
        text = self._chat(prompt, [p for _, p in frames],
                          schema=CRITIQUE_SCHEMA, max_tokens=2000, timeout=timeout)
        vision = self._parse_json(text) or {
            "score": base["score"], "summary": text[:300], "issues": []}

        vissues = vision.get("issues", []) or []
        for vi in vissues:
            if isinstance(vi.get("action"), str):
                vi["action"] = {"action": vi["action"]}
            else:
                vi.setdefault("action", {"action": "replace"})
        merged = {
            "score": int(round(0.5 * base["score"] + 0.5 * vision.get("score", base["score"]))),
            "summary": f"metrics: {base['summary']} | vision: {str(vision.get('summary', ''))[:160]}",
            "variety": base["variety"],
            "issues": base["issues"] + vissues,
            "provider": f"ollama:{self.model}",
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
    if name == "ollama":
        return OllamaProvider()
    return LocalProvider()
