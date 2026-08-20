"""
Local-vision setup / doctor.

    python -m pipeline.run doctor          # inspect only: hardware + prereqs + advice
    python -m pipeline.run setup           # doctor, and offer exact install/pull steps
    python -m pipeline.run setup --pull    # actually pull the recommended/VISION_MODEL
    python -m pipeline.run setup --test     # run one real local inference to verify

Design rules honored here:
  * Never silently download a multi-GB model - `--pull` is explicit and prints
    the size first.
  * Detect hardware and recommend a model that actually fits (small VLM on a
    modest CPU/iGPU box; larger only when there's RAM/VRAM for it).
  * Never fall back to a cloud API. If local inference can't run, say exactly
    why and how to fix it.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path

from . import config


# --- approximate on-disk sizes, for honest "what will be downloaded" messaging
MODEL_SIZES_GB = {
    "moondream": 1.7,
    "qwen2.5vl:3b": 3.2,
    "llava-phi3": 2.9,
    "llava": 4.7,
    "qwen2.5vl:7b": 6.0,
    "llama3.2-vision": 7.8,
    "llama3.2-vision:11b": 7.8,
}


# ---------------------------------------------------------------------------
def _total_ram_gb() -> float:
    """Best-effort total RAM in GB, no third-party dependency."""
    try:                                   # Windows
        if platform.system() == "Windows":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS(); m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1024**3, 1)
        # POSIX
        import os
        return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
    except Exception:
        return 0.0


def detect_hardware() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "ram_gb": _total_ram_gb(),
    }


def recommend_model(hw: dict) -> tuple[str, str]:
    """Return (model, reason) sized to the machine's RAM (CPU-only assumed)."""
    ram = hw.get("ram_gb", 0.0)
    if ram and ram < 8:
        return "moondream", f"{ram} GB RAM - smallest reliable VLM (~1.7 GB)."
    if ram and ram < 24:
        return ("qwen2.5vl:3b",
                f"{ram} GB RAM, CPU-class inference - 3B VLM (~3.2 GB) is the "
                "accuracy/speed sweet spot; use `moondream` if you want it faster.")
    return ("qwen2.5vl:7b",
            f"{ram} GB RAM - room for a 7B VLM (~6 GB) for stronger understanding.")


# ---------------------------------------------------------------------------
def _ollama_installed() -> str | None:
    return shutil.which("ollama")


def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(
                f"{config.OLLAMA_HOST.rstrip('/')}/api/tags", timeout=2.0) as r:
            import json
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def _ffmpeg_ok() -> str | None:
    try:
        from . import ffmpeg_util
        exe = ffmpeg_util.ffmpeg_exe()
        return exe if Path(exe).exists() else None
    except Exception:
        return None


def _has(model: str, installed: list[str]) -> bool:
    # tolerate ":latest" suffixing by Ollama
    base = model.split(":")[0]
    return any(m == model or m.split(":")[0] == base for m in installed)


# ---------------------------------------------------------------------------
def doctor() -> dict:
    hw = detect_hardware()
    rec_model, rec_reason = recommend_model(hw)
    configured = config.VISION_MODEL
    ffmpeg = _ffmpeg_ok()
    ollama_bin = _ollama_installed()
    reachable = config.ollama_reachable()
    installed = _ollama_models() if reachable else []

    print("=== Local Vision - doctor ===")
    print(f"OS         : {hw['os']} ({hw['machine']})")
    print(f"CPU        : {hw['cpu']}")
    print(f"RAM        : {hw['ram_gb']} GB")
    print(f"ffmpeg     : {'OK  ' + ffmpeg if ffmpeg else 'MISSING - pip install imageio-ffmpeg'}")
    print(f"Ollama bin : {ollama_bin or 'not found on PATH'}")
    print(f"Ollama srv : {'reachable at ' + config.OLLAMA_HOST if reachable else 'NOT running'}")
    if reachable:
        print(f"models     : {', '.join(installed) or '(none pulled yet)'}")
    print(f"configured : VISION_MODEL={configured}  VISION_PROVIDER={config.VISION_PROVIDER}"
          f"  -> active '{config.active_vision_provider()}'")
    print(f"recommend  : {rec_model}   ({rec_reason})")

    # Actionable next step
    print("\n--- next step ---")
    if not ollama_bin and not reachable:
        print("Ollama is not installed. Install it (one-time), then it runs as a service:")
        print("  Windows : winget install Ollama.Ollama")
        print("  macOS   : brew install ollama   (or download from ollama.com)")
        print("  Linux   : curl -fsSL https://ollama.com/install.sh | sh")
        print("Then:  python -m pipeline.run setup --pull --test")
    elif not reachable:
        print("Ollama is installed but the server isn't answering. Start it:")
        print("  ollama serve      (or just run the Ollama app once)")
    elif not _has(configured, installed):
        size = MODEL_SIZES_GB.get(configured)
        sz = f" (~{size} GB download)" if size else ""
        print(f"Server is up. Pull your model{sz}:")
        print(f"  ollama pull {configured}")
        print("  or:  python -m pipeline.run setup --pull --test")
    else:
        print(f"All set. Local vision is ready with '{configured}'.")
        print("  Try:  python -m pipeline.run vision <image-or-video>")
        print("  Or run the edit loop with local semantic critique:")
        print("        python -m pipeline.run all --profile amv --provider ollama")

    return {"hardware": hw, "recommend": rec_model, "ffmpeg": bool(ffmpeg),
            "ollama_installed": bool(ollama_bin), "ollama_reachable": reachable,
            "installed_models": installed, "configured_model": configured,
            "model_present": _has(configured, installed)}


def pull(model: str | None = None) -> bool:
    """Explicitly pull a model (prints size first). Returns success."""
    model = model or config.VISION_MODEL
    if not _ollama_installed():
        print("Cannot pull: Ollama is not installed. Run `doctor` for install steps.")
        return False
    size = MODEL_SIZES_GB.get(model)
    print(f"Pulling '{model}'"
          + (f" - about {size} GB will be downloaded to your Ollama cache."
             if size else " (download size unknown)."))
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Pull failed: {e}")
        return False


def test_inference() -> bool:
    """Run ONE real local inference to prove the stack works end to end."""
    from . import vision_provider
    # prefer a real keyframe; otherwise synthesize a tiny test image
    kfs = sorted(config.KEYFRAMES.glob("*.jpg"))
    if kfs:
        img = kfs[0]
    else:
        from PIL import Image, ImageDraw
        img = config.CACHE / "vision_selftest.jpg"
        im = Image.new("RGB", (384, 216), (18, 22, 40))
        d = ImageDraw.Draw(im)
        d.rectangle([40, 60, 180, 170], fill=(210, 90, 60))
        d.ellipse([230, 40, 330, 140], fill=(240, 210, 90))
        d.text((20, 12), "vision self-test", fill=(235, 235, 235))
        im.save(img, quality=90)

    print(f"Running local inference on {img.name} via '{config.VISION_MODEL}' ...")
    provider = vision_provider.OllamaProvider()
    try:
        result = provider.analyze_image(img)
    except RuntimeError as e:
        print(f"\nFAILED: {e}")
        return False
    import json
    print("OK - local vision responded:\n" + json.dumps(result, indent=2)[:1200])
    return True


def run(pull_flag: bool = False, test_flag: bool = False) -> None:
    info = doctor()
    if pull_flag:
        print()
        if pull():
            info["model_present"] = True
    if test_flag:
        print()
        test_inference()
