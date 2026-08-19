"""
Orchestrator CLI — the single entry point for the whole pipeline.

    python -m pipeline.run analyze [sources...]     # footage + audio -> library
    python -m pipeline.run plan   [--profile amv]   # library + music -> edit plan v1
    python -m pipeline.run render <plan.json>       # render one plan to mp4
    python -m pipeline.run iterate [--profile amv] [--iters 4]
                                                    # full AI critique/revise loop
    python -m pipeline.run all    [--profile amv]   # analyze + iterate end-to-end

The critique/revise loop:
  footage -> analysis -> clip library -> music analysis -> initial edit ->
  (render) -> critique -> revise -> ... until improvement is negligible.

With the local critic (no API key) the critique reasons over the structured
plan + analysis, so intermediate renders aren't needed to score — only v1 and
the converged final are rendered (fast). With a vision key, every iteration is
rendered and the rendered frames are critiqued by the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config, audio as audio_mod, ingest, visual, library
from . import select as select_mod, render as render_mod, revise as revise_mod
from . import vision_provider, setup_vision, ffmpeg_util
from .io_util import load_json, save_json


# ---------------------------------------------------------------------------
def cmd_analyze(args) -> None:
    sources = args.sources or None
    print("[1/4] audio analysis...")
    a = audio_mod.analyze(args.track, force=args.force)
    print(f"      tempo {a['tempo_bpm']} BPM, {len(a['beats'])} beats, "
          f"drops {a['drops']}, {len(a['sections'])} sections")
    print("[2/4] footage: shots + adaptive keyframes...")
    s = ingest.analyze(sources, force=args.force)
    print(f"      {s['n_sources']} source(s), {s['n_shots']} shots")
    print("[3/4] embeddings + dedup...")
    e = visual.compute(force=args.force)
    print(f"      {e['n_shots']} shots -> {e['n_clusters']} visual clusters")
    print("[4/4] clip library + semantic tags...")
    lib = library.build(force=args.force)
    provider = vision_provider.get_provider(args.provider)
    tags = provider.tag_clips(lib)
    for c in lib["clips"]:
        c["tags"] = tags.get(c["id"], {})
    save_json(config.ANALYSIS / "clips.json", lib)
    print(f"      {lib['n_usable']}/{lib['n_clips']} usable clips, "
          f"tagged by '{provider.name}' provider")


def cmd_plan(args) -> dict:
    plan = select_mod.build_plan(args.profile, version=1)
    print(f"plan v1: {len(plan['clips'])} cuts, "
          f"{plan['meta']['durationInFrames'] / plan['meta']['fps']:.1f}s, profile={args.profile}")
    return plan


def cmd_render(args) -> None:
    out = render_mod.render(Path(args.plan))
    print("rendered ->", out)


# ---------------------------------------------------------------------------
def cmd_iterate(args) -> None:
    audio = load_json(config.ANALYSIS / "audio.json")
    clips_doc = load_json(config.ANALYSIS / "clips.json")
    if not audio or not clips_doc:
        raise SystemExit("run `analyze` first")

    provider = vision_provider.get_provider(args.provider)
    render_each = provider.semantic   # any pixel-level provider judges real frames
    print(f"critic: {provider.name}  (render every iteration: {render_each})")

    plan = select_mod.build_plan(args.profile, version=1)
    print(f"v1: {len(plan['clips'])} cuts")

    # always render v1 so the initial edit is inspectable
    draft = render_mod.render(config.EDITS / "edit_plan.v1.json")
    print(f"v1 draft -> {draft}")

    history = []
    prev_score = None
    for it in range(1, args.iters + 1):
        if render_each and it > 1:
            draft = render_mod.render(config.EDITS / f"edit_plan.v{plan['version']}.json")
        crit = provider.critique(draft, plan, audio, clips_doc)
        save_json(config.EDITS / f"critique.v{plan['version']}.json", crit)
        print(f"  v{plan['version']}: {crit['summary']}")
        history.append({"version": plan["version"], "score": crit["score"]})

        if crit["score"] >= 96:
            print("  -> excellent; stopping.")
            break
        if prev_score is not None and (crit["score"] - prev_score) < 2 and it > 1:
            print("  -> improvement negligible; stopping.")
            break
        prev_score = crit["score"]

        new_plan = revise_mod.revise(plan, crit, audio, clips_doc)
        if new_plan["notes"][-1]["applied"] == []:
            print("  -> no actionable changes; stopping.")
            break
        plan = new_plan

    # persist history BEFORE the (slow) final render so it survives interruption
    save_json(config.EDITS / "history.json", {"history": history, "final_version": plan["version"]})
    print("score history:", " -> ".join(f"v{h['version']}:{h['score']}" for h in history))

    # render the final converged plan
    final_plan_path = config.EDITS / f"edit_plan.v{plan['version']}.json"
    final = render_mod.render(final_plan_path, out_path=config.OUT / "final.mp4")
    print(f"\nFINAL edit: v{plan['version']}  ->  {final}")


def cmd_all(args) -> None:
    cmd_analyze(args)
    cmd_iterate(args)


# ---------------------------------------------------------------------------
def cmd_doctor(args) -> None:
    setup_vision.doctor()


def cmd_setup(args) -> None:
    setup_vision.run(pull_flag=args.pull, test_flag=args.test)


_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def cmd_vision(args) -> None:
    """General-purpose local vision: analyze ANY image or video into the schema.

    Not AMV-specific. Images -> one structured record. Videos -> coarse-sampled
    per-timestamp records (multi-stage: sample -> analyze -> structured report).
    """
    import json
    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"not found: {src}")

    provider = vision_provider.get_provider(args.provider)
    if not getattr(provider, "semantic", False):
        raise SystemExit(
            f"provider '{provider.name}' has no semantic vision. "
            "Run `python -m pipeline.run doctor`, then use --provider ollama.")

    out_dir = config.ANALYSIS / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() in _IMG_EXT:
        rec = provider.analyze_image(src)
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        dest = out_dir / f"{src.stem}.json"
        save_json(dest, rec)
        print(f"\nsaved -> {dest}")
        return

    if src.suffix.lower() in _VID_EXT:
        meta = ffmpeg_util.probe_media(src)
        dur = meta.get("duration") or 0.0
        every = max(0.5, float(args.every))
        times = [round(t, 2) for t in _frange(0.0, dur, every)] or [0.0]
        if len(times) > args.max_frames:                 # bound local compute
            step = len(times) / args.max_frames
            times = [times[int(i * step)] for i in range(args.max_frames)]
        print(f"video {src.name}: {dur:.1f}s, analyzing {len(times)} frames "
              f"(every ~{every}s) with '{provider.name}'...")

        tmp = config.CACHE / "vision_frames"
        tmp.mkdir(exist_ok=True)
        records = []
        for i, t in enumerate(times):
            frame = tmp / f"v{i:03d}.jpg"
            import subprocess
            subprocess.run([
                ffmpeg_util.ffmpeg_exe(), "-hide_banner", "-nostdin",
                "-ss", f"{t:.3f}", "-i", str(src),
                "-frames:v", "1", "-vf", "scale=512:-2", "-q:v", "4", "-y", str(frame),
            ], capture_output=True)
            if not frame.exists():
                continue
            rec = provider.analyze_image(frame)
            rec["timestamp"] = t
            records.append(rec)
            print(f"  t={t:>6.2f}s  {str(rec.get('scene',''))[:70]}")

        report = {"source": str(src), "duration": dur, "n_frames": len(records),
                  "model": config.VISION_MODEL, "frames": records}
        dest = out_dir / f"{src.stem}.vision.json"
        save_json(dest, report)
        print(f"\nsaved -> {dest}")
        return

    raise SystemExit(f"unsupported file type: {src.suffix}")


def _frange(start: float, stop: float, step: float):
    t = start
    while t < stop:
        yield t
        t += step


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(prog="pipeline.run", description="Autonomous visual-model video editor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", default="amv", help="edit profile: amv | hype | montage")
    common.add_argument("--provider", default=None,
                        help="vision provider: auto | ollama | anthropic | local")
    common.add_argument("--force", action="store_true", help="ignore analysis caches")

    a = sub.add_parser("analyze", parents=[common], help="footage + audio -> clip library")
    a.add_argument("sources", nargs="*", help="footage file(s)/folder (default public/footage.mp4)")
    a.add_argument("--track", default=None, help="music track (default public/track_v1.wav)")
    a.set_defaults(func=cmd_analyze)

    p = sub.add_parser("plan", parents=[common], help="build edit plan v1")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("render", help="render one plan JSON to mp4")
    r.add_argument("plan")
    r.set_defaults(func=cmd_render)

    it = sub.add_parser("iterate", parents=[common], help="critique/revise loop")
    it.add_argument("--iters", type=int, default=4)
    it.set_defaults(func=cmd_iterate)

    al = sub.add_parser("all", parents=[common], help="analyze + iterate")
    al.add_argument("sources", nargs="*")
    al.add_argument("--track", default=None)
    al.add_argument("--iters", type=int, default=4)
    al.set_defaults(func=cmd_all)

    doc = sub.add_parser("doctor", help="inspect hardware + local-vision prereqs")
    doc.set_defaults(func=cmd_doctor)

    st = sub.add_parser("setup", help="doctor + optional model pull / inference test")
    st.add_argument("--pull", action="store_true", help="download the recommended/VISION_MODEL")
    st.add_argument("--test", action="store_true", help="run one real local inference")
    st.set_defaults(func=cmd_setup)

    vis = sub.add_parser("vision", parents=[common],
                         help="general-purpose local analysis of ANY image or video")
    vis.add_argument("source", help="path to an image or video file")
    vis.add_argument("--every", type=float, default=2.0,
                     help="video: seconds between sampled frames (default 2.0)")
    vis.add_argument("--max-frames", dest="max_frames", type=int, default=40,
                     help="video: cap analyzed frames (default 40)")
    vis.set_defaults(func=cmd_vision)

    args = ap.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        # Local-inference / provider errors are expected + actionable: show the
        # clean message (with install/fix steps) instead of a Python traceback.
        raise SystemExit(f"\n{e}")


if __name__ == "__main__":
    main()
