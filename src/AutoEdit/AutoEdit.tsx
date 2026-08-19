import React from "react";
import {
  AbsoluteFill,
  Audio,
  OffthreadVideo,
  Sequence,
  interpolate,
  random,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  type CalculateMetadataFunction,
} from "remotion";
import type { EditPlan, PlanClip } from "./types";

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const easeOut = (p: number) => 1 - Math.pow(1 - p, 3);

// ---------------------------------------------------------------------------
// A single graded footage slice. Renders OffthreadVideo seeked to sourceStart,
// with an entrance transition + optional ken-burns / shake / flash effects.
// Local frame 0 corresponds to this clip's Sequence `from`.
// ---------------------------------------------------------------------------
const Clip: React.FC<{
  clip: PlanClip;
  grade: EditPlan["grade"];
  lead: number; // frames the sequence starts BEFORE the slot (crossfade overlap)
  totalDur: number; // sequence length in frames (duration + lead)
}> = ({ clip, grade, lead, totalDur }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = clip.transitionIn;

  // seek: at the slot start we want sourceStart to show; the sequence begins
  // `lead` frames earlier, so shift the trim back by `lead`.
  const startFrom = Math.max(0, Math.round(clip.sourceStart * fps) - lead);

  let opacity = 1;
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let blur = 0;

  // entrance -----------------------------------------------------------------
  if (t.type === "fade" || t.type === "swell") {
    // overlap crossfade across the lead region
    const span = lead > 0 ? lead : t.dur;
    opacity = interpolate(frame, [0, Math.max(1, span)], [0, 1], clamp);
    if (t.type === "swell") {
      const e = easeOut(interpolate(frame, [0, Math.max(1, span)], [0, 1], clamp));
      scale *= interpolate(e, [0, 1], [1.18, 1]);
    }
  } else if (t.type === "punch") {
    const e = easeOut(interpolate(frame, [0, t.dur], [0, 1], clamp));
    opacity = interpolate(frame, [0, t.dur * 0.35], [0, 1], clamp);
    scale *= interpolate(e, [0, 1], [1.4, 1]);
    blur += (1 - e) * 8;
  } else if (t.type === "whip") {
    const e = easeOut(interpolate(frame, [0, t.dur], [0, 1], clamp));
    opacity = frame > 1 ? 1 : 0;
    tx += (1 - e) * 1400;
    blur += (1 - e) * 24;
  } else if (t.type === "slide") {
    const e = easeOut(interpolate(frame, [0, t.dur], [0, 1], clamp));
    tx += (1 - e) * -1200;
  } else if (t.type === "glitch") {
    const flick = frame < t.dur ? (random(`g${clip.id}${frame}`) > 0.4 ? 1 : 0.2) : 1;
    opacity = frame > t.dur ? 1 : flick;
    tx += frame < t.dur ? (random(`gx${clip.id}${frame}`) - 0.5) * 120 : 0;
  }
  // "cut" and "flash" have no geometric entrance (flash handled as overlay).

  // ken-burns: slow push across the whole clip -------------------------------
  if (clip.effects.includes("kenburns")) {
    const p = interpolate(frame, [0, totalDur], [0, 1], clamp);
    scale *= interpolate(p, [0, 1], [1.06, 1.16]);
    const dir = random(`kb${clip.id}`) > 0.5 ? 1 : -1;
    tx += dir * interpolate(p, [0, 1], [0, 24]);
    ty += interpolate(p, [0, 1], [0, 10]);
  }
  // shake: high-frequency jitter for impact ----------------------------------
  if (clip.effects.includes("shake")) {
    const amp = 10;
    tx += (random(`sx${clip.id}${Math.floor(frame / 1)}`) - 0.5) * amp;
    ty += (random(`sy${clip.id}${Math.floor(frame / 1)}`) - 0.5) * amp;
    scale *= 1.02;
  }

  const filter = `saturate(${grade.saturation}) contrast(${grade.contrast}) brightness(${grade.brightness})`;

  return (
    <AbsoluteFill style={{ opacity, backgroundColor: "#000" }}>
      <AbsoluteFill
        style={{
          transform: `scale(${scale}) translate(${tx}px, ${ty}px)`,
          filter: blur ? `${filter} blur(${blur}px)` : filter,
        }}
      >
        <OffthreadVideo
          src={staticFile(clip.source)}
          startFrom={startFrom}
          playbackRate={clip.speed}
          muted
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </AbsoluteFill>
      {/* dusk tint wash — recolours toward the profile's mood */}
      <AbsoluteFill
        style={{
          background: grade.tint,
          opacity: grade.tintOpacity,
          mixBlendMode: "soft-light",
          pointerEvents: "none",
        }}
      />
      {/* per-cut flash on drops */}
      {clip.effects.includes("flash") && (
        <AbsoluteFill
          style={{
            background: "#fff",
            opacity: interpolate(frame, [0, 1, 7], [0, 0.85, 0], clamp),
            mixBlendMode: "screen",
            pointerEvents: "none",
          }}
        />
      )}
    </AbsoluteFill>
  );
};

// Persistent grade over everything: grain, vignette, letterbox. -------------
const PostFX: React.FC<{ grade: EditPlan["grade"] }> = ({ grade }) => {
  const frame = useCurrentFrame();
  const { height } = useVideoConfig();
  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      {/* vignette */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,${grade.vignette}) 130%)`,
        }}
      />
      {/* film grain (cheap animated noise) */}
      {grade.grain > 0 && (
        <AbsoluteFill
          style={{
            opacity: grade.grain * 3,
            mixBlendMode: "overlay",
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
            backgroundSize: "300px 300px",
            transform: `translate(${(frame % 5) * 12}px, ${(frame % 3) * 15}px)`,
          }}
        />
      )}
      {/* letterbox bars */}
      {grade.letterbox > 0 && (
        <>
          <AbsoluteFill style={{ height: grade.letterbox, top: 0, background: "#000" }} />
          <AbsoluteFill
            style={{ height: grade.letterbox, top: height - grade.letterbox, background: "#000" }}
          />
        </>
      )}
    </AbsoluteFill>
  );
};

// Beat-flash accents on strong downbeats / drops. --------------------------
const BeatFlash: React.FC = () => {
  const frame = useCurrentFrame();
  const op = interpolate(frame, [0, 1, 6], [0, 0.5, 0], clamp);
  return (
    <AbsoluteFill
      style={{ background: "#fff", opacity: op, mixBlendMode: "screen", pointerEvents: "none" }}
    />
  );
};

// ---------------------------------------------------------------------------
export const AutoEdit: React.FC<EditPlan> = (plan) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {plan.clips.map((clip) => {
        const overlap = clip.transitionIn.type === "fade" || clip.transitionIn.type === "swell";
        const lead = overlap ? Math.min(clip.transitionIn.dur, clip.trackStart) : 0;
        const from = clip.trackStart - lead;
        const totalDur = clip.duration + lead;
        return (
          <Sequence key={clip.id} from={from} durationInFrames={totalDur} layout="none">
            <Clip clip={clip} grade={plan.grade} lead={lead} totalDur={totalDur} />
          </Sequence>
        );
      })}

      {plan.beatFlashes.map((f, i) => (
        <Sequence key={`bf${i}`} from={f} durationInFrames={6} layout="none">
          <BeatFlash />
        </Sequence>
      ))}

      <PostFX grade={plan.grade} />
      <Audio src={staticFile(plan.meta.audio)} />
    </AbsoluteFill>
  );
};

export const calculateAutoEditMetadata: CalculateMetadataFunction<EditPlan> = ({ props }) => ({
  durationInFrames: props.meta.durationInFrames,
  fps: props.meta.fps,
  width: props.meta.width,
  height: props.meta.height,
});
