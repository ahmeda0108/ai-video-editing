import React from "react";
import {
  AbsoluteFill,
  Audio,
  OffthreadVideo,
  Sequence,
  interpolate,
  random,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  type CalculateMetadataFunction,
} from "remotion";
import { loadFont as loadAnton } from "@remotion/google-fonts/Anton";
import { loadFont as loadOswald } from "@remotion/google-fonts/Oswald";
import type { Caption, EditPlan, PlanClip } from "./types";

const anton = loadAnton().fontFamily;
const oswald = loadOswald().fontFamily;

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
          muted={!clip.diegetic}
          volume={clip.diegetic ? clip.diegeticVolume ?? 1 : undefined}
          toneMapped={false}
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

// Kinetic caption/text overlay (title / lower-third line / drop hit / end card).
// ---------------------------------------------------------------------------
// Premium KINETIC underlay: not a flat caption. Each letter springs up and
// un-blurs on a stagger; a chromatic RGB-split flares wide on entry then
// resolves to a hairline; a blurred blue echo gives depth; an accent rule
// draws out from center; the whole thing drifts + breathes so it lives in the
// shot. The white core is legible (normal blend + soft shadow); the coloured
// fringes are screen-blended so they glow INTO the footage.
// ---------------------------------------------------------------------------
const KineticUnderlay: React.FC<{ cap: Caption; scale: number }> = ({ cap, scale }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const chars = React.useMemo(() => [...cap.text], [cap.text]);

  const gIn = interpolate(frame, [0, 20], [0, 1], clamp);
  const gOut = interpolate(frame, [cap.dur - 16, cap.dur], [1, 0], clamp);
  const alive = gIn * gOut;

  const t01 = interpolate(frame, [0, cap.dur], [0, 1], clamp);
  const driftX = interpolate(t01, [0, 1], [-18, 18]) * scale;
  const driftY = Math.sin(frame / 26) * 4 * scale;
  const breathe = 1 + Math.sin(frame / 34) * 0.014;
  const split = (interpolate(gIn, [0, 1], [18, 2], clamp) + (1 - gOut) * 12) * scale;
  const size = 150 * scale;

  const base: React.CSSProperties = {
    fontFamily: anton, fontSize: size, lineHeight: 1.05,
    letterSpacing: 12 * scale, textTransform: "uppercase", whiteSpace: "nowrap",
    display: "flex", justifyContent: "center",
  };

  const letters = (color: string, shadow?: string) => (
    <div style={{ ...base, color, textShadow: shadow }}>
      {chars.map((ch, i) => {
        const s = spring({ frame: frame - i * 2.4, fps, config: { damping: 13, stiffness: 130, mass: 0.8 } });
        return (
          <span key={i} style={{
            display: "inline-block", opacity: s,
            transform: `translateY(${interpolate(s, [0, 1], [32 * scale, 0])}px)`,
            filter: `blur(${interpolate(s, [0, 1], [16, 0])}px)`,
          }}>{ch === " " ? " " : ch}</span>
        );
      })}
    </div>
  );

  const layer = (node: React.ReactNode, dx: number, opacity: number,
                 blend?: React.CSSProperties["mixBlendMode"]) => (
    <div style={{ position: "absolute", left: 0, right: 0, top: 0,
      transform: `translateX(${dx}px)`, opacity, mixBlendMode: blend }}>{node}</div>
  );

  return (
    <div style={{
      position: "absolute", left: "50%", top: "50%",
      transform: `translate(-50%,-50%) translate(${driftX}px,${driftY}px) scale(${breathe})`,
      opacity: alive, pointerEvents: "none",
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>
      <div style={{ position: "relative", display: "inline-block" }}>
        {/* transparent spacer sizes the relative box */}
        <div style={{ ...base, color: "transparent" }}>
          {chars.map((c, i) => <span key={i} style={{ display: "inline-block" }}>{c === " " ? " " : c}</span>)}
        </div>
        {layer(<div style={{ ...base, color: "#8fb4ff", filter: "blur(24px)", transform: "scale(1.08)" }}>{cap.text}</div>, 0, 0.16)}
        {layer(letters("#ff2a42"), -split, 0.55, "screen")}
        {layer(letters("#22d3ff"), split, 0.55, "screen")}
        {layer(letters("#ffffff", "0 4px 26px rgba(0,0,0,0.55), 0 0 50px rgba(120,160,255,0.35)"), 0, 0.92)}
        <div style={{
          position: "absolute", left: "50%", bottom: -20 * scale, height: 3 * scale,
          width: `${interpolate(gIn, [0, 1], [0, 100], clamp) * gOut}%`,
          transform: "translateX(-50%)", opacity: 0.7,
          background: "linear-gradient(90deg, transparent, #cfe0ff, transparent)",
        }} />
      </div>
      {cap.sub && (
        <div style={{
          fontFamily: oswald, fontWeight: 600, fontSize: 26 * scale, letterSpacing: 14 * scale,
          textTransform: "uppercase", color: "#dbe6ff", marginTop: 22 * scale,
          opacity: interpolate(frame, [12, 28], [0, 0.85], clamp) * gOut,
          clipPath: `inset(0 ${interpolate(gIn, [0, 1], [100, 0], clamp)}% 0 0)`,
        }}>{cap.sub}</div>
      )}
    </div>
  );
};

const CaptionView: React.FC<{ cap: Caption }> = ({ cap }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const scale = Math.min(width, height * 1.6) / 1080; // scale text to frame size

  if (cap.style === "underlay") {
    return (
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <KineticUnderlay cap={cap} scale={scale} />
      </AbsoluteFill>
    );
  }
  const p = interpolate(frame, [0, 12], [0, 1], clamp); // entrance 0..1
  const e = easeOut(p);
  const out = interpolate(frame, [cap.dur - 10, cap.dur], [1, 0], clamp);

  const posY = cap.pos === "upper" ? "22%" : cap.pos === "lower" ? "76%" : "50%";
  const shadow = "0 4px 24px rgba(0,0,0,0.9), 0 0 60px rgba(0,0,0,0.6)";

  let mainSize = 92;
  let tx = 0;
  let sc = 1;
  let op = e * out;
  let blur = 0;
  let bar = false;

  if (cap.style === "title" || cap.style === "end") {
    mainSize = 150;
    sc = interpolate(e, [0, 1], [cap.style === "title" ? 1.28 : 1.08, 1]);
    blur = (1 - e) * 10;
  } else if (cap.style === "line") {
    mainSize = 66;
    tx = 0;
    op = e * out;
    bar = true;
  } else if (cap.style === "hit") {
    mainSize = 168;
    sc = interpolate(e, [0, 1], [1.7, 1]);
    // quick shake on entrance
    if (frame < 8) {
      tx += (random(`hit${cap.text}${frame}`) - 0.5) * 22;
    }
    blur = (1 - e) * 6;
  } else if (cap.style === "underlay") {
    // faint typographic wash blended INTO the footage — a lyric fragment, not
    // a hard caption. Slow drift + very low opacity so it never competes.
    mainSize = 132;
    sc = interpolate(e, [0, 1], [1.06, 1]);
    op = e * out; // final alpha is applied via the text color below
  }

  const underlay = cap.style === "underlay";
  const lower = cap.style === "line" || cap.pos === "lower";
  return (
    <AbsoluteFill
      style={{
        justifyContent: lower ? "flex-end" : "center",
        alignItems: "center",
        paddingBottom: lower ? height * 0.14 : 0,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: cap.style === "line" ? undefined : posY,
          transform: `translateY(${cap.style === "line" ? 0 : "-50%"}) translateX(${tx}px) scale(${sc})`,
          opacity: underlay ? op * 0.32 : op,
          mixBlendMode: underlay ? "overlay" : undefined,
          filter: blur ? `blur(${blur}px)` : undefined,
          textAlign: "center",
          padding: bar ? "10px 30px" : 0,
          background: bar
            ? "linear-gradient(90deg, transparent, rgba(0,0,0,0.55) 20%, rgba(0,0,0,0.55) 80%, transparent)"
            : undefined,
        }}
      >
        <div
          style={{
            fontFamily: anton,
            fontSize: mainSize * scale,
            lineHeight: 1.02,
            color: "#fff",
            letterSpacing: cap.style === "line" ? 2 : underlay ? 16 : 6,
            textShadow: underlay
              ? "0 2px 40px rgba(0,0,0,0.4)"
              : cap.style === "hit"
                ? `${shadow}, 0 0 40px rgba(255,60,60,0.7)`
                : shadow,
            textTransform: "uppercase",
          }}
        >
          {cap.text}
        </div>
        {cap.sub && (
          <div
            style={{
              fontFamily: oswald,
              fontWeight: 600,
              fontSize: mainSize * 0.24 * scale,
              color: "#fff",
              opacity: 0.9,
              letterSpacing: 10,
              textTransform: "uppercase",
              marginTop: 12,
              textShadow: shadow,
            }}
          >
            {cap.sub}
          </div>
        )}
      </div>
    </AbsoluteFill>
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

      {(plan.captions ?? []).map((cap, i) => (
        <Sequence key={`cap${i}`} from={cap.start} durationInFrames={cap.dur} layout="none">
          <CaptionView cap={cap} />
        </Sequence>
      ))}

      {(() => {
        const total = plan.meta.durationInFrames;
        const fpsN = plan.meta.fps;
        const fi = Math.round((plan.meta.fadeInSec ?? 0) * fpsN);
        const fo = Math.round((plan.meta.fadeOutSec ?? 0) * fpsN);
        if (fi <= 0 && fo <= 0) return null;
        return (
          <Sequence from={0} durationInFrames={total} layout="none">
            <FadeToBlack total={total} fadeIn={fi} fadeOut={fo} />
          </Sequence>
        );
      })()}

      <Audio
        src={staticFile(plan.meta.audio)}
        startFrom={Math.round((plan.meta.audioStart ?? 0) * plan.meta.fps)}
        volume={(f) =>
          musicVolumeAt(f, plan.meta.musicVolume ?? 1, plan.audioDucks ?? []) *
          fadeEnvelope(
            f,
            plan.meta.durationInFrames,
            Math.round((plan.meta.fadeInSec ?? 0) * plan.meta.fps),
            Math.round((plan.meta.fadeOutSec ?? 0) * plan.meta.fps)
          )
        }
      />
    </AbsoluteFill>
  );
};

// 0 at the very start/end (black + silent), 1 in the body. Drives both the
// black overlay and the music fade so audio and picture fade together.
function fadeEnvelope(frame: number, total: number, fadeIn: number, fadeOut: number): number {
  let v = 1;
  if (fadeIn > 0) v = Math.min(v, interpolate(frame, [0, fadeIn], [0, 1], clamp));
  if (fadeOut > 0) v = Math.min(v, interpolate(frame, [total - fadeOut, total], [1, 0], clamp));
  return Math.max(0, v);
}

const FadeToBlack: React.FC<{ total: number; fadeIn: number; fadeOut: number }> = ({
  total,
  fadeIn,
  fadeOut,
}) => {
  const frame = useCurrentFrame();
  const opacity = 1 - fadeEnvelope(frame, total, fadeIn, fadeOut);
  if (opacity <= 0.001) return null;
  return <AbsoluteFill style={{ backgroundColor: "#000", opacity }} />;
};

// Music level at a given composition frame: sits at `base`, but dips to each
// duck's `to` (ramped over DUCK_RAMP frames) so a diegetic clip's native audio
// punches through cleanly, then swells back.
const DUCK_RAMP = 9;
function musicVolumeAt(
  frame: number,
  base: number,
  ducks: { start: number; dur: number; to: number }[]
): number {
  let v = base;
  for (const d of ducks) {
    const seg = interpolate(
      frame,
      [d.start - DUCK_RAMP, d.start, d.start + d.dur, d.start + d.dur + DUCK_RAMP],
      [base, d.to, d.to, base],
      clamp
    );
    if (seg < v) v = seg;
  }
  return Math.max(0, v);
}

export const calculateAutoEditMetadata: CalculateMetadataFunction<EditPlan> = ({ props }) => ({
  durationInFrames: props.meta.durationInFrames,
  fps: props.meta.fps,
  width: props.meta.width,
  height: props.meta.height,
});
