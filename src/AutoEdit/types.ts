// The edit-plan is the contract between the Python selection/critique pipeline
// and this Remotion renderer. A plan JSON (edits/edit_plan.vN.json) is passed
// verbatim as the composition's input props.

export type TransitionType =
  | "cut"
  | "fade"
  | "swell"
  | "punch"
  | "whip"
  | "slide"
  | "flash"
  | "glitch";

export type EffectType = "kenburns" | "shake" | "flash";

export interface PlanClip {
  id: string;
  clipId: string; // id in the source clip library (for traceability)
  source: string; // staticFile path (relative to public/)
  sourceStart: number; // seconds into the source video
  trackStart: number; // frames on the timeline
  duration: number; // frames on the timeline
  speed: number; // playbackRate (1 = normal, <1 = slow-mo)
  transitionIn: { type: TransitionType; dur: number };
  effects: EffectType[];
  level: "low" | "mid" | "high";
  isDrop: boolean;
  intensity: number;
  impact: number;
  reason: string;
  diegetic?: boolean; // if true, this clip's native movie audio plays (music ducks)
  diegeticVolume?: number; // 0..1 gain for the native audio when diegetic
}

// "underlay" = subtle, low-opacity typographic text integrated into the frame
// (e.g. a lyric fragment behind the action) — distinct from the bold overlays.
export type CaptionStyle = "title" | "line" | "hit" | "end" | "underlay";

export interface Caption {
  text: string; // main line
  sub?: string; // optional under/over line
  start: number; // frame (timeline)
  dur: number; // frames
  style: CaptionStyle;
  pos?: "center" | "lower" | "upper";
}

export interface PlanGrade {
  preset: string;
  saturation: number;
  contrast: number;
  brightness: number;
  vignette: number;
  grain: number;
  letterbox: number;
  tint: string;
  tintOpacity: number;
}

export interface EditPlan {
  // Remotion requires composition props to extend Record<string, unknown>.
  [key: string]: unknown;
  version: number;
  profile: string;
  meta: {
    fps: number;
    width: number;
    height: number;
    durationInFrames: number;
    audio: string;
    audioStart?: number; // seconds into the track where playback begins
    musicVolume?: number; // base music level (0..1); default 1
    fadeInSec?: number; // fade up from black + music in, over this many seconds
    fadeOutSec?: number; // fade down to black + music out, at the end
    tempo_bpm: number;
  };
  grade: PlanGrade;
  clips: PlanClip[];
  beatFlashes: number[];
  captions?: Caption[];
  // Frame windows where the music ducks (down to `to`) so a diegetic clip's
  // native audio can punch through. Ramped in/out by the renderer.
  audioDucks?: { start: number; dur: number; to: number }[];
  audioInfo?: unknown;
  notes?: unknown[];
}
