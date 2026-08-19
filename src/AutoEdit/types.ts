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
    tempo_bpm: number;
  };
  grade: PlanGrade;
  clips: PlanClip[];
  beatFlashes: number[];
  audioInfo?: unknown;
  notes?: unknown[];
}
