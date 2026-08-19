import "./index.css";
import { Composition } from "remotion";
import { HelloWorld, myCompSchema } from "./HelloWorld";
import { Logo, myCompSchema2 } from "./HelloWorld/Logo";
import { AutoEdit, calculateAutoEditMetadata } from "./AutoEdit/AutoEdit";
import type { EditPlan } from "./AutoEdit/types";

// Self-contained placeholder plan so the composition loads in Studio without a
// generated plan. Real renders override every prop via `--props <plan>.json`.
const placeholderPlan: EditPlan = {
  version: 0,
  profile: "amv",
  meta: { fps: 30, width: 1920, height: 1080, durationInFrames: 90, audio: "track_v1.wav", tempo_bpm: 120 },
  grade: {
    preset: "neonDusk", saturation: 0.9, contrast: 1.08, brightness: 0.98,
    vignette: 0.45, grain: 0.05, letterbox: 90, tint: "#243056", tintOpacity: 0.16,
  },
  clips: [
    {
      id: "c000", clipId: "placeholder", source: "footage.mp4", sourceStart: 10,
      trackStart: 0, duration: 90, speed: 1, transitionIn: { type: "fade", dur: 9 },
      effects: ["kenburns"], level: "mid", isDrop: false, intensity: 0.3, impact: 0.3,
      reason: "placeholder",
    },
  ],
  beatFlashes: [],
  notes: [],
};

// Each <Composition> is an entry in the sidebar!

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Data-driven auto-editor: renders any edit_plan.vN.json (via --props). */}
      <Composition
        id="AutoEdit"
        component={AutoEdit}
        durationInFrames={placeholderPlan.meta.durationInFrames}
        fps={placeholderPlan.meta.fps}
        width={placeholderPlan.meta.width}
        height={placeholderPlan.meta.height}
        defaultProps={placeholderPlan}
        calculateMetadata={calculateAutoEditMetadata}
      />

      <Composition
        // You can take the "id" to render a video:
        // npx remotion render HelloWorld
        id="HelloWorld"
        component={HelloWorld}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
        // You can override these props for each render:
        // https://www.remotion.dev/docs/parametrized-rendering
        schema={myCompSchema}
        defaultProps={{
          titleText: "Welcome to Remotion",
          titleColor: "#000000",
          logoColor1: "#91EAE4",
          logoColor2: "#86A8E7",
        }}
      />

      {/* Mount any React component to make it show up in the sidebar and work on it individually! */}
      <Composition
        id="OnlyLogo"
        component={Logo}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
        schema={myCompSchema2}
        defaultProps={{
          logoColor1: "#91dAE2" as const,
          logoColor2: "#86A8E7" as const,
        }}
      />
    </>
  );
};
