import type { ReactNode } from "react";
import type { Live } from "../types";
import { Icon } from "./Icon";

export const ISSUE_TEXT: Record<string, string> = {
  blurry: "Blurry: hold the pack still or refocus",
  tooDark: "Too dark",
  tooBright: "Too bright",
  glare: "Glare on the blister: adjust the light",
  analysisFailed: "Analysis failed",
  packNotFound: "Pack not found: all four tray markers must be visible",
  lowDetectorConfidence: "Pack outline unclear",
  packNotConvex: "Pack outline is not a clean rectangle",
  packTooSmall: "Pack too small: move the camera closer",
  packTouchesFrameEdge: "Pack touches the image edge",
  perspectiveTooSteep: "Camera too tilted: look straight down at the pack",
  aspectRatioMismatch: "Pack shape does not match the layout",
};

/** The live camera with the found pack outlined and what stands in the way of a good photo. */
export function LiveView({ live, footer }: { live: Live | null; footer?: ReactNode }) {
  const issues = live ? [...live.quality_issues, ...live.registration_issues] : [];
  const points = live?.quad?.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");

  return (
    <div className="card stack">
      <div className="card-head">
        <Icon name="camera" />
        <h2>Live camera</h2>
        <span className="spacer" />
        {live ? (
          <span className="badge verified">Live · {live.locate_ms} ms</span>
        ) : (
          <span className="badge neutral">Waiting</span>
        )}
      </div>
      <div className="media">
        <img src="/api/camera/stream?width=1280&fps=15" alt="Live camera" />
        <svg viewBox="0 0 100 100" preserveAspectRatio="none">
          {points && (
            <polygon
              points={points}
              fill={live?.usable ? "rgba(127, 112, 255, 0.08)" : "none"}
              vectorEffect="non-scaling-stroke"
              strokeLinejoin="round"
              style={{ stroke: live?.usable ? "var(--accent-strong)" : "var(--review)", strokeWidth: 2.5 }}
            />
          )}
        </svg>
        <div className="hud">
          {live === null && <span>Waiting for the camera…</span>}
          {live?.usable && (
            <span>
              <span className="dot ok" /> Pack locked · ready
            </span>
          )}
          {issues.map((issue) => (
            <span key={issue}>
              <span className="dot" style={{ color: "var(--review)" }} /> {ISSUE_TEXT[issue] ?? issue}
            </span>
          ))}
          {live?.codes?.map((c) => (
            <span key={c}>
              <span className="dot" style={{ color: "var(--count)" }} /> Card {c}
            </span>
          ))}
        </div>
      </div>
      {footer}
    </div>
  );
}

/** Sharpness, exposure and glare of the current frame as three small meters. */
export function CaptureQuality({ live, minimumSharpness = 60 }: { live: Live | null; minimumSharpness?: number }) {
  const issues = new Set(live?.quality_issues ?? []);
  const sharp = live?.sharpness ?? null;
  const sharpShare = sharp === null ? 0 : Math.min(1, sharp / (minimumSharpness * 4));
  const exposure = issues.has("tooDark") ? "Too dark" : issues.has("tooBright") ? "Too bright" : live ? "Good" : "—";
  const glare = issues.has("glare");
  const meter = (share: number, ok: boolean) => (
    <div className="meter">
      <span style={{ width: `${Math.round(share * 100)}%`, background: ok ? "var(--verified)" : "var(--review)" }} />
    </div>
  );
  return (
    <div className="card stack">
      <div className="overline">Capture quality</div>
      <div className="metrics">
        <div className="metric">
          <span className="small muted">Sharpness</span>
          <span className="mono">{sharp === null ? "—" : Math.round(sharp)}</span>
          {meter(sharpShare, sharp !== null && sharp >= minimumSharpness)}
        </div>
        <div className="metric">
          <span className="small muted">Exposure</span>
          <span>{exposure}</span>
          {meter(live ? (exposure === "Good" ? 0.7 : 0.3) : 0, exposure === "Good")}
        </div>
        <div className="metric">
          <span className="small muted">Glare</span>
          <span>{live ? (glare ? "On the pack" : "None") : "—"}</span>
          {meter(live ? (glare ? 0.6 : 0.08) : 0, !glare)}
        </div>
      </div>
    </div>
  );
}
