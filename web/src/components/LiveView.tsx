import { useEffect } from "react";
import { useData } from "../hooks";
import type { Live } from "../types";

const ISSUE_TEXT: Record<string, string> = {
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

export function LiveView({ onCodes }: { onCodes?: (codes: string[]) => void } = {}) {
  const { data: live } = useData<Live>("/api/live", 200);
  const codes = live?.codes?.join("|") ?? "";
  useEffect(() => {
    if (onCodes) onCodes(codes ? codes.split("|") : []);
  }, [codes, onCodes]);
  const issues = live ? [...live.quality_issues, ...live.registration_issues] : [];
  const points = live?.quad?.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");

  return (
    <div className="stack">
      <div className="media">
        <img src="/api/camera/stream?width=1280&fps=15" alt="Live camera" />
        <svg viewBox="0 0 100 100" preserveAspectRatio="none">
          {points && (
            <polygon
              points={points}
              fill="none"
              stroke={live?.usable ? "#2ecc71" : "#f5a623"}
              strokeWidth={0.4}
              vectorEffect="non-scaling-stroke"
              style={{ strokeWidth: 3 }}
            />
          )}
        </svg>
      </div>
      <div className="row small">
        {live === null && <span className="muted">Waiting for the camera…</span>}
        {live && live.usable && <span className="banner status-verified small">Pack found · ready to capture</span>}
        {live &&
          issues.map((issue) => (
            <span key={issue} className="banner status-needsReview small">
              {ISSUE_TEXT[issue] ?? issue}
            </span>
          ))}
        {live?.codes?.map((c) => (
          <span key={c} className="banner status-countMatched small">
            Card {c}
          </span>
        ))}
        {live && <span className="muted">{live.locate_ms} ms</span>}
      </div>
    </div>
  );
}
