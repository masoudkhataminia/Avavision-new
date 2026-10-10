import type { ReactNode } from "react";
import type { CheckView } from "../types";
import { Icon } from "./Icon";
import { key } from "./PackGrid";

const OUTLINE: Record<string, string> = {
  needsReview: "var(--review)",
  mismatch: "var(--mismatch)",
};

/** The flattened evidence photo with flagged compartments outlined and every detection drawn on it. */
export function EvidenceView({
  check,
  selected,
  footer,
}: {
  check: CheckView;
  selected: string | null;
  footer?: ReactNode;
}) {
  const layout = check.layout;
  const g = layout.grid_region;
  const cw = g.width / layout.columns;
  const ch = g.height / layout.rows;
  const status = new Map((check.result?.compartments ?? []).map((c) => [key(c), c.status]));
  const frames = check.result?.usable_frames ?? 0;
  return (
    <div className="card stack">
      <div className="card-head">
        <Icon name="scan-eye" />
        <h2>Evidence photo</h2>
        <span className="spacer" />
        <span className={`badge ${frames > 1 ? "verified" : "neutral"}`}>
          {frames} photo{frames === 1 ? "" : "s"} used
        </span>
      </div>
      <div className="media">
        <img src={`/api/check/evidence.jpg?session=${check.session_id}&r=${check.result?.id}`} alt="Evidence" />
        <svg viewBox="0 0 1 1" preserveAspectRatio="none">
          {Array.from({ length: layout.rows }, (_, row) =>
            Array.from({ length: layout.columns }, (_, column) => {
              const k = `${row}:${column}`;
              const isSelected = selected === k;
              const flag = OUTLINE[status.get(k) ?? ""];
              const inset = 0.004;
              return (
                <rect
                  key={k}
                  x={g.x + column * cw + inset}
                  y={g.y + row * ch + inset}
                  width={cw - 2 * inset}
                  height={ch - 2 * inset}
                  rx={0.01}
                  fill={isSelected ? "rgba(127, 112, 255, 0.16)" : "none"}
                  style={{ stroke: isSelected ? "var(--accent-strong)" : (flag ?? "rgba(255, 255, 255, 0.14)") }}
                  strokeDasharray={!isSelected && status.get(k) === "needsReview" ? "6 4" : undefined}
                  strokeWidth={isSelected || flag ? 2.5 : 1}
                  vectorEffect="non-scaling-stroke"
                />
              );
            }),
          )}
          {(check.detections ?? []).map((d, i) => (
            <rect
              key={i}
              x={d.box[0]}
              y={d.box[1]}
              width={d.box[2]}
              height={d.box[3]}
              fill="none"
              style={{ stroke: d.counted ? (d.identity ? "var(--verified)" : "var(--count)") : "var(--mismatch)" }}
              strokeOpacity={0.85}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            >
              <title>
                {`${Math.round(d.confidence * 100)}%${d.identity ? ` · ${d.identity}` : ""}${d.decision ? ` · ${d.decision}` : ""}`}
              </title>
            </rect>
          ))}
        </svg>
      </div>
      {footer}
    </div>
  );
}
