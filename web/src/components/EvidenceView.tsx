import type { CheckView } from "../types";

/** The flattened evidence photo with the compartment grid and every detection drawn on it. */
export function EvidenceView({ check, selected }: { check: CheckView; selected: string | null }) {
  const layout = check.layout;
  const g = layout.grid_region;
  const cw = g.width / layout.columns;
  const ch = g.height / layout.rows;
  return (
    <div className="media">
      <img src={`/api/check/evidence.jpg?session=${check.session_id}&r=${check.result?.id}`} alt="Evidence" />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none">
        {Array.from({ length: layout.rows }, (_, row) =>
          Array.from({ length: layout.columns }, (_, column) => {
            const isSelected = selected === `${row}:${column}`;
            return (
              <rect
                key={`${row}:${column}`}
                x={g.x + column * cw}
                y={g.y + row * ch}
                width={cw}
                height={ch}
                fill={isSelected ? "rgba(255,255,255,0.18)" : "none"}
                stroke={isSelected ? "#ffffff" : "rgba(255,200,0,0.45)"}
                strokeWidth={isSelected ? 3 : 1}
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
            stroke={d.counted ? (d.identity ? "#2ecc71" : "#4aa3ff") : "#ff6b4a"}
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          >
            <title>
              {`${Math.round(d.confidence * 100)}%${d.identity ? ` · ${d.identity}` : ""}${d.decision ? ` · ${d.decision}` : ""}`}
            </title>
          </rect>
        ))}
      </svg>
    </div>
  );
}
