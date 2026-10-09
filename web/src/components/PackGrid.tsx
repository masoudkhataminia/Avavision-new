import type { CompartmentView, Layout, ReviewOutcome } from "../types";

export const key = (c: { row: number; column: number }) => `${c.row}:${c.column}`;

const STATUS_TEXT: Record<string, string> = {
  verified: "Verified",
  countMatched: "Count OK",
  needsReview: "Review",
  mismatch: "Mismatch",
};

const REVIEW_MARK: Record<ReviewOutcome, string> = { confirmedCorrect: "✓", corrected: "✎", unresolved: "✗" };

export function PackGrid({
  layout,
  compartments,
  selected,
  reviews,
  onSelect,
}: {
  layout: Layout;
  compartments: CompartmentView[];
  selected: string | null;
  reviews: Record<string, ReviewOutcome>;
  onSelect: (k: string) => void;
}) {
  const byKey = new Map(compartments.map((c) => [key(c), c]));
  return (
    <div className="grid" style={{ gridTemplateColumns: `5.5rem repeat(${layout.columns}, minmax(0, 1fr))` }}>
      <span />
      {layout.column_labels.map((label) => (
        <span key={label} className="head">
          {label}
        </span>
      ))}
      {layout.row_labels.map((rowLabel, row) => [
        <span key={`r${row}`} className="side">
          {rowLabel}
        </span>,
        ...layout.column_labels.map((_, column) => {
          const c = byKey.get(key({ row, column }));
          if (!c) return <span key={`${row}:${column}`} />;
          const empty = (c.expected_count ?? 0) === 0 && (c.observed_count ?? 0) === 0 && c.status !== "mismatch";
          const review = reviews[key(c)];
          return (
            <div
              key={key(c)}
              className={`cell status-${c.status} ${empty ? "empty" : ""} ${selected === key(c) ? "selected" : ""}`}
              onClick={() => onSelect(key(c))}
              title={[c.label, ...c.findings].join("\n")}
            >
              {review && <span className="mark">{REVIEW_MARK[review]}</span>}
              {!review && c.requires_review && <span className="mark">●</span>}
              <span className="count">
                {c.observed_count ?? "?"}/{c.expected_count ?? "?"}
              </span>
              <span>{STATUS_TEXT[c.status]}</span>
              {c.spot_check && <span className="small">spot check</span>}
              {c.advisory && c.advisory.verdict === "disagrees" && <span className="small">2nd opinion ✗</span>}
            </div>
          );
        }),
      ])}
    </div>
  );
}
