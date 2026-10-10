import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Appearance, Catalog, CompartmentView, Layout, Profile, ReviewOutcome } from "../types";
import { Icon, type IconName } from "./Icon";
import { describeAppearance, pillColour, Tablet } from "./Tablet";

export const key = (c: { row: number; column: number }) => `${c.row}:${c.column}`;

export const STATUS_TEXT: Record<string, string> = {
  verified: "Verified",
  countMatched: "Count OK",
  needsReview: "Review",
  mismatch: "Mismatch",
  retakeRequired: "Retake",
};

const REVIEW_ICON: Record<ReviewOutcome, IconName> = {
  confirmedCorrect: "check",
  corrected: "pencil",
  unresolved: "x",
};

/** What one compartment should hold, with each medication's catalog appearance. */
export type Planned = { id: string; name: string; quantity: number; appearance?: Appearance };

export function compartmentLabel(layout: Layout, c: { row: number; column: number }): string {
  const column = layout.column_labels[c.column] ?? `C${c.column + 1}`;
  const row = layout.row_labels[c.row] ?? `R${c.row + 1}`;
  return `${column} · ${row}`;
}

function medicationName(catalog: Catalog | null, id: string): { name: string; appearance?: Appearance } {
  const m = catalog?.medications.find((x) => x.id === id);
  return m ? { name: `${m.name} ${m.strength}`.trim(), appearance: m.appearance } : { name: id };
}

/** Planned contents per compartment: from the analysed result when there is one, otherwise from the profile. */
export function planFor(
  catalog: Catalog | null,
  profile: Profile | null,
  compartments?: CompartmentView[],
): Map<string, Planned[]> {
  const plan = new Map<string, Planned[]>();
  if (compartments?.length) {
    for (const c of compartments) {
      plan.set(
        key(c),
        c.expected.map((e) => ({
          id: e.id,
          name: e.name,
          quantity: e.quantity,
          appearance: medicationName(catalog, e.id).appearance,
        })),
      );
    }
    return plan;
  }
  for (const e of profile?.compartments ?? []) {
    plan.set(
      key(e.compartment),
      e.items
        .filter((i) => i.quantity > 0)
        .map((i) => ({ id: i.medication_id, quantity: i.quantity, ...medicationName(catalog, i.medication_id) })),
    );
  }
  return plan;
}

const signature = (items: Planned[] | undefined) =>
  (items ?? [])
    .map((i) => `${i.id}×${i.quantity}`)
    .sort()
    .join("|");

const total = (items: Planned[] | undefined) => (items ?? []).reduce((sum, i) => sum + i.quantity, 0);

function colours(items: Planned[] | undefined): string[] {
  return [...new Set((items ?? []).map((i) => pillColour(i.appearance?.colour)))].slice(0, 4);
}

function MedDots({ items }: { items: Planned[] | undefined }) {
  return (
    <span className="med-dots">
      {colours(items).map((colour) => (
        <i key={colour} style={{ background: colour }} />
      ))}
    </span>
  );
}

type Hover = { k: string; rect: DOMRect };

/** The pack as it lies on the tray: a 3D blister tile per compartment, its contents on hover. */
export function PackGrid({
  layout,
  plan,
  compartments,
  selected,
  reviews = {},
  onSelect,
}: {
  layout: Layout;
  plan: Map<string, Planned[]>;
  compartments?: CompartmentView[];
  selected?: string | null;
  reviews?: Record<string, ReviewOutcome>;
  onSelect?: (k: string) => void;
}) {
  const byKey = new Map((compartments ?? []).map((c) => [key(c), c]));
  const [hover, setHover] = useState<Hover | null>(null);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const enter = (k: string, element: HTMLElement) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setHover({ k, rect: element.getBoundingClientRect() }), 220);
  };
  const leave = () => {
    window.clearTimeout(timer.current);
    setHover(null);
  };

  const longest = Math.max(0, ...layout.row_labels.map((l) => l.length));
  const side = longest > 4 ? Math.min(84, 14 + longest * 7) : 44;
  return (
    <div
      className="pack-grid"
      style={{ "--columns": layout.columns, "--side": `${side}px` } as CSSProperties}
      onMouseLeave={leave}
    >
      <div className="pack-row">
        <span />
        {layout.column_labels.map((label) => (
          <span key={label} className="head overline">
            {label}
          </span>
        ))}
      </div>
      {layout.row_labels.map((rowLabel, row) => (
        <div key={rowLabel} className="pack-row band">
          <span className="side overline">{rowLabel}</span>
          {layout.column_labels.map((_, column) => {
            const k = key({ row, column });
            const items = plan.get(k);
            const c = byKey.get(k);
            const expected = c ? c.expected_count : total(items);
            const empty = !expected && !c?.observed_count && c?.status !== "mismatch";
            const review = reviews[k];
            const status = c && !empty ? `s-${c.status}` : "";
            return (
              <button
                key={k}
                type="button"
                className={`cell ${status} ${empty ? "empty" : ""} ${selected === k ? "selected" : ""}`}
                onClick={() => onSelect?.(k)}
                onMouseEnter={(e) => enter(k, e.currentTarget)}
                onFocus={(e) => enter(k, e.currentTarget)}
                onBlur={leave}
                aria-label={compartmentLabel(layout, { row, column })}
              >
                <span className="top">
                  <span className="count">
                    {empty ? "—" : c ? `${c.observed_count ?? "–"}/${c.expected_count ?? "–"}` : expected}
                  </span>
                  <MedDots items={items} />
                </span>
                <span className="bottom">
                  <span className="dot" />
                  <span className="state">{empty ? "Empty" : c ? STATUS_TEXT[c.status] : "Expected"}</span>
                  <MedDots items={items} />
                </span>
                {review && (
                  <span className="mark">
                    <Icon name={REVIEW_ICON[review]} size={14} />
                  </span>
                )}
                {!review && c?.advisory?.verdict === "disagrees" && (
                  <span className="mark status-mismatch">
                    <Icon name="sparkles" size={14} />
                  </span>
                )}
                {!review && c?.spot_check && c.advisory?.verdict !== "disagrees" && (
                  <span className="mark status-countMatched">
                    <Icon name="eye" size={14} />
                  </span>
                )}
              </button>
            );
          })}
        </div>
      ))}
      {hover && (
        <Popover
          hover={hover}
          label={compartmentLabel(layout, {
            row: Number(hover.k.split(":")[0]),
            column: Number(hover.k.split(":")[1]),
          })}
          items={plan.get(hover.k) ?? []}
          compartment={byKey.get(hover.k)}
          peers={
            [...plan.entries()].filter(
              ([k, v]) => k !== hover.k && v.length && signature(v) === signature(plan.get(hover.k)),
            ).length
          }
        />
      )}
    </div>
  );
}

function Popover({
  hover,
  label,
  items,
  compartment,
  peers,
}: {
  hover: Hover;
  label: string;
  items: Planned[];
  compartment?: CompartmentView;
  peers: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);
  useLayoutEffect(() => {
    const height = ref.current?.offsetHeight ?? 0;
    const width = ref.current?.offsetWidth ?? 300;
    const { rect } = hover;
    const left = Math.min(Math.max(12, rect.left + rect.width / 2 - width / 2), window.innerWidth - width - 12);
    const below = rect.bottom + 8;
    const top = below + height > window.innerHeight - 12 ? Math.max(12, rect.top - height - 8) : below;
    setPosition({ left, top });
  }, [hover]);

  const expected = items.reduce((sum, i) => sum + i.quantity, 0);
  const seen = compartment?.observed_count;
  const difference = seen !== null && seen !== undefined ? seen - expected : 0;
  const finding = compartment?.findings[0];
  return createPortal(
    <div ref={ref} className="popover" style={position ?? { visibility: "hidden" }}>
      <div className="card-head">
        <strong>{label}</strong>
        <span className="spacer" />
        <span className="small faint">
          {compartment
            ? `${compartment.expected_count ?? "?"} expected · ${seen ?? "?"} seen`
            : `${expected} tablet${expected === 1 ? "" : "s"}`}
        </span>
      </div>
      {items.length === 0 && <div className="small muted">Nothing belongs in this compartment.</div>}
      {items.map((item) => (
        <div key={item.id} className="popover-row">
          <Tablet appearance={item.appearance} />
          <span className="text">
            <span className="ellipsis">{item.name}</span>
            <span className="small faint ellipsis">
              {describeAppearance(item.appearance) || "appearance not recorded"}
            </span>
          </span>
          <span className="mono muted">×{item.quantity}</span>
        </div>
      ))}
      {compartment && difference !== 0 && (
        <div className="popover-row extra">
          <span className="tablet-slot status-mismatch">
            <Icon name="circle-alert" size={18} />
          </span>
          <span className="text">
            <span>{difference > 0 ? `${difference} more than expected` : `${-difference} fewer than expected`}</span>
            <span className="small faint">{difference > 0 ? "Not expected here" : "Possibly missing"}</span>
          </span>
          <span className="mono status-mismatch">
            {difference > 0 ? "+" : ""}
            {difference}
          </span>
        </div>
      )}
      {finding ? (
        <div className="small muted">{finding}</div>
      ) : (
        peers > 0 && (
          <div className="small muted">
            Same tablets as {peers} other compartment{peers === 1 ? "" : "s"}
          </div>
        )
      )}
    </div>,
    document.body,
  );
}

/** Legend under the grid: how many compartments ended in each state, or how much the pack holds. */
export function PackLegend({ plan, compartments }: { plan: Map<string, Planned[]>; compartments?: CompartmentView[] }) {
  if (compartments?.length) {
    const counts = new Map<string, number>();
    for (const c of compartments)
      if ((c.expected_count ?? 0) > 0 || c.observed_count || c.status === "mismatch")
        counts.set(c.status, (counts.get(c.status) ?? 0) + 1);
    return (
      <div className="row">
        {["mismatch", "needsReview", "countMatched", "verified"]
          .filter((s) => counts.get(s))
          .map((s) => (
            <span key={s} className={`badge ${s}`}>
              {counts.get(s)} {STATUS_TEXT[s].toLowerCase()}
            </span>
          ))}
      </div>
    );
  }
  const filled = [...plan.values()].filter((items) => items.length).length;
  const tablets = [...plan.values()].reduce((sum, items) => sum + total(items), 0);
  return (
    <div className="small muted">
      {filled} doses · {tablets} tablets in this pack
    </div>
  );
}

/** Every medication in the pack with its appearance and how many doses contain it. */
export function PackMedications({ plan, limit = 6 }: { plan: Map<string, Planned[]>; limit?: number }) {
  const medications = new Map<string, Planned & { doses: number }>();
  for (const items of plan.values()) {
    for (const item of items) {
      const known = medications.get(item.id);
      medications.set(item.id, { ...item, doses: (known?.doses ?? 0) + 1 });
    }
  }
  const all = [...medications.values()].sort((a, b) => b.doses - a.doses || a.name.localeCompare(b.name));
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="overline">In this pack · {all.length} medications</div>
      {all.slice(0, limit).map((m) => (
        <div key={m.id} className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
          <span
            className="dot"
            style={{
              background: pillColour(m.appearance?.colour),
              width: 9,
              height: 9,
              boxShadow: "inset 0 0 0 1px rgba(0, 0, 0, 0.22)",
            }}
          />
          <span className="ellipsis" style={{ fontWeight: 500 }}>
            {m.name}
          </span>
          <span className="small faint ellipsis" style={{ flex: 1 }}>
            {describeAppearance(m.appearance)}
          </span>
          <span className="small muted mono">
            {m.doses} dose{m.doses === 1 ? "" : "s"}
          </span>
        </div>
      ))}
      {all.length > limit && <div className="small faint">+ {all.length - limit} more</div>}
    </div>
  );
}
