import { useMemo, useState } from "react";
import { api } from "../api";
import { useAction, useData } from "../hooks";
import type { Catalog, Draft, ExpectedItem, Layout, Profile, ProfileIssue, ProfileSummary, Status } from "../types";

const ck = (row: number, column: number) => `${row}:${column}`;

function emptyProfile(layout: Layout): Profile {
  const compartments = [];
  for (let row = 0; row < layout.rows; row++)
    for (let column = 0; column < layout.columns; column++) compartments.push({ compartment: { row, column }, items: [] });
  return { id: crypto.randomUUID(), reference: "", layout_id: layout.id, compartments, created_at: new Date().toISOString() };
}

const ISSUE_TEXT: Record<string, string> = {
  unsupportedLayout: "This layout cannot be filled from a chart",
  notMatched: "No catalog medication matches",
  ambiguousMatch: "Several catalog medications match",
  fractionalDose: "Half tablets are not placed automatically",
  unknownStartDay: "The chart does not say which weekday the pack starts",
  unclear: "Not read with certainty",
  noDoses: "No doses found",
  scheduleNote: "Schedule needs checking by hand",
  chartWarning: "Chart warning",
};

export function ProfilesPage() {
  const { data: profiles, reload } = useData<ProfileSummary[]>("/api/profiles");
  const { data: layouts } = useData<Layout[]>("/api/layouts");
  const { data: catalog } = useData<Catalog>("/api/catalog");
  const { data: status } = useData<Status>("/api/status");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [issues, setIssues] = useState<ProfileIssue[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [overrides, setOverrides] = useState<Record<number, string>>({});
  const [importReference, setImportReference] = useState("");
  const [saved, setSaved] = useState(false);
  const { busy, error, run } = useAction();

  const layout = useMemo(
    () => layouts?.find((l) => l.id === (profile?.layout_id ?? status?.layout.id)) ?? layouts?.[0] ?? null,
    [layouts, profile, status],
  );
  const names = useMemo(() => new Map(catalog?.medications.map((m) => [m.id, `${m.name} ${m.strength}`.trim()])), [catalog]);

  const open = (id: string) =>
    run("open", async () => {
      const response = await api.get<{ profile: Profile; issues: ProfileIssue[] }>(`/api/profiles/${id}`);
      setProfile(response.profile);
      setIssues(response.issues);
      setSelected(null);
      setDraft(null);
      setSaved(true);
    });

  const save = () =>
    profile &&
    run("save", async () => {
      const response = await api.put<{ profile: Profile; issues: ProfileIssue[] }>(`/api/profiles/${profile.id}`, profile);
      setIssues(response.issues);
      setSaved(true);
      reload();
    });

  const remove = (id: string) =>
    window.confirm("Delete this profile?") &&
    run("delete", async () => {
      await api.delete(`/api/profiles/${id}`);
      if (profile?.id === id) setProfile(null);
      reload();
    });

  const items = (row: number, column: number): ExpectedItem[] =>
    profile?.compartments.find((c) => c.compartment.row === row && c.compartment.column === column)?.items ?? [];

  const setItems = (targets: [number, number][], next: ExpectedItem[]) => {
    if (!profile) return;
    const keys = new Set(targets.map(([r, c]) => ck(r, c)));
    setProfile({
      ...profile,
      compartments: profile.compartments.map((c) =>
        keys.has(ck(c.compartment.row, c.compartment.column)) ? { ...c, items: next.map((i) => ({ ...i })) } : c,
      ),
    });
    setSaved(false);
  };

  const importChart = (file: File) =>
    run("import", async () => {
      const params = new URLSearchParams({ reference: importReference || file.name, filename: file.name });
      if (layout) params.set("layout_id", layout.id);
      const response = await api.upload<{ draft: Draft }>(`/api/profiles/import?${params}`, file);
      setDraft(response.draft);
      setOverrides({});
    });

  const redraft = () =>
    draft &&
    run("redraft", async () => {
      const response = await api.post<{ draft: Draft }>("/api/profiles/redraft", {
        chart: draft.chart,
        reference: draft.profile.reference,
        layout_id: draft.profile.layout_id,
        overrides,
      });
      setDraft(response.draft);
    });

  const [row, column] = selected ? selected.split(":").map(Number) : [-1, -1];
  const cellItems = selected ? items(row, column) : [];

  return (
    <div className="stack">
      {error && <div className="error">{error}</div>}
      <div className="split">
        <div className="panel stack">
          <div className="row">
            <h2 style={{ marginRight: "auto" }}>Pack profiles</h2>
            <button
              className="primary"
              disabled={!layout}
              onClick={() => {
                if (!layout) return;
                setProfile(emptyProfile(layout));
                setIssues([]);
                setDraft(null);
                setSaved(false);
              }}
            >
              New profile
            </button>
          </div>
          <table>
            <thead>
              <tr>
                <th>Reference</th>
                <th>Doses</th>
                <th>Layout</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {profiles?.map((p) => (
                <tr key={p.id}>
                  <td>
                    <a href="#profiles" onClick={() => open(p.id)}>
                      {p.reference}
                    </a>
                    {p.issues > 0 && <span className="small status-needsReview"> {p.issues} issue(s)</span>}
                  </td>
                  <td>{p.doses}</td>
                  <td className="small muted">{p.layout_id}</td>
                  <td>
                    <button className="danger" onClick={() => remove(p.id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel stack">
          <h2>Import a medication chart</h2>
          <p className="muted small">
            PDF, photo or text of a packing sheet. The expert reads it; matching and placing doses is done by the
            station, and every doubt is listed. Nothing is used until you review and save the profile.
          </p>
          <div className="row">
            <label>
              Pack reference
              <input value={importReference} onChange={(e) => setImportReference(e.target.value)} />
            </label>
            <label>
              Chart file
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.webp,.txt,.csv"
                disabled={busy !== null}
                onChange={(e) => e.target.files?.[0] && importChart(e.target.files[0])}
              />
            </label>
          </div>
          {busy === "import" && <div className="notice">Reading the chart… (this can take up to a minute)</div>}
          {draft && (
            <div className="stack">
              <table className="small">
                <thead>
                  <tr>
                    <th>Line</th>
                    <th>Doses</th>
                    <th>Catalog medication</th>
                  </tr>
                </thead>
                <tbody>
                  {draft.chart.lines.map((line, i) => (
                    <tr key={i}>
                      <td>
                        {line.name} {line.strength}
                        {!line.in_pack && <span className="muted"> (not packed)</span>}
                        {draft.issues
                          .filter((issue) => issue.line === i)
                          .map((issue, k) => (
                            <div key={k} className="status-needsReview">
                              {ISSUE_TEXT[issue.kind] ?? issue.kind}
                              {issue.detail && `: ${issue.detail}`}
                            </div>
                          ))}
                      </td>
                      <td>
                        {line.doses.map((d) => `${d.quantity} ${d.time}`).join(", ")}
                        <div className="muted">{line.days.length === 7 ? "daily" : line.days.join(", ")}</div>
                      </td>
                      <td>
                        {line.in_pack && (
                          <select
                            value={overrides[i] ?? draft.matches[String(i)] ?? ""}
                            onChange={(e) => setOverrides({ ...overrides, [i]: e.target.value })}
                          >
                            <option value="">— choose —</option>
                            {catalog?.medications.map((m) => (
                              <option key={m.id} value={m.id}>
                                {m.name} {m.strength}
                              </option>
                            ))}
                          </select>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {draft.issues
                .filter((issue) => issue.line === null)
                .map((issue, k) => (
                  <div key={k} className="error small">
                    {ISSUE_TEXT[issue.kind] ?? issue.kind}: {issue.detail}
                  </div>
                ))}
              <div className="row">
                <button onClick={redraft} disabled={busy !== null}>
                  Apply my choices
                </button>
                <button
                  className="primary"
                  onClick={() => {
                    setProfile(draft.profile);
                    setIssues(draft.profile_issues);
                    setSaved(false);
                  }}
                >
                  Open in editor
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {profile && layout && (
        <div className="split">
          <div className="panel stack">
            <div className="row">
              <label>
                Reference
                <input
                  value={profile.reference}
                  onChange={(e) => {
                    setProfile({ ...profile, reference: e.target.value });
                    setSaved(false);
                  }}
                />
              </label>
              <label>
                Header-card code (QR / barcode)
                <input
                  value={profile.barcode ?? ""}
                  onChange={(e) => {
                    setProfile({ ...profile, barcode: e.target.value.trim() || null });
                    setSaved(false);
                  }}
                />
              </label>
              <span className="muted small">{layout.display_name}</span>
              <button className="primary" onClick={save} disabled={busy !== null || saved}>
                {saved ? "Saved" : "Save profile"}
              </button>
            </div>
            <div className="editor-grid" style={{ gridTemplateColumns: `5rem repeat(${layout.columns}, minmax(0, 1fr))` }}>
              <span />
              {layout.column_labels.map((l) => (
                <span key={l} className="small muted">
                  {l}
                </span>
              ))}
              {layout.row_labels.map((label, r) => [
                <span key={`l${r}`} className="small muted">
                  {label}
                </span>,
                ...layout.column_labels.map((_, c) => (
                  <div
                    key={ck(r, c)}
                    className={`editor-cell ${selected === ck(r, c) ? "selected" : ""}`}
                    onClick={() => setSelected(ck(r, c))}
                  >
                    {items(r, c).map((i) => (
                      <div key={i.medication_id}>
                        {i.quantity} × {names.get(i.medication_id) ?? i.medication_id}
                      </div>
                    ))}
                  </div>
                )),
              ])}
            </div>
            {issues.length > 0 && (
              <div className="error small">
                {issues.length} issue(s): {[...new Set(issues.map((i) => i.kind))].join(", ")}
              </div>
            )}
          </div>

          <div className="panel stack">
            {selected ? (
              <>
                <h3>
                  {layout.column_labels[column]} · {layout.row_labels[row]}
                </h3>
                {cellItems.map((item, k) => (
                  <div key={k} className="row">
                    <select
                      value={item.medication_id}
                      onChange={(e) =>
                        setItems(
                          [[row, column]],
                          cellItems.map((x, j) => (j === k ? { ...x, medication_id: e.target.value } : x)),
                        )
                      }
                    >
                      {catalog?.medications.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} {m.strength}
                        </option>
                      ))}
                    </select>
                    <input
                      type="number"
                      min={1}
                      value={item.quantity}
                      style={{ width: "5rem" }}
                      onChange={(e) =>
                        setItems(
                          [[row, column]],
                          cellItems.map((x, j) => (j === k ? { ...x, quantity: Number(e.target.value) } : x)),
                        )
                      }
                    />
                    <button onClick={() => setItems([[row, column]], cellItems.filter((_, j) => j !== k))}>Remove</button>
                  </div>
                ))}
                <div className="row">
                  <button
                    disabled={!catalog?.medications.length}
                    onClick={() =>
                      setItems([[row, column]], [...cellItems, { medication_id: catalog!.medications[0].id, quantity: 1 }])
                    }
                  >
                    Add medication
                  </button>
                  <button
                    onClick={() =>
                      setItems(
                        layout.column_labels.map((_, c) => [row, c] as [number, number]),
                        cellItems,
                      )
                    }
                  >
                    Copy to every day
                  </button>
                </div>
              </>
            ) : (
              <p className="muted">Select a compartment to edit what belongs in it.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
