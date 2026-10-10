import { useState } from "react";
import { api } from "../api";
import { useAction, useData } from "../hooks";
import type { BrainView, Catalog } from "../types";

const TRUST_TEXT = { learning: "Learning", trusted: "Trusted", suspended: "Suspended" };

export function BrainPage({ onChange }: { onChange: () => void }) {
  const { data: brain, reload } = useData<BrainView>("/api/brain");
  const { data: catalog } = useData<Catalog>("/api/catalog");
  const [teachId, setTeachId] = useState("");
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [message, setMessage] = useState<string | null>(null);
  const { busy, error, run } = useAction();

  const refresh = () => {
    reload();
    onChange();
  };

  const teach = () =>
    run("teach", async () => {
      const result = await api.post<{ pills: number; learning: { exemplars_added: number; duplicates_skipped: number } }>(
        "/api/brain/teach",
        { medication_id: teachId },
      );
      setMessage(
        `Learned from ${result.pills} tablets: ${result.learning.exemplars_added} new, ${result.learning.duplicates_skipped} already known.`,
      );
      refresh();
    });

  const resolve = (taskId: string, chosen: Record<string, string>) =>
    run("resolve", async () => {
      await api.post(`/api/brain/tasks/${taskId}/resolve`, { labels: chosen });
      refresh();
    });

  const discard = (taskId: string) =>
    run("discard", async () => {
      await api.delete(`/api/brain/tasks/${taskId}`);
      refresh();
    });

  const forget = (id: string, name: string) =>
    window.confirm(`Forget everything the brain learned about ${name}? Its trust starts again from zero.`) &&
    run("forget", async () => {
      await api.delete(`/api/brain/medications/${encodeURIComponent(id)}`);
      refresh();
    });

  const recalibrate = () =>
    run("recalibrate", async () => {
      await api.post("/api/brain/recalibrate");
      refresh();
    });

  const names = new Map(catalog?.medications.map((m) => [m.id, `${m.name} ${m.strength}`.trim()]));

  if (!brain) return <div className="panel muted">Loading…</div>;
  const tp = brain.trust_policy;

  return (
    <div className="stack">
      {error && <div className="error">{error}</div>}
      {message && <div className="notice">{message}</div>}
      <div className="split">
        <div className="panel stack">
          <h2>The brain</h2>
          <div className="row small">
            <span>
              <b>{brain.summary.exemplar_count}</b> pill images
            </span>
            <span>
              <b>{brain.summary.known_medications}</b> medications known
            </span>
            <span>
              <b>{brain.summary.trusted_medications.length}</b> trusted
            </span>
            <span className="muted">embedder {brain.summary.embedder_id}</span>
          </div>
          <p className="small muted">
            The brain can always flag a pill that looks wrong. It may confirm a medication's identity only after{" "}
            {tp.minimum_predictions} pharmacist-checked identifications with a 95% lower bound of precision above{" "}
            {Math.round(tp.minimum_precision_lower_bound * 100)}% and {tp.minimum_streak} correct in a row. One wrong
            identification suspends it.
          </p>
          <div className="small">
            Self-test:{" "}
            {brain.policy.calibrated
              ? `calibrated (threshold ${brain.policy.accept_similarity?.toFixed(3)})`
              : "not calibrated yet — needs at least two medications taught from different packs"}
            {brain.calibration &&
              ` · last run ${new Date(brain.calibration.calibrated_at).toLocaleString()}: ${brain.calibration.outcome}, precision ≥ ${(brain.calibration.precision_lower_bound * 100).toFixed(1)}%, coverage ${(brain.calibration.coverage * 100).toFixed(0)}%`}
          </div>
          <div className="row">
            <button onClick={recalibrate} disabled={busy !== null}>
              Run self-test now
            </button>
          </div>
        </div>
        <div className="panel stack">
          <h2>Teach from the camera</h2>
          <p className="small muted">
            Fill the pack (or a tray) with tablets of one medication only, check them yourself, then teach. Teach each
            medication from several different packs.
          </p>
          <div className="row">
            <select value={teachId} onChange={(e) => setTeachId(e.target.value)}>
              <option value="">— medication —</option>
              {catalog?.medications.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} {m.strength}
                </option>
              ))}
            </select>
            <button className="primary" disabled={!teachId || busy !== null} onClick={teach}>
              {busy === "teach" ? "Learning…" : "Teach"}
            </button>
          </div>
        </div>
      </div>

      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>Medication</th>
              <th>Images</th>
              <th>Packs</th>
              <th>Trust</th>
              <th>Track record</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {brain.medications.map((m) => (
              <tr key={m.id}>
                <td>{m.name}</td>
                <td>{m.exemplars}</td>
                <td>{m.groups}</td>
                <td>
                  <span className={m.trust === "trusted" ? "status-verified" : m.trust === "suspended" ? "status-mismatch" : ""}>
                    {TRUST_TEXT[m.trust]}
                  </span>
                  <div className="progress">
                    <div style={{ width: `${m.progress * 100}%` }} />
                  </div>
                </td>
                <td className="small">
                  {m.correct} correct · {m.false_identifications} wrong · streak {m.streak}
                </td>
                <td>
                  {m.exemplars > 0 && (
                    <button className="danger" onClick={() => forget(m.id, m.name)}>
                      Forget
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel stack">
        <h2>Labelling queue ({brain.tasks.length})</h2>
        <p className="small muted">
          Compartments you confirmed that hold several medications. Say which tablet is which and the brain learns from
          them.
        </p>
        {brain.tasks.slice(0, 5).map((task) => {
          const chosen = Object.fromEntries(task.pills.map((p) => [p.id, labels[p.id] ?? p.suggestion ?? ""]));
          const ready = Object.values(chosen).every(Boolean);
          return (
            <div key={task.id} className="stack" style={{ borderTop: "1px solid var(--border)", paddingTop: "0.75rem" }}>
              <div className="small">
                <b>{task.compartment}</b> · expected{" "}
                {Object.entries(task.expected)
                  .map(([id, q]) => `${q} × ${names.get(id) ?? id}`)
                  .join(", ")}
              </div>
              <div className="crops">
                {task.pills.map((pill) => (
                  <figure key={pill.id}>
                    {pill.crop && <img src={`/api/crops/${pill.crop}`} alt="tablet" />}
                    <select value={chosen[pill.id]} onChange={(e) => setLabels({ ...labels, [pill.id]: e.target.value })}>
                      <option value="">—</option>
                      {Object.keys(task.expected).map((id) => (
                        <option key={id} value={id}>
                          {names.get(id) ?? id}
                        </option>
                      ))}
                    </select>
                  </figure>
                ))}
              </div>
              <div className="row">
                <button className="primary" disabled={!ready || busy !== null} onClick={() => resolve(task.id, chosen)}>
                  Save labels
                </button>
                <button onClick={() => discard(task.id)} disabled={busy !== null}>
                  Skip
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
