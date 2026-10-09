import { useEffect, useState } from "react";
import { api } from "../api";
import { useAction } from "../hooks";
import type { AuditSummary } from "../types";

type Page = { total: number; entries: AuditSummary[] };

export function AuditPage({ onChange }: { onChange: () => void }) {
  const [entries, setEntries] = useState<AuditSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [detail, setDetail] = useState<{ entry: AuditSummary; record: Record<string, unknown> } | null>(null);
  const { busy, error, run } = useAction();

  const load = (before?: number) =>
    run("load", async () => {
      const page = await api.get<Page>(`/api/audit?limit=50${before !== undefined ? `&before=${before}` : ""}`);
      setTotal(page.total);
      setEntries((current) => (before === undefined ? page.entries : [...current, ...page.entries]));
    });

  useEffect(() => {
    load();
  }, []);

  const verify = () =>
    run("verify", async () => {
      const result = await api.post<{ intact: boolean; defect: { sequence: number; defect: string } | null }>(
        "/api/audit/verify",
      );
      setVerdict(
        result.intact
          ? `Intact: all ${total} entries verified.`
          : `BROKEN at entry ${result.defect?.sequence} (${result.defect?.defect}). New checks cannot be recorded.`,
      );
      onChange();
    });

  const open = (sequence: number) =>
    run("open", async () => setDetail(await api.get(`/api/audit/${sequence}`)));

  return (
    <div className="split">
      <div className="panel stack">
        <div className="row">
          <h2 style={{ marginRight: "auto" }}>Audit trail ({total})</h2>
          <button onClick={verify} disabled={busy !== null}>
            Verify hash chain
          </button>
        </div>
        {verdict && <div className={verdict.startsWith("Intact") ? "notice" : "error"}>{verdict}</div>}
        {error && <div className="error">{error}</div>}
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>Pack</th>
              <th>Result</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.sequence} onClick={() => open(e.sequence)} style={{ cursor: "pointer" }}>
                <td>{e.sequence}</td>
                <td className="small">{new Date(e.created_at).toLocaleString()}</td>
                <td>{e.reference}</td>
                <td>
                  <span className={`small status-${e.status}`}>{e.status}</span>
                </td>
                <td>
                  {e.decision} · {e.pharmacist}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {entries.length < total && (
          <button onClick={() => load(entries[entries.length - 1].sequence)} disabled={busy !== null}>
            Load older
          </button>
        )}
      </div>
      {detail && (
        <div className="panel stack">
          <h2>
            Entry #{detail.entry.sequence} · {detail.entry.reference}
          </h2>
          <div className="small muted">hash {detail.entry.hash}</div>
          {detail.entry.evidence.map((file) => (
            <div key={file} className="media">
              <img src={`/api/evidence/${file}`} alt="Evidence" />
            </div>
          ))}
          <pre className="small" style={{ maxHeight: "28rem", overflow: "auto", whiteSpace: "pre-wrap" }}>
            {JSON.stringify(detail.record, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
