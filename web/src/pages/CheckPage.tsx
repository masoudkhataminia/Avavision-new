import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { EvidenceView } from "../components/EvidenceView";
import { LiveView } from "../components/LiveView";
import { key, PackGrid } from "../components/PackGrid";
import { useAction, useData } from "../hooks";
import type { CheckView, CompartmentView, ProfileSummary, ReviewOutcome, Status } from "../types";

const PACK_STATUS: Record<string, string> = {
  verified: "Verified — spot checks still required",
  countMatched: "Counts match — identity not checked automatically",
  needsReview: "Needs review",
  mismatch: "Mismatch found",
  retakeRequired: "Retake required",
};

const FAULTS = ["none", "missing", "extra", "swapped", "foreign", "emptyTray"];

function storedInitials(): string {
  try {
    return localStorage.getItem("avavision.pharmacist") ?? "";
  } catch {
    return "";
  }
}

export function CheckPage({ status, onChange }: { status: Status | null; onChange: () => void }) {
  const [check, setCheck] = useState<CheckView | null>(null);
  const [profileId, setProfileId] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [reviews, setReviews] = useState<Record<string, ReviewOutcome>>({});
  const [pharmacist, setPharmacist] = useState(storedInitials);
  const [acknowledged, setAcknowledged] = useState(false);
  const [note, setNote] = useState("");
  const [fault, setFault] = useState("none");
  const { busy, error, run } = useAction();
  const { data: profiles } = useData<ProfileSummary[]>("/api/profiles");

  const update = useCallback(
    (view: CheckView | null | undefined) => {
      if (view === undefined) return;
      setCheck(view);
      onChange();
    },
    [onChange],
  );

  useEffect(() => {
    api.get<CheckView | null>("/api/check").then((view) => {
      setCheck(view);
      if (view) setProfileId(view.profile.id);
    });
  }, []);

  useEffect(() => {
    if (!profileId && profiles?.length) setProfileId(profiles[0].id);
  }, [profiles, profileId]);

  const resetReview = () => {
    setReviews({});
    setSelected(null);
    setAcknowledged(false);
    setNote("");
  };

  const start = () =>
    run("start", async () => {
      resetReview();
      setFault("none");
      update(await api.post<CheckView>("/api/check/start", { profile_id: profileId }));
    });
  const capture = useCallback(
    () => run("capture", async () => update(await api.post<CheckView>("/api/check/capture"))),
    [run, update],
  );
  const retake = () =>
    run("retake", async () => {
      resetReview();
      update(await api.post<CheckView>("/api/check/retake"));
    });
  const cancel = () =>
    run("cancel", async () => {
      await api.post("/api/check/cancel");
      resetReview();
      update(null);
    });
  const secondOpinion = () => run("review", async () => update(await api.post<CheckView>("/api/check/review", {})));
  const explain = () => run("explain", async () => update(await api.post<CheckView>("/api/check/explain")));
  const injectFault = (value: string) =>
    run("fault", async () => {
      setFault(value);
      await api.post("/api/demo/fault", { fault: value });
    });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.code === "Space" && check?.phase === "capturing" && !busy && target.tagName !== "INPUT") {
        event.preventDefault();
        capture();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [check, busy, capture]);

  const result = check?.result ?? null;
  const compartments = result?.compartments ?? [];
  const requiring = compartments.filter((c) => c.requires_review);
  const unreviewed = requiring.filter((c) => !reviews[key(c)]);
  const unresolved = compartments.filter((c) => reviews[key(c)] === "unresolved");
  const releaseBlockers = [
    !pharmacist.trim() && "enter your initials",
    unreviewed.length > 0 && `${unreviewed.length} compartment(s) still to inspect`,
    unresolved.length > 0 && `${unresolved.length} compartment(s) unresolved`,
    result?.has_pack_findings && !acknowledged && "acknowledge the pack findings",
    result?.status === "retakeRequired" && "retake the photos",
  ].filter(Boolean) as string[];
  const current = useMemo(() => compartments.find((c) => key(c) === selected) ?? null, [compartments, selected]);

  const setReview = (c: CompartmentView, outcome: ReviewOutcome) => setReviews((r) => ({ ...r, [key(c)]: outcome }));
  const confirmRemaining = () => {
    const easy = unreviewed.filter((c) => c.status === "verified" || c.status === "countMatched");
    if (!easy.length) return;
    if (!window.confirm(`I have inspected these ${easy.length} compartments and their content is correct.`)) return;
    setReviews((r) => ({ ...r, ...Object.fromEntries(easy.map((c) => [key(c), "confirmedCorrect" as const])) }));
  };

  const signOff = (decision: "released" | "withheld") =>
    run("sign", async () => {
      try {
        localStorage.setItem("avavision.pharmacist", pharmacist);
      } catch {
        /* storage unavailable */
      }
      const body = {
        pharmacist,
        decision,
        reviews: Object.entries(reviews).map(([k, outcome]) => {
          const [row, column] = k.split(":").map(Number);
          return { compartment: { row, column }, outcome };
        }),
        acknowledged_pack_findings: acknowledged,
        note: note || null,
      };
      update(await api.post<CheckView>("/api/check/sign-off", body));
    });

  const starter = (
    <div className="panel row">
      <label>
        Pack profile
        <select value={profileId} onChange={(e) => setProfileId(e.target.value)}>
          {profiles?.map((p) => (
            <option key={p.id} value={p.id}>
              {p.reference} · {p.doses} doses{p.issues ? " · has issues" : ""}
            </option>
          ))}
        </select>
      </label>
      <button className="primary big" disabled={!profileId || busy !== null} onClick={start}>
        Start check
      </button>
      {!profiles?.length && <span className="muted">Create a profile first (Profiles).</span>}
    </div>
  );

  return (
    <div className="stack">
      {error && <div className="error">{error}</div>}

      {(!check || check.phase === "completed") && starter}

      {check?.phase === "completed" && (
        <div className="panel stack">
          <h2>
            Pack {check.profile.reference} signed off · audit entry #{check.audit_sequence}
          </h2>
          {check.learning && (
            <p className="muted">
              The brain learned {check.learning.exemplars_added} new pill images, recorded{" "}
              {check.learning.observations_recorded} identifications and queued {check.learning.tasks_queued} labelling
              task(s).
            </p>
          )}
        </div>
      )}

      {check?.phase === "capturing" && (
        <div className="split">
          <LiveView />
          <div className="panel stack">
            <h2>Checking {check.profile.reference}</h2>
            <p className="muted">
              Place the pack on the tray with all four markers visible, then capture (<kbd>Space</kbd>).
            </p>
            <div className="row">
              <button className="primary big" disabled={busy !== null} onClick={capture}>
                {busy === "capture" ? "Analysing…" : "Capture"}
              </button>
              <button onClick={cancel} disabled={busy !== null}>
                Cancel
              </button>
            </div>
            {status?.demo && (
              <label>
                Demo: packing error to simulate
                <select value={fault} onChange={(e) => injectFault(e.target.value)}>
                  {FAULTS.map((f) => (
                    <option key={f}>{f}</option>
                  ))}
                </select>
              </label>
            )}
          </div>
        </div>
      )}

      {check?.phase === "analyzed" && result && (
        <div className="split">
          <div className="stack">
            {check.evidence ? <EvidenceView check={check} selected={selected} /> : <LiveView />}
            <div className="panel">
              <PackGrid
                layout={check.layout}
                compartments={compartments}
                selected={selected}
                reviews={reviews}
                onSelect={setSelected}
              />
            </div>
          </div>

          <div className="stack">
            <div className={`banner status-${result.status}`}>
              {check.profile.reference}: {PACK_STATUS[result.status] ?? result.status}
            </div>
            {result.pack_findings.map((f) => (
              <div key={f} className="error">
                {f}
              </div>
            ))}
            <div className="muted small">
              {typeof check.timings.total_ms === "number" && `Analysed in ${Math.round(check.timings.total_ms)} ms · `}
              {result.usable_frames} usable photos · {result.capability === "identity" ? "identity model" : "count model"}
              {result.trusted_medications.length > 0 && ` · brain trusted for ${result.trusted_medications.length}`}
            </div>

            {current ? (
              <div className="panel stack">
                <h3>
                  {current.label} <span className={`small status-${current.status}`}>{current.status}</span>
                </h3>
                <div className="small">
                  Expected:{" "}
                  {current.expected.length
                    ? current.expected.map((e) => `${e.quantity} × ${e.name}`).join(", ")
                    : "empty"}{" "}
                  · seen: {current.observed_count ?? "uncertain"}
                </div>
                {current.findings.length > 0 && (
                  <ul className="small">
                    {current.findings.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                )}
                {current.advisory && (
                  <div className="notice small">
                    Second opinion ({current.advisory.source}): {current.advisory.verdict}
                    {current.advisory.observed_count !== null && `, counted ${current.advisory.observed_count}`}
                    {current.advisory.note && ` — ${current.advisory.note}`}
                  </div>
                )}
                <div className="row">
                  <button
                    className={reviews[key(current)] === "confirmedCorrect" ? "primary" : ""}
                    onClick={() => setReview(current, "confirmedCorrect")}
                  >
                    ✓ Correct
                  </button>
                  <button
                    className={reviews[key(current)] === "corrected" ? "primary" : ""}
                    onClick={() => setReview(current, "corrected")}
                  >
                    ✎ Corrected
                  </button>
                  <button
                    className={reviews[key(current)] === "unresolved" ? "primary" : ""}
                    onClick={() => setReview(current, "unresolved")}
                  >
                    ✗ Unresolved
                  </button>
                </div>
              </div>
            ) : (
              <div className="panel muted">Select a compartment to inspect it.</div>
            )}

            <div className="panel stack">
              <div className="row">
                <button onClick={confirmRemaining} disabled={!unreviewed.some((c) => c.status !== "needsReview" && c.status !== "mismatch")}>
                  Confirm remaining OK compartments
                </button>
                <button onClick={retake} disabled={busy !== null}>
                  Retake
                </button>
                {status?.expert.enabled && (
                  <>
                    <button onClick={secondOpinion} disabled={busy !== null || !check.evidence}>
                      {busy === "review" ? "Asking…" : "Second opinion"}
                    </button>
                    <button onClick={explain} disabled={busy !== null}>
                      {busy === "explain" ? "Explaining…" : "Explain"}
                    </button>
                  </>
                )}
              </div>
              {check.review_errors.length > 0 && <div className="error small">{check.review_errors.join("; ")}</div>}
              {check.explanation && (
                <div className="notice">
                  <p>{check.explanation.summary}</p>
                  <ol className="small">
                    {check.explanation.steps.map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ol>
                </div>
              )}
            </div>

            <div className="panel stack">
              <h3>Pharmacist sign-off</h3>
              <div className="row">
                <label>
                  Initials
                  <input value={pharmacist} onChange={(e) => setPharmacist(e.target.value)} size={8} />
                </label>
                <label style={{ flex: 1 }}>
                  Note
                  <input value={note} onChange={(e) => setNote(e.target.value)} />
                </label>
              </div>
              {result.has_pack_findings && (
                <label className="inline">
                  <input type="checkbox" checked={acknowledged} onChange={(e) => setAcknowledged(e.target.checked)} />
                  I have dealt with the pack findings above
                </label>
              )}
              {releaseBlockers.length > 0 && <div className="small muted">To release: {releaseBlockers.join(" · ")}</div>}
              <div className="row">
                <button
                  className="primary big"
                  disabled={busy !== null || releaseBlockers.length > 0}
                  onClick={() => signOff("released")}
                >
                  Release pack
                </button>
                <button className="danger" disabled={busy !== null || !pharmacist.trim()} onClick={() => signOff("withheld")}>
                  Withhold
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
