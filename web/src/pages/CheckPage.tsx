import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { Heading } from "../App";
import { EvidenceView } from "../components/EvidenceView";
import { Icon, type IconName } from "../components/Icon";
import { CaptureQuality, ISSUE_TEXT, LiveView } from "../components/LiveView";
import { key, PackGrid, PackLegend, PackMedications, planFor, STATUS_TEXT } from "../components/PackGrid";
import { useAction, useData } from "../hooks";
import type {
  AuditSummary,
  Catalog,
  CheckView,
  CompartmentView,
  Layout,
  Live,
  Profile,
  ProfileSummary,
  ReviewOutcome,
  Status,
} from "../types";

const PACK_STATUS: Record<string, string> = {
  verified: "Verified",
  countMatched: "Counts match",
  needsReview: "Needs review",
  mismatch: "Mismatch found",
  retakeRequired: "Retake required",
};

const PACK_NOTE: Record<string, string> = {
  verified: "Spot checks are still required",
  countMatched: "Identity is not checked automatically: look at every compartment",
  needsReview: "Some compartments need your eyes",
  mismatch: "At least one compartment is wrong",
  retakeRequired: "The photos are not good enough to judge the pack",
};

const STATUS_ICON: Record<string, IconName> = {
  verified: "circle-check",
  countMatched: "list-checks",
  needsReview: "eye",
  mismatch: "triangle-alert",
  retakeRequired: "rotate-ccw",
};

const REVIEW_TEXT: Record<ReviewOutcome, string> = {
  confirmedCorrect: "Checked",
  corrected: "Corrected",
  unresolved: "Unresolved",
};

const FAULTS = ["none", "missing", "extra", "swapped", "foreign", "emptyTray", "wrongCard"];

function storedInitials(): string {
  try {
    return localStorage.getItem("avavision.pharmacist") ?? "";
  } catch {
    return "";
  }
}

const clock = (iso?: string) =>
  (iso ? new Date(iso) : new Date()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function CheckItem({ state, title, detail }: { state: "ok" | "warn" | "idle"; title: string; detail?: string }) {
  const icon: IconName = state === "ok" ? "check" : state === "warn" ? "triangle-alert" : "circle-dot";
  const colour = state === "ok" ? "status-verified" : state === "warn" ? "status-needsReview" : "faint";
  return (
    <div className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
      <span className={colour} style={{ marginTop: 2 }}>
        <Icon name={icon} size={18} />
      </span>
      <span className="stack" style={{ gap: 0 }}>
        <span>{title}</span>
        {detail && <span className="small faint">{detail}</span>}
      </span>
    </div>
  );
}

export function CheckPage({
  status,
  onChange,
  onHeading,
}: {
  status: Status | null;
  onChange: () => void;
  onHeading: (heading: Heading | null) => void;
}) {
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
  const { data: catalog } = useData<Catalog>("/api/catalog");
  const { data: layouts } = useData<Layout[]>("/api/layouts");
  const { data: audit, reload: reloadAudit } = useData<{ total: number; entries: AuditSummary[] }>(
    "/api/audit?limit=5",
  );
  const analysed = check?.phase === "analyzed";
  const { data: live } = useData<Live>(analysed ? null : "/api/live", 200);
  const shownProfileId = check && check.phase !== "completed" ? check.profile.id : profileId;
  const { data: profileData } = useData<{ profile: Profile }>(
    shownProfileId ? `/api/profiles/${encodeURIComponent(shownProfileId)}` : null,
  );
  const profile = profileData?.profile.id === shownProfileId ? profileData.profile : null;

  const update = useCallback(
    (view: CheckView | null | undefined) => {
      if (view === undefined) return;
      setCheck(view);
      onChange();
      if (view?.phase === "completed") reloadAudit();
    },
    [onChange, reloadAudit],
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
  const secondOpinion = (advisor: "claude" | "local") =>
    run("review", async () => update(await api.post<CheckView>("/api/check/review", { advisor })));
  const stopReview = () => run("stop", async () => update(await api.post<CheckView>("/api/check/review/stop")));

  // Second opinions arrive one compartment at a time: follow them while they run.
  const following = check?.reviewing != null || busy === "review";
  useEffect(() => {
    if (!following) return;
    const timer = window.setInterval(() => {
      api
        .get<CheckView | null>("/api/check")
        .then((view) => view && view.phase === "analyzed" && setCheck(view))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [following]);
  const explain = () => run("explain", async () => update(await api.post<CheckView>("/api/check/explain")));
  const injectFault = (value: string) =>
    run("fault", async () => {
      setFault(value);
      await api.post("/api/demo/fault", { fault: value });
    });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        event.code === "Space" &&
        check?.phase === "capturing" &&
        !busy &&
        !["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)
      ) {
        event.preventDefault();
        capture();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [check, busy, capture]);

  const result = check?.result ?? null;
  const compartments = useMemo(() => (analysed ? (result?.compartments ?? []) : []), [analysed, result]);
  const requiring = compartments.filter((c) => c.requires_review);
  const unreviewed = requiring.filter((c) => !reviews[key(c)]);
  const unresolved = compartments.filter((c) => reviews[key(c)] === "unresolved");
  const releaseBlockers = [
    !pharmacist.trim() && "enter your initials",
    unreviewed.length > 0 && `${unreviewed.length} compartment(s) still to inspect`,
    unresolved.length > 0 && `${unresolved.length} compartment(s) unresolved`,
    result?.has_pack_findings && !acknowledged && "acknowledge the pack findings",
    result?.status === "retakeRequired" && "retake the photos",
    check?.reviewing && `wait for the second opinion (${check.reviewing.done} of ${check.reviewing.total})`,
  ].filter(Boolean) as string[];
  const current = useMemo(() => compartments.find((c) => key(c) === selected) ?? null, [compartments, selected]);

  const setReview = (c: CompartmentView, outcome: ReviewOutcome) => setReviews((r) => ({ ...r, [key(c)]: outcome }));
  const easy = unreviewed.filter((c) => c.status === "countMatched" && !c.spot_check);
  const confirmRemaining = () => {
    // Spot checks and flagged compartments must each be inspected and marked on their own.
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

  // A header card in view opens its profile and starts the check (once per card).
  const [lastCard, setLastCard] = useState<string | null>(null);
  const onCodes = useCallback(
    (codes: string[]) => {
      if (busy || (check && check.phase !== "completed")) return;
      const match = profiles?.find((p) => p.barcode && codes.includes(p.barcode) && p.barcode !== lastCard);
      if (!match) return;
      setLastCard(match.barcode);
      setProfileId(match.id);
      run("start", async () => {
        resetReview();
        setFault("none");
        update(await api.post<CheckView>("/api/check/start", { profile_id: match.id }));
      });
    },
    [busy, check, profiles, lastCard, run, update],
  );
  const codes = live?.codes?.join("|") ?? "";
  useEffect(() => {
    onCodes(codes ? codes.split("|") : []);
  }, [codes, onCodes]);

  // What the pack should hold, laid out like the card.
  const layout =
    (check && check.phase !== "completed" ? check.layout : null) ??
    layouts?.find((l) => l.id === profile?.layout_id) ??
    null;
  const plan = useMemo(() => planFor(catalog, profile, compartments), [catalog, profile, compartments]);
  const doses = [...plan.values()].filter((items) => items.length).length;
  const tablets = [...plan.values()].reduce((sum, items) => sum + items.reduce((s, i) => s + i.quantity, 0), 0);
  const medicationCount = new Set([...plan.values()].flatMap((items) => items.map((i) => i.id))).size;

  const barcode =
    check && check.phase !== "completed" ? check.profile.barcode : profiles?.find((p) => p.id === profileId)?.barcode;
  const cardMatched =
    !!barcode && (analysed ? check!.card_codes.includes(barcode) : (live?.codes ?? []).includes(barcode));
  const reference = check && check.phase !== "completed" ? check.profile.reference : (profile?.reference ?? "No pack");

  useEffect(() => {
    const phase = !check || check.phase === "completed" ? "ready" : check.phase === "capturing" ? "capturing" : clock();
    onHeading({
      overline: `Pack check · ${phase}`,
      title: reference,
      mono: true,
      subtitle: layout?.display_name,
      badge: cardMatched ? "Header card matched" : undefined,
    });
  }, [check, reference, layout, cardMatched, onHeading]);
  useEffect(() => () => onHeading(null), [onHeading]);

  // ------------------------------------------------------------------ left column: camera or evidence

  const cameraColumn = (
    <div className="stack">
      {analysed && check.evidence ? (
        <EvidenceView
          check={check}
          selected={selected}
          footer={
            <div className="stack" style={{ gap: 10 }}>
              <div className="row" style={{ flexWrap: "nowrap", gap: 6 }}>
                <span
                  className="small faint ellipsis"
                  style={{ flex: 1 }}
                  title={
                    result?.capability === "identity" ? "Identity model" : "Count model: identity is checked by you"
                  }
                >
                  {typeof check.timings.total_ms === "number" &&
                    `Analysed in ${(check.timings.total_ms / 1000).toFixed(1)} s`}
                </span>
                <button className="compact" onClick={retake} disabled={busy !== null}>
                  <Icon name="rotate-ccw" size={16} /> Retake
                </button>
              </div>
              {(status?.expert.enabled || status?.local_advisor.enabled) && (
                <div className="row" style={{ flexWrap: "nowrap", gap: 6 }}>
                  <span className="small faint" style={{ flex: 1 }}>
                    Ask for
                  </span>
                  {status?.local_advisor.enabled && (
                    <button
                      className="ghost compact"
                      onClick={() => secondOpinion("local")}
                      disabled={busy !== null || !!check.reviewing}
                      title={`${status.local_advisor.model}, running on this computer`}
                    >
                      <Icon name="layers" size={16} /> Offline opinion
                    </button>
                  )}
                  {status?.expert.enabled && (
                    <>
                      <button
                        className="ghost compact"
                        onClick={() => secondOpinion("claude")}
                        disabled={busy !== null || !!check.reviewing}
                        title={status.expert.model}
                      >
                        <Icon name="sparkles" size={16} /> Claude opinion
                      </button>
                      <button className="ghost compact" onClick={explain} disabled={busy !== null}>
                        <Icon name="info" size={16} />
                        {busy === "explain" ? "Explaining…" : "Explain"}
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          }
        />
      ) : (
        <LiveView
          live={live}
          footer={
            <span className="small faint">
              {check?.phase === "capturing" ? (
                <>
                  Press <kbd>Space</kbd> to capture when the pack is steady
                </>
              ) : (
                "Show the pack's header card to start its check"
              )}
            </span>
          }
        />
      )}
      {!analysed && <CaptureQuality live={live} />}
    </div>
  );

  // ------------------------------------------------------------------ middle column: the pack

  const packColumn = (
    <div className="stack">
      <div className="card stack">
        <div className="card-head">
          <div className="stack" style={{ gap: 2 }}>
            <h2>{analysed ? "Your pack" : "Expected contents"}</h2>
            <span className="small faint">
              {profile
                ? `Profile ${profile.reference} · laid out like the card`
                : "Choose a profile to see its contents"}
            </span>
          </div>
          <span className="spacer" />
          <Icon name="layout-grid" className="faint" />
        </div>
        {layout ? (
          <PackGrid
            layout={layout}
            plan={plan}
            compartments={analysed ? compartments : undefined}
            selected={analysed ? selected : null}
            reviews={analysed ? reviews : undefined}
            onSelect={analysed ? setSelected : undefined}
          />
        ) : (
          <div className="muted">{profiles?.length === 0 ? "Create a profile first (Profiles)." : "Loading…"}</div>
        )}
        <PackLegend plan={plan} compartments={analysed ? compartments : undefined} />
      </div>
      {plan.size > 0 && (
        <div className="card">
          <PackMedications plan={plan} />
        </div>
      )}
    </div>
  );

  // ------------------------------------------------------------------ right column: before capture

  const qualityIssues = live?.quality_issues ?? [];
  const registrationIssues = live?.registration_issues ?? [];
  const checklist = (
    <div className="card stack">
      <div className="overline">Before capture</div>
      {barcode ? (
        cardMatched ? (
          <CheckItem state="ok" title="Header card matches the profile" detail={barcode} />
        ) : (live?.codes?.length ?? 0) > 0 ? (
          <CheckItem state="warn" title="A different header card is in view" detail={live!.codes.join(", ")} />
        ) : (
          <CheckItem state="idle" title="Header card not seen yet" detail={`Expected ${barcode}`} />
        )
      ) : (
        <CheckItem state="idle" title="Profile chosen by hand" detail="It has no header card" />
      )}
      {live?.usable || (live && registrationIssues.length === 0) ? (
        <CheckItem state="ok" title="Pack inside the frame" detail="Its outline is locked" />
      ) : (
        <CheckItem
          state={live ? "warn" : "idle"}
          title={
            live
              ? (ISSUE_TEXT[registrationIssues[0]] ?? registrationIssues[0] ?? "Pack not found")
              : "Waiting for the camera"
          }
        />
      )}
      {status && (
        <CheckItem
          state={status.layout.calibrated ? "ok" : "warn"}
          title={status.layout.calibrated ? "Layout calibrated" : "Layout not calibrated"}
          detail={status.layout.calibrated ? status.layout.name : "Calibrate it in Settings"}
        />
      )}
      {live && (
        <CheckItem
          state={qualityIssues.length ? "warn" : "ok"}
          title={
            qualityIssues.length ? qualityIssues.map((i) => ISSUE_TEXT[i] ?? i).join(" · ") : "Light and focus are good"
          }
          detail={qualityIssues.includes("glare") ? "Compartments under glare will go to review" : undefined}
        />
      )}
    </div>
  );

  const recent = (
    <div className="card stack">
      <div className="card-head">
        <span className="overline">Recent checks</span>
        <span className="spacer" />
        <span className="small faint">{audit?.total ?? 0} in the audit log</span>
      </div>
      {audit?.entries.length === 0 && <span className="small muted">No pack signed off yet.</span>}
      {audit?.entries.map((e) => (
        <div key={e.sequence} className="row" style={{ flexWrap: "nowrap" }}>
          <span className="mono" style={{ fontWeight: 500 }}>
            {e.reference}
          </span>
          <span className="small faint" style={{ flex: 1 }}>
            {clock(e.created_at)} · {e.pharmacist}
          </span>
          <span className={`badge ${e.decision === "released" ? "verified" : "mismatch"}`}>
            {e.decision === "released" ? "Released" : "Withheld"}
          </span>
        </div>
      ))}
    </div>
  );

  const starter = (
    <div className="card summary ready stack">
      <div className="summary-head">
        <span className="marker accent">
          <Icon name="scan-eye" size={22} />
        </span>
        <div className="stack" style={{ gap: 2 }}>
          <h2>Start a check</h2>
          <span className="small muted">Choose the profile, or show the pack's header card</span>
        </div>
      </div>
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
      <button className="primary big wide" disabled={!profileId || busy !== null} onClick={start}>
        <Icon name="scan-line" size={20} /> Start check
      </button>
    </div>
  );

  const capturing = check?.phase === "capturing" && (
    <div className="card summary ready stack">
      <div className="summary-head">
        <span className="marker accent">
          <Icon name="scan-eye" size={22} />
        </span>
        <div className="stack" style={{ gap: 2 }}>
          <h2>{live?.usable ? "Ready to check" : "Place the pack on the tray"}</h2>
          <span className="small muted">
            {live?.usable ? "Pack found" : "All of the pack must be in view"} · profile {check.profile.reference}
          </span>
        </div>
      </div>
      <button className="primary big wide" disabled={busy !== null} onClick={capture}>
        <Icon name="camera" size={20} />
        {busy === "capture" ? "Analysing…" : "Capture and check"}
        {busy !== "capture" && <kbd>Space</kbd>}
      </button>
      <div className="stats">
        <div>
          <div className="value">{doses}</div>
          <div className="label">Doses</div>
        </div>
        <div>
          <div className="value">{tablets}</div>
          <div className="label">Tablets</div>
        </div>
        <div>
          <div className="value">{medicationCount}</div>
          <div className="label">Medications</div>
        </div>
        <div>
          <div className="value">{check.layout.rows * check.layout.columns}</div>
          <div className="label">Compartments</div>
        </div>
      </div>
      <div className="row">
        {status?.demo && (
          <label style={{ flex: 1 }}>
            Demo: packing error to simulate
            <select value={fault} onChange={(e) => injectFault(e.target.value)}>
              {FAULTS.map((f) => (
                <option key={f}>{f}</option>
              ))}
            </select>
          </label>
        )}
        <button className="ghost" onClick={cancel} disabled={busy !== null} style={{ alignSelf: "flex-end" }}>
          <Icon name="x" size={16} /> Cancel
        </button>
      </div>
    </div>
  );

  const completed = check?.phase === "completed" && (
    <div className="card summary verified stack">
      <div className="summary-head">
        <span className="marker verified">
          <Icon name="shield-check" size={22} />
        </span>
        <div className="stack" style={{ gap: 2 }}>
          <h2>Pack {check.profile.reference} signed off</h2>
          <span className="small muted">Audit entry #{check.audit_sequence}</span>
        </div>
      </div>
      {check.learning && (
        <span className="small muted">
          The brain learned {check.learning.exemplars_added} new tablet images, recorded{" "}
          {check.learning.observations_recorded} identifications and queued {check.learning.tasks_queued} labelling
          task(s).
        </span>
      )}
    </div>
  );

  // ------------------------------------------------------------------ right column: result and sign-off

  const counted = compartments.filter(
    (c) => (c.expected_count ?? 0) > 0 || c.observed_count || c.status === "mismatch",
  );
  const tally = (s: string) => counted.filter((c) => c.status === s).length;
  const flagged = requiring.filter(
    (c) => c.status === "needsReview" || c.status === "mismatch" || c.spot_check || c.advisory?.verdict === "disagrees",
  );
  const inspected = requiring.filter((c) => reviews[key(c)]).length;

  const summary = analysed && result && (
    <div className={`card summary ${result.status} stack`}>
      <div className="summary-head">
        <span className={`marker ${result.status === "retakeRequired" ? "needsReview" : result.status}`}>
          <Icon name={STATUS_ICON[result.status] ?? "info"} size={22} />
        </span>
        <div className="stack" style={{ gap: 2 }}>
          <h2 className={`status-${result.status}`}>{PACK_STATUS[result.status] ?? result.status}</h2>
          <span className="small muted">
            {tally("mismatch") > 0 && `${tally("mismatch")} compartment(s) wrong · `}
            {flagged.length > 0 ? `${flagged.length} need your eyes` : PACK_NOTE[result.status]}
          </span>
        </div>
      </div>
      {counted.length > 0 && (
        <div className="distribution">
          {(["countMatched", "verified", "needsReview", "mismatch"] as const)
            .filter((s) => tally(s))
            .map((s) => (
              <span
                key={s}
                style={{
                  flex: tally(s),
                  background: `var(--${s === "countMatched" ? "count" : s === "needsReview" ? "review" : s})`,
                }}
              />
            ))}
        </div>
      )}
      <div className="stats">
        {(["countMatched", "needsReview", "mismatch", "verified"] as const).map((s) => (
          <div key={s}>
            <div className={`value ${tally(s) ? `status-${s}` : "faint"}`}>{tally(s)}</div>
            <div className="label">{STATUS_TEXT[s]}</div>
          </div>
        ))}
      </div>
      {result.pack_findings.map((f) => (
        <div key={f} className="error small">
          {f}
        </div>
      ))}
      {result.has_pack_findings && (
        <label className="inline small">
          <input type="checkbox" checked={acknowledged} onChange={(e) => setAcknowledged(e.target.checked)} />I have
          dealt with the pack findings above
        </label>
      )}
      <div className="divider" />
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <label style={{ width: 92 }}>
          Pharmacist
          <input className="mono" value={pharmacist} onChange={(e) => setPharmacist(e.target.value)} placeholder="MK" />
        </label>
        <label style={{ flex: 1 }}>
          Note
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Optional" />
        </label>
      </div>
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <button
          className="danger"
          style={{ flex: 1 }}
          disabled={busy !== null || !pharmacist.trim()}
          onClick={() => signOff("withheld")}
        >
          <Icon name="x" size={16} /> Withhold
        </button>
        <button
          className="primary"
          style={{ flex: 1.4 }}
          disabled={busy !== null || releaseBlockers.length > 0}
          onClick={() => signOff("released")}
        >
          <Icon name="shield-check" size={16} /> Release pack
        </button>
      </div>
      {check.reviewing && (
        <div className="insight small" style={{ alignItems: "center" }}>
          <Icon name="sparkles" size={16} />
          <span className="stack" style={{ gap: 6, flex: 1 }}>
            <span>
              Second opinion from {check.reviewing.advisor}: {check.reviewing.done} of {check.reviewing.total}
              {check.reviewing.stop && " · stopping"}
            </span>
            <span className="meter">
              <span
                style={{
                  width: `${Math.round((100 * check.reviewing.done) / Math.max(1, check.reviewing.total))}%`,
                  background: "var(--accent)",
                }}
              />
            </span>
          </span>
          <button className="ghost compact" onClick={stopReview} disabled={check.reviewing.stop || busy === "stop"}>
            Stop
          </button>
        </div>
      )}
      {releaseBlockers.length > 0 && <div className="small faint">To release: {releaseBlockers.join(" · ")}</div>}
      {check.review_errors.length > 0 && <div className="error small">{check.review_errors.join("; ")}</div>}
    </div>
  );

  const attention = analysed && (
    <div className="card stack" style={{ gap: 8 }}>
      <div className="card-head">
        <span className="overline">Needs your eyes</span>
        <span className="spacer" />
        <span className="small faint">
          {inspected} of {requiring.length} inspected
        </span>
      </div>
      <div className="list">
        {flagged.map((c) => {
          const review = reviews[key(c)];
          const state = review
            ? "verified"
            : c.status === "mismatch"
              ? "mismatch"
              : c.status === "needsReview"
                ? "needsReview"
                : "countMatched";
          return (
            <button
              key={key(c)}
              className={`list-item ${selected === key(c) ? "selected" : ""}`}
              onClick={() => setSelected(key(c))}
            >
              <span className={`marker ${state}`}>
                <Icon
                  name={
                    review
                      ? "check"
                      : c.spot_check && c.status === "countMatched"
                        ? "eye"
                        : (STATUS_ICON[c.status] ?? "info")
                  }
                  size={16}
                />
              </span>
              <span className="text">
                <span>{c.label}</span>
                <span className="small faint ellipsis">
                  {c.findings[0] ??
                    (c.spot_check
                      ? "Spot check: look at this one closely"
                      : `${c.observed_count ?? "?"} seen, ${c.expected_count ?? "?"} expected`)}
                </span>
              </span>
              <span className={`badge ${review ? (review === "unresolved" ? "mismatch" : "verified") : state}`}>
                {review
                  ? REVIEW_TEXT[review]
                  : c.spot_check && c.status === "countMatched"
                    ? "Spot check"
                    : STATUS_TEXT[c.status]}
              </span>
            </button>
          );
        })}
        {flagged.length === 0 && <span className="small muted">Nothing was flagged.</span>}
      </div>
      {easy.length > 0 && (
        <button onClick={confirmRemaining}>
          <Icon name="list-checks" size={16} /> I inspected the {easy.length} "Count OK" compartments
        </button>
      )}
    </div>
  );

  const detail =
    analysed && result && current ? (
      <div className="card stack">
        <div className="card-head">
          <h2>{current.label}</h2>
          <span className={`mono status-${current.status}`} style={{ fontSize: "1.07rem" }}>
            {current.observed_count ?? "?"} / {current.expected_count ?? "?"}
          </span>
          <span className="spacer" />
          <span className={`small status-${current.status}`}>{STATUS_TEXT[current.status]}</span>
        </div>
        <div className="compare">
          <div className="stack" style={{ gap: 8 }}>
            <span className="overline">
              <span className="dot" style={{ display: "inline-block", color: "var(--mismatch)", marginRight: 6 }} />
              Seen now
            </span>
            <div className="crops">
              {current.pills.map((p) => (
                <figure key={p.index}>
                  <img src={`/api/check/pills/${p.index}.jpg?r=${result.id}`} alt="tablet" />
                  <figcaption>
                    {p.size ?? "size unknown"}
                    {p.identity && <div>looks like {p.identity}</div>}
                    {p.fits && p.fits.length === 0 && (
                      <div className="status-needsReview">fits no known size/colour</div>
                    )}
                  </figcaption>
                </figure>
              ))}
              {current.pills.length === 0 && <span className="small muted">no confidently counted tablet</span>}
            </div>
          </div>
          <div className="stack" style={{ gap: 8 }}>
            <span className="overline">
              <span className="dot" style={{ display: "inline-block", color: "var(--verified)", marginRight: 6 }} />
              Confirmed reference
            </span>
            <div className="crops">
              {current.expected.map((e) => (
                <figure key={e.id}>
                  {e.reference ? (
                    <img src={`/api/medications/${encodeURIComponent(e.id)}/reference.jpg`} alt={e.name} />
                  ) : (
                    <div className="no-reference">no reference yet</div>
                  )}
                  <figcaption>
                    {e.quantity} × {e.name}
                    {e.range && <div className="faint">{e.range}</div>}
                  </figcaption>
                </figure>
              ))}
              {current.expected.length === 0 && <span className="small muted">nothing belongs here</span>}
            </div>
          </div>
        </div>
        {current.findings.length > 0 && (
          <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
            {current.findings.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        )}
        {current.advisory && (
          <div className="insight small">
            <Icon name="sparkles" size={16} />
            <span>
              Second opinion ({current.advisory.source}): {current.advisory.verdict}
              {current.advisory.observed_count !== null && `, counted ${current.advisory.observed_count}`}
              {current.advisory.note && ` — ${current.advisory.note}`}
            </span>
          </div>
        )}
        <div className="row" style={{ flexWrap: "nowrap" }}>
          <button
            className={reviews[key(current)] === "confirmedCorrect" ? "primary" : ""}
            onClick={() => setReview(current, "confirmedCorrect")}
          >
            <Icon name="circle-check" size={16} /> It is correct
          </button>
          <button
            className={reviews[key(current)] === "corrected" ? "primary" : ""}
            onClick={() => setReview(current, "corrected")}
          >
            <Icon name="pencil" size={16} /> I corrected it
          </button>
          <button
            className={reviews[key(current)] === "unresolved" ? "primary" : "ghost"}
            onClick={() => setReview(current, "unresolved")}
          >
            <Icon name="x" size={16} /> Unresolved
          </button>
        </div>
      </div>
    ) : (
      analysed && (
        <div className="card small muted row">
          <Icon name="info" size={16} /> Select a compartment in the pack or the list to inspect it.
        </div>
      )
    );

  const explanation = analysed && check.explanation && (
    <div className="card stack">
      <div className="insight">
        <Icon name="sparkles" size={16} />
        <span>{check.explanation.summary}</span>
      </div>
      <ol className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
        {check.explanation.steps.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
    </div>
  );

  return (
    <div className="stack">
      {error && <div className="error">{error}</div>}
      <div className="check-layout">
        {cameraColumn}
        {packColumn}
        <div className="stack">
          {analysed ? (
            <>
              {summary}
              {attention}
              {detail}
              {explanation}
            </>
          ) : (
            <>
              {completed}
              {check?.phase === "capturing" ? capturing : starter}
              {checklist}
              {recent}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
