import { useEffect, useState } from "react";
import { useData } from "./hooks";
import { AuditPage } from "./pages/AuditPage";
import { BrainPage } from "./pages/BrainPage";
import { CatalogPage } from "./pages/CatalogPage";
import { CheckPage } from "./pages/CheckPage";
import { ProfilesPage } from "./pages/ProfilesPage";
import { SettingsPage } from "./pages/SettingsPage";
import type { Status } from "./types";

const PAGES = ["check", "profiles", "catalog", "brain", "audit", "settings"] as const;
type Page = (typeof PAGES)[number];

function currentPage(): Page {
  const hash = window.location.hash.replace("#", "") as Page;
  return PAGES.includes(hash) ? hash : "check";
}

function acceleration(providers: string[]): string {
  if (providers.includes("TensorrtExecutionProvider")) return "TensorRT";
  if (providers.includes("CUDAExecutionProvider")) return "CUDA";
  if (providers.includes("DmlExecutionProvider")) return "DirectML";
  if (providers.includes("QNNExecutionProvider")) return "NPU";
  if (providers.includes("builtin")) return "classic";
  return "CPU";
}

export function App() {
  const [page, setPage] = useState<Page>(currentPage);
  const { data: status, reload } = useData<Status>("/api/status", 5000);

  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (next: Page) => {
    window.location.hash = next;
    setPage(next);
  };

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">AvaVision</span>
        <nav className="nav">
          {PAGES.map((p) => (
            <button key={p} className={p === page ? "active" : ""} onClick={() => go(p)}>
              {p[0].toUpperCase() + p.slice(1)}
            </button>
          ))}
        </nav>
        {status && (
          <div className="chips">
            {status.demo && <span className="chip warn">Demo camera</span>}
            {status.camera.error && <span className="chip bad">Camera: {status.camera.error}</span>}
            {!status.layout.calibrated && <span className="chip warn">Layout not calibrated</span>}
            <span className="chip">Embedder: {acceleration(status.embedder.providers)}</span>
            <span className="chip">
              Brain: {status.brain.exemplar_count} pills · {status.brain.trusted_medications.length} trusted
            </span>
            <span className={`chip ${status.expert.enabled && !status.expert.key ? "warn" : ""}`}>
              Expert: {status.expert.enabled ? (status.expert.key ? "ready" : "no key") : "off"}
            </span>
            {status.audit.defect && <span className="chip bad">AUDIT CHAIN BROKEN</span>}
          </div>
        )}
      </header>
      <main>
        {page === "check" && <CheckPage status={status} onChange={reload} />}
        {page === "profiles" && <ProfilesPage />}
        {page === "catalog" && <CatalogPage />}
        {page === "brain" && <BrainPage onChange={reload} />}
        {page === "audit" && <AuditPage onChange={reload} />}
        {page === "settings" && <SettingsPage status={status} onChange={reload} />}
      </main>
    </div>
  );
}
