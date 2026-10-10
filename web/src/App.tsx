import { useCallback, useEffect, useState } from "react";
import { Icon, type IconName } from "./components/Icon";
import { useData } from "./hooks";
import { AuditPage } from "./pages/AuditPage";
import { BrainPage } from "./pages/BrainPage";
import { CatalogPage } from "./pages/CatalogPage";
import { CheckPage } from "./pages/CheckPage";
import { ProfilesPage } from "./pages/ProfilesPage";
import { SettingsPage } from "./pages/SettingsPage";
import type { Status } from "./types";

const PAGES: { id: string; label: string; icon: IconName; subtitle: string }[] = [
  { id: "check", label: "Check", icon: "scan-line", subtitle: "Verify a pack" },
  { id: "profiles", label: "Profiles", icon: "layout-grid", subtitle: "What belongs in each compartment" },
  { id: "catalog", label: "Catalog", icon: "pill", subtitle: "Medications and how they look" },
  { id: "brain", label: "Brain", icon: "brain", subtitle: "What the station has learned" },
  { id: "audit", label: "Audit", icon: "shield-check", subtitle: "Signed-off packs, tamper-evident" },
  { id: "settings", label: "Settings", icon: "settings-2", subtitle: "Station, camera and expert" },
];
type Page = (typeof PAGES)[number]["id"];

/** The top bar's title; the check page replaces it with the pack being checked. */
export type Heading = { overline: string; title: string; subtitle?: string; badge?: string; mono?: boolean };

type Theme = "dark" | "light";

function currentPage(): Page {
  const hash = window.location.hash.replace("#", "");
  return PAGES.some((p) => p.id === hash) ? hash : "check";
}

function storedValue(name: string): string | null {
  try {
    return localStorage.getItem(name);
  } catch {
    return null;
  }
}

function initialTheme(): Theme {
  const stored = storedValue("avavision.theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function acceleration(providers: string[]): string {
  if (providers.includes("TensorrtExecutionProvider")) return "TensorRT";
  if (providers.includes("CUDAExecutionProvider")) return "CUDA";
  if (providers.includes("DmlExecutionProvider")) return "DirectML";
  if (providers.includes("QNNExecutionProvider")) return "NPU";
  if (providers.includes("builtin")) return "classic";
  return "CPU";
}

const CAMERA_NAME = { iphone: "iPhone camera", usb: "USB camera", demo: "Simulated camera" } as const;

export function App() {
  const [page, setPage] = useState<Page>(currentPage);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [heading, setHeading] = useState<Heading | null>(null);
  const { data: status, reload } = useData<Status>("/api/status", 5000);

  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("avavision.theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  const go = (next: Page) => {
    window.location.hash = next;
    setPage(next);
  };
  const onHeading = useCallback((h: Heading | null) => setHeading(h), []);

  const info = PAGES.find((p) => p.id === page)!;
  const shown: Heading =
    page === "check" && heading ? heading : { overline: "AvaVision", title: info.label, subtitle: info.subtitle };
  const cameraOk = status ? !status.camera.error : false;
  const initials = (storedValue("avavision.pharmacist") ?? "").trim().slice(0, 3).toUpperCase();

  return (
    <div className="app">
      <aside className="rail">
        <div className="logo">
          <Icon name="scan-eye" size={22} />
        </div>
        {PAGES.map((p) => (
          <button key={p.id} className={`nav-item ${p.id === page ? "active" : ""}`} onClick={() => go(p.id)}>
            <Icon name={p.icon} size={20} />
            {p.label}
          </button>
        ))}
        <span className="rail-spacer" />
        <button
          className="nav-item"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Switch to the light look" : "Switch to the dark look"}
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size={20} />
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        <div className="rail-status" title={status?.camera.error ?? "Camera delivering pictures"}>
          <span className={`tile ${cameraOk ? "ok-tile" : "bad-tile"}`}>
            <Icon name={status?.camera.source === "iphone" ? "smartphone" : "camera"} size={20} />
          </span>
          <span className={cameraOk ? "status-verified" : "status-mismatch"}>{cameraOk ? "Live" : "No camera"}</span>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="stack" style={{ gap: 2, minWidth: 0 }}>
            <span className="overline">{shown.overline}</span>
            <div className="title-row">
              <h1 className={shown.mono ? "mono" : ""}>{shown.title}</h1>
              {shown.subtitle && <span className="muted">{shown.subtitle}</span>}
              {shown.badge && <span className="badge verified">{shown.badge}</span>}
            </div>
          </div>
          <span className="spacer" />
          {status && (
            <div className="chips">
              {status.audit.defect && (
                <span className="chip bad">
                  <Icon name="triangle-alert" size={16} /> Audit chain broken
                </span>
              )}
              <span
                className={`chip ${status.camera.error ? "bad" : status.demo ? "warn" : ""}`}
                title={status.camera.error ?? ""}
              >
                <Icon name={status.camera.source === "iphone" ? "smartphone" : "camera"} size={16} />
                {status.camera.error ? `Camera: ${status.camera.error}` : CAMERA_NAME[status.camera.source]}
                {!status.camera.error && <span className="dot ok" />}
              </span>
              <span className={`chip ${status.layout.calibrated ? "" : "warn"}`}>
                <Icon name="layout-grid" size={16} />
                {status.layout.calibrated ? "Layout calibrated" : "Layout not calibrated"}
                {status.layout.calibrated && <span className="dot ok" />}
              </span>
              <span className="chip" title={`Embedder on ${acceleration(status.embedder.providers)}`}>
                <Icon name="brain" size={16} />
                Brain · {status.brain.exemplar_count} tablets · {status.brain.trusted_medications.length} trusted
              </span>
              {status.expert.enabled && (
                <span className={`chip ${status.expert.key ? "" : "warn"}`}>
                  <Icon name="sparkles" size={16} />
                  {status.expert.key ? "Expert ready" : "Expert: no key"}
                </span>
              )}
            </div>
          )}
          <span className="avatar" title="Pharmacist on duty">
            {initials || <Icon name="pencil" size={16} />}
          </span>
        </header>
        <main className="content">
          {page === "check" && <CheckPage status={status} onChange={reload} onHeading={onHeading} />}
          {page === "profiles" && <ProfilesPage />}
          {page === "catalog" && <CatalogPage />}
          {page === "brain" && <BrainPage onChange={reload} />}
          {page === "audit" && <AuditPage onChange={reload} />}
          {page === "settings" && <SettingsPage status={status} onChange={reload} />}
        </main>
      </div>
    </div>
  );
}
