import { useEffect, useState } from "react";
import { api } from "../api";
import { PhonePanel } from "../components/PhonePanel";
import { useAction, useData } from "../hooks";
import type { Layout, LocalAdvisorSettings, LocalAdvisorStatus, Settings, Status } from "../types";

const EFFORTS = ["low", "medium", "high", "xhigh", "max"];

export function SettingsPage({ status, onChange }: { status: Status | null; onChange: () => void }) {
  const { data: loaded, reload } = useData<Settings>("/api/settings");
  const { data: layouts, reload: reloadLayouts } = useData<Layout[]>("/api/layouts");
  const [settingUp, setSettingUp] = useState(false);
  const { data: local, reload: reloadLocal } = useData<LocalAdvisorStatus>(
    "/api/advisor/local",
    settingUp ? 1000 : 5000,
  );
  const [settings, setSettings] = useState<Settings | null>(null);
  const [layout, setLayout] = useState<Layout | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [preview, setPreview] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const { busy, error, run } = useAction();

  useEffect(() => setSettings(loaded), [loaded]);
  // While the offline model is being set up, follow it closely; when it is done its setting is on.
  const setupStage = local?.setup?.stage;
  useEffect(() => {
    const active = setupStage === "installing" || setupStage === "starting" || setupStage === "downloading";
    setSettingUp(active);
    if (setupStage === "done") {
      reload();
      onChange();
    }
  }, [setupStage, reload, onChange]);
  useEffect(() => {
    if (layouts && settings) setLayout(layouts.find((l) => l.id === settings.layout_id) ?? null);
  }, [layouts, settings?.layout_id]);

  if (!settings) return <div className="panel muted">Loading…</div>;

  const set = <K extends keyof Settings>(field: K, value: Settings[K]) => setSettings({ ...settings, [field]: value });
  const setExpert = (field: keyof Settings["expert"], value: string | number | boolean) =>
    setSettings({ ...settings, expert: { ...settings.expert, [field]: value } });
  const setLocal = (field: keyof LocalAdvisorSettings, value: string | boolean) =>
    setSettings({ ...settings, local_advisor: { ...settings.local_advisor, [field]: value } });

  const save = () =>
    run("save", async () => {
      await api.put("/api/settings", settings);
      setMessage("Settings saved. Camera changes apply after restarting the station.");
      reload();
      onChange();
    });

  const saveKey = (key: string | null) =>
    run("key", async () => {
      const result = await api.put<{ stored_in: string }>("/api/settings/api-key", { key });
      setApiKey("");
      setMessage(key ? `API key stored in the ${result.stored_in}.` : "API key removed.");
      onChange();
    });

  const setUpLocal = () =>
    run("local", async () => {
      await api.post("/api/advisor/local/setup");
      setSettingUp(true);
      reloadLocal();
    });

  const saveLayout = (calibrated: boolean) =>
    layout &&
    run("layout", async () => {
      await api.put(`/api/layouts/${layout.id}`, { ...layout, is_calibrated: calibrated });
      setMessage(calibrated ? "Layout saved as calibrated." : "Layout saved (not calibrated).");
      reloadLayouts();
      onChange();
    });

  const grid = (field: "x" | "y" | "width" | "height", value: number) =>
    layout && setLayout({ ...layout, grid_region: { ...layout.grid_region, [field]: value }, is_calibrated: false });

  return (
    <div className="stack">
      {error && <div className="error">{error}</div>}
      {message && <div className="notice">{message}</div>}
      <div className="split">
        <div className="panel stack">
          <h2>Station</h2>
          <div className="row">
            <label>
              Station id
              <input value={settings.station_id} onChange={(e) => set("station_id", e.target.value)} />
            </label>
            <label>
              Pack layout
              <select
                value={settings.layout_id}
                disabled={status?.demo}
                onChange={(e) => set("layout_id", e.target.value)}
              >
                {layouts?.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.display_name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Finding the pack
              <select
                value={settings.finder_mode}
                onChange={(e) => set("finder_mode", e.target.value as Settings["finder_mode"])}
              >
                <option value="markers">Tray markers (recommended)</option>
                <option value="outline">Pack outline</option>
              </select>
            </label>
          </div>
          <h3>Camera</h3>
          <label>
            Camera
            <select
              value={settings.camera_source}
              disabled={status?.demo}
              onChange={(e) => set("camera_source", e.target.value as Settings["camera_source"])}
            >
              <option value="usb">Camera attached to this computer</option>
              <option value="iphone">iPhone over Wi-Fi (Safari, no app)</option>
            </select>
          </label>
          {settings.camera_source !== status?.camera.source && !status?.demo && (
            <div className="small muted">Save, then restart the station to switch the camera.</div>
          )}
          {status?.camera.source === "iphone" && <PhonePanel />}
          <div className="row" hidden={settings.camera_source !== "usb"}>
            <label>
              Camera number
              <input type="number" min={0} value={settings.camera_index} onChange={(e) => set("camera_index", Number(e.target.value))} />
            </label>
            <label>
              Width
              <input type="number" value={settings.camera_width} onChange={(e) => set("camera_width", Number(e.target.value))} />
            </label>
            <label>
              Height
              <input type="number" value={settings.camera_height} onChange={(e) => set("camera_height", Number(e.target.value))} />
            </label>
            <label>
              Fixed exposure (blank = auto)
              <input
                value={settings.camera_exposure ?? ""}
                onChange={(e) => set("camera_exposure", e.target.value === "" ? null : Number(e.target.value))}
              />
            </label>
          </div>
          <label className="inline">
            <input
              type="checkbox"
              checked={settings.keep_evidence_images}
              onChange={(e) => set("keep_evidence_images", e.target.checked)}
            />
            Keep an evidence photo of every check
          </label>
          <h3>Expert (Claude)</h3>
          <label className="inline">
            <input type="checkbox" checked={settings.expert_enabled} onChange={(e) => set("expert_enabled", e.target.checked)} />
            Use the expert for chart import, explanations and second opinions (never to accept a pack)
          </label>
          <div className="row">
            <label>
              Explanation language
              <select value={settings.language} onChange={(e) => set("language", e.target.value)}>
                <option value="en">English</option>
                <option value="fa">Persian</option>
              </select>
            </label>
            <label>
              Model
              <input value={settings.expert.model} onChange={(e) => setExpert("model", e.target.value)} />
            </label>
            {(["chart_effort", "review_effort", "explain_effort"] as const).map((field) => (
              <label key={field}>
                {field.replace("_effort", "")} effort
                <select value={settings.expert[field]} onChange={(e) => setExpert(field, e.target.value)}>
                  {EFFORTS.map((effort) => (
                    <option key={effort}>{effort}</option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <label className="inline">
            <input
              type="checkbox"
              checked={settings.expert.fallbacks}
              onChange={(e) => setExpert("fallbacks", e.target.checked)}
            />
            Retry declined requests on Anthropic's recommended fallback model
          </label>
          <div className="row">
            <label style={{ flex: 1 }}>
              API key {status?.expert.key ? "(set)" : "(not set)"}
              <input type="password" value={apiKey} placeholder="sk-ant-…" onChange={(e) => setApiKey(e.target.value)} />
            </label>
            <button disabled={!apiKey || busy !== null} onClick={() => saveKey(apiKey)}>
              Store key
            </button>
            <button disabled={busy !== null} onClick={() => saveKey(null)}>
              Remove
            </button>
          </div>
          <h3>Offline model (second opinions without internet)</h3>
          <p className="small muted" style={{ margin: 0 }}>
            A vision model running in Ollama on this computer or the pharmacy network, Qwen3-VL by default. Like the
            expert it can only send compartments to review, never accept them. Install Ollama, then run{" "}
            <kbd>ollama pull {settings.local_advisor.model}</kbd>.
          </p>
          <label className="inline">
            <input
              type="checkbox"
              checked={settings.local_advisor.enabled}
              onChange={(e) => setLocal("enabled", e.target.checked)}
            />
            Use the offline model for second opinions
          </label>
          <div className="row">
            <label style={{ flex: 1 }}>
              Ollama address
              <input value={settings.local_advisor.url} onChange={(e) => setLocal("url", e.target.value)} />
            </label>
            <label style={{ flex: 1 }}>
              Model
              <input value={settings.local_advisor.model} onChange={(e) => setLocal("model", e.target.value)} />
            </label>
          </div>
          <label className="inline">
            <input
              type="checkbox"
              checked={settings.local_advisor.automatic}
              onChange={(e) => setLocal("automatic", e.target.checked)}
            />
            Ask it about every accepted compartment right after capture (needs a graphics card)
          </label>
          {local && <LocalModelState local={local} model={settings.local_advisor.model} />}
          {local && !settingUp && (!local.running || !local.model_installed) && (
            <div className="row">
              <button disabled={busy !== null} onClick={setUpLocal}>
                Install and set up automatically
              </button>
              <span className="small muted">Installs Ollama if needed and downloads the model (about 6 GB).</span>
            </div>
          )}
          <div className="row">
            <button className="primary" disabled={busy !== null} onClick={save}>
              Save settings
            </button>
            <a href="/api/markers.png" download="avavision-markers.png">
              Download tray markers
            </a>
          </div>
        </div>

        {layout && (
          <div className="panel stack">
            <h2>Calibrate {layout.display_name}</h2>
            <p className="small muted">
              Place an empty pack on the tray. Adjust the grid until every box sits on one compartment, then save it as
              calibrated. Until then no compartment can be accepted automatically.
            </p>
            <div className="row">
              {(["x", "y", "width", "height"] as const).map((field) => (
                <label key={field}>
                  {field}
                  <input
                    type="number"
                    step={0.002}
                    min={0}
                    max={1}
                    value={layout.grid_region[field]}
                    onChange={(e) => grid(field, Number(e.target.value))}
                  />
                </label>
              ))}
            </div>
            <div className="row">
              <button onClick={() => saveLayout(false)} disabled={busy !== null}>
                Save grid
              </button>
              <button onClick={() => setPreview(Date.now())}>Show on camera</button>
              <button className="primary" onClick={() => saveLayout(true)} disabled={busy !== null}>
                Save as calibrated
              </button>
            </div>
            {preview > 0 && (
              <div className="media">
                <img src={`/api/camera/rectified.jpg?t=${preview}`} alt="Calibration preview" />
              </div>
            )}
            <p className="small muted">The preview uses the saved grid: save first, then show it on the camera.</p>
          </div>
        )}
      </div>
    </div>
  );
}

const STAGE_TEXT: Record<string, string> = {
  installing: "Installing Ollama",
  starting: "Starting Ollama",
  downloading: "Downloading the model",
};

function LocalModelState({ local, model }: { local: LocalAdvisorStatus; model: string }) {
  const setup = local.setup;
  if (setup && STAGE_TEXT[setup.stage]) {
    const share = setup.total ? setup.completed / setup.total : 0;
    return (
      <div className="notice small stack" style={{ gap: 6 }}>
        <span>
          {STAGE_TEXT[setup.stage]}
          {setup.total > 0 &&
            ` · ${(setup.completed / 1e9).toFixed(1)} of ${(setup.total / 1e9).toFixed(1)} GB (${Math.round(share * 100)}%)`}
        </span>
        <div className="meter">
          <span style={{ width: `${Math.round(share * 100)}%`, background: "var(--accent)" }} />
        </div>
      </div>
    );
  }
  if (setup?.stage === "failed") return <div className="error small">{setup.error}</div>;
  if (local.running && local.model_installed) {
    const where =
      local.graphics_share === null
        ? ""
        : local.graphics_share >= 0.9
          ? " It runs on the graphics card."
          : local.graphics_share > 0
            ? ` Only ${Math.round(local.graphics_share * 100)}% fits on the graphics card: answers will be slow.`
            : " It runs on the processor only: answers take a minute or more.";
    return (
      <div className="notice small">
        Ollama is running and {model} is installed.{where}
        {!local.enabled && " Turn it on above and save."}
      </div>
    );
  }
  return <div className="error small">{local.problem ?? "The offline model is not ready."}</div>;
}
