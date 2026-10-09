import { useState } from "react";
import { api } from "../api";
import { useAction, useData } from "../hooks";
import type { PhoneView } from "../types";

/** Pairing and status of the iPhone camera (station.phone): the phone streams over the local Wi-Fi. */
export function PhonePanel() {
  const { data: phone, error: loadError, reload } = useData<PhoneView>("/api/phone", 1500);
  const [pairing, setPairing] = useState(Date.now());
  const { busy, error, run } = useAction();

  const pairAgain = () =>
    run("pair", async () => {
      await api.post("/api/phone/pair");
      setPairing(Date.now());
      reload();
    });

  if (!phone) return <div className="small muted">{loadError ?? "Loading…"}</div>;
  const state = phone.connected
    ? `Connected · ${phone.resolution?.join("×")} · ${phone.fps} pictures/s · ${phone.device ?? "iPhone"}`
    : phone.seconds_since_frame !== null
      ? `Not connected (last picture ${phone.seconds_since_frame} s ago)`
      : "Not connected yet";

  return (
    <div className="stack">
      <div className={phone.connected ? "notice" : "error"}>{state}</div>
      {phone.problem && <div className="error">{phone.problem}</div>}
      {error && <div className="error">{error}</div>}
      <div className="row" style={{ alignItems: "flex-start" }}>
        <div className="stack" style={{ flex: 1 }}>
          <b>1. Once per iPhone: trust the station</b>
          <span className="small">
            Scan with the iPhone's Camera app, open in Safari, and follow the steps on the phone (download, install,
            then switch on full trust).
          </span>
          {phone.setup_url && <img src="/api/phone/setup.png" alt="Setup QR code" style={{ width: 180 }} />}
          <span className="small muted">{phone.setup_url}</span>
        </div>
        <div className="stack" style={{ flex: 1 }}>
          <b>2. Camera (pairs this station)</b>
          <span className="small">
            Scan, open in Safari, tap Start camera. Keep the phone on its stand, pointing straight down, plugged in.
          </span>
          {phone.address && <img src={`/api/phone/camera.png?p=${pairing}`} alt="Camera QR code" style={{ width: 180 }} />}
          <button onClick={pairAgain} disabled={busy !== null}>
            Pair again (disconnects the current iPhone)
          </button>
        </div>
      </div>
      <span className="small muted">
        The iPhone and this computer must be on the same Wi-Fi. If Windows asks, allow AvaVision on private networks.
        Certificate fingerprint (must match the phone's setup page): <code>{phone.fingerprint}</code>
      </span>
    </div>
  );
}
