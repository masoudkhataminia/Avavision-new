import { useState } from "react";
import { api } from "../api";
import { useAction, useData } from "../hooks";
import type { Catalog, Medication } from "../types";

const blank: Medication = { id: "", name: "", strength: "", appearance: { colour: null, shape: null, imprint: null } };

function slug(name: string, strength: string): string {
  return `${name} ${strength}`
    .toLowerCase()
    .replace(/[^a-z0-9.]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

export function CatalogPage() {
  const { data: catalog, reload } = useData<Catalog>("/api/catalog");
  const [editing, setEditing] = useState<Medication | null>(null);
  const [isNew, setIsNew] = useState(false);
  const { busy, error, run } = useAction();

  const save = () =>
    editing &&
    run("save", async () => {
      const medication = { ...editing, id: isNew ? slug(editing.name, editing.strength) : editing.id };
      await api.put(`/api/catalog/medications/${encodeURIComponent(medication.id)}`, medication);
      setEditing(null);
      reload();
    });

  const remove = (m: Medication) =>
    window.confirm(`Remove ${m.name} ${m.strength} from the catalog? Profiles using it will show an issue.`) &&
    run("delete", async () => {
      await api.delete(`/api/catalog/medications/${encodeURIComponent(m.id)}`);
      reload();
    });

  const appearance = (field: "colour" | "shape" | "imprint", value: string) =>
    editing && setEditing({ ...editing, appearance: { ...editing.appearance, [field]: value || null } });

  return (
    <div className="split">
      <div className="panel stack">
        <div className="row">
          <h2 style={{ marginRight: "auto" }}>Medication catalog</h2>
          <button
            className="primary"
            onClick={() => {
              setEditing(structuredClone(blank));
              setIsNew(true);
            }}
          >
            Add medication
          </button>
        </div>
        {error && <div className="error">{error}</div>}
        <table>
          <thead>
            <tr>
              <th>Medication</th>
              <th>Appearance</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {catalog?.medications.map((m) => (
              <tr key={m.id}>
                <td>
                  {m.name} {m.strength}
                  <div className="small muted">{m.id}</div>
                </td>
                <td className="small">
                  {[m.appearance.colour, m.appearance.shape, m.appearance.imprint].filter(Boolean).join(", ")}
                </td>
                <td className="row">
                  <button
                    onClick={() => {
                      setEditing(structuredClone(m));
                      setIsNew(false);
                    }}
                  >
                    Edit
                  </button>
                  <button className="danger" onClick={() => remove(m)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {editing && (
        <div className="panel stack">
          <h2>{isNew ? "New medication" : `${editing.name} ${editing.strength}`}</h2>
          <label>
            Name
            <input value={editing.name} onChange={(e) => setEditing({ ...editing, name: e.target.value })} />
          </label>
          <label>
            Strength
            <input value={editing.strength} onChange={(e) => setEditing({ ...editing, strength: e.target.value })} />
          </label>
          <label>
            Colour
            <input value={editing.appearance.colour ?? ""} onChange={(e) => appearance("colour", e.target.value)} />
          </label>
          <label>
            Shape
            <input value={editing.appearance.shape ?? ""} onChange={(e) => appearance("shape", e.target.value)} />
          </label>
          <label>
            Imprint
            <input value={editing.appearance.imprint ?? ""} onChange={(e) => appearance("imprint", e.target.value)} />
          </label>
          {isNew && <div className="small muted">Id: {slug(editing.name, editing.strength) || "—"}</div>}
          <div className="row">
            <button className="primary" disabled={!editing.name.trim() || busy !== null} onClick={save}>
              Save
            </button>
            <button onClick={() => setEditing(null)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}
