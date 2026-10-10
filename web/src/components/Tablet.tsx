import type { CSSProperties } from "react";
import type { Appearance } from "../types";

// Catalog colours are free text ("white", "light blue", "orange-red"); the first known word decides.
const COLOURS: [string, string][] = [
  ["white", "#f3f1ec"],
  ["cream", "#efe6cf"],
  ["ivory", "#efe9d8"],
  ["beige", "#e3d3b4"],
  ["yellow", "#f2cf4a"],
  ["orange", "#f39a3d"],
  ["peach", "#f6bf98"],
  ["pink", "#f2a2bf"],
  ["red", "#d9534a"],
  ["maroon", "#8e3040"],
  ["brown", "#93613f"],
  ["green", "#79c48a"],
  ["teal", "#4fb6ad"],
  ["blue", "#69a6ea"],
  ["purple", "#a688e3"],
  ["lilac", "#c3a8ee"],
  ["violet", "#a688e3"],
  ["grey", "#b6bac1"],
  ["gray", "#b6bac1"],
  ["black", "#3a3c41"],
];

const UNKNOWN = "#c9ccd2";

export function pillColour(colour: string | null | undefined): string {
  const text = (colour ?? "").toLowerCase();
  let best: [number, string] = [Infinity, UNKNOWN];
  for (const [word, value] of COLOURS) {
    const at = text.indexOf(word);
    if (at >= 0 && at < best[0]) best = [at, value];
  }
  return best[1];
}

export function pillShape(shape: string | null | undefined): "round" | "oval" | "capsule" {
  const text = (shape ?? "").toLowerCase();
  if (text.includes("capsule")) return "capsule";
  if (/oval|oblong|ellip|almond|caplet/.test(text)) return "oval";
  return "round";
}

export function describeAppearance(appearance: Appearance | undefined): string {
  if (!appearance) return "";
  return [appearance.colour, appearance.shape].filter(Boolean).join(" ");
}

/** A small 3D glyph of a tablet from its catalog appearance (colour and shape). */
export function Tablet({ appearance }: { appearance?: Appearance }) {
  return (
    <span className="tablet-slot">
      <span
        className={`tablet ${pillShape(appearance?.shape)}`}
        style={{ "--pill": pillColour(appearance?.colour) } as CSSProperties}
      />
    </span>
  );
}
