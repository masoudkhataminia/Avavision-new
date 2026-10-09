// Shapes of the station API (src/avavision/station). Only the fields the interface uses.

export type Status = {
  version: string;
  station_id: string;
  demo: boolean;
  layout: { id: string; name: string; calibrated: boolean };
  camera: { error: string | null; frames: number };
  providers: string[];
  embedder: { id: string; providers: string[]; note: string | null };
  model: { id: string; version: string; capability: string };
  brain: { exemplar_count: number; known_medications: number; trusted_medications: string[] };
  expert: { enabled: boolean; model: string; fallbacks: boolean; key: boolean };
  audit: { entries: number; defect: { sequence: number; defect: string } | null };
  check: string | null;
};

export type Live = {
  usable: boolean;
  quality_issues: string[];
  registration_issues: string[];
  quad: [number, number][] | null;
  sharpness: number | null;
  locate_ms: number;
  codes: string[];
};

export type Index = { row: number; column: number };

export type Rect = { x: number; y: number; width: number; height: number };

export type Layout = {
  id: string;
  display_name: string;
  rows: number;
  columns: number;
  row_labels: string[];
  column_labels: string[];
  width_mm: number;
  height_mm: number;
  grid_region: Rect;
  border_band: number;
  is_calibrated: boolean;
};

export type Appearance = { colour: string | null; shape: string | null; imprint: string | null };
export type Medication = { id: string; name: string; strength: string; appearance: Appearance };
export type Catalog = { medications: Medication[]; references?: string[] };

export type ExpectedItem = { medication_id: string; quantity: number };
export type Expectation = { compartment: Index; items: ExpectedItem[] };
export type Profile = {
  id: string;
  reference: string;
  barcode?: string | null;
  layout_id: string;
  compartments: Expectation[];
  created_at: string;
};
export type ProfileSummary = {
  id: string;
  reference: string;
  barcode: string | null;
  layout_id: string;
  doses: number;
  created_at: string;
  issues: number;
};
export type ProfileIssue = { kind: string; compartment: Index | null; medication_id: string | null; detail: string | null };

export type CompartmentStatus = "verified" | "countMatched" | "needsReview" | "mismatch";

export type Advisory = { compartment: Index; verdict: "agrees" | "disagrees" | "unsure"; observed_count: number | null; note: string; source: string };

export type CompartmentView = {
  row: number;
  column: number;
  label: string;
  status: CompartmentStatus;
  findings: string[];
  expected: { id: string; name: string; quantity: number; reference: boolean; range: string | null }[];
  pills: { index: number; size: string | null; identity: string | null; fits: string[] | null }[];
  expected_count: number | null;
  observed_count: number | null;
  spot_check: boolean;
  requires_review: boolean;
  advisory: Advisory | null;
};

export type DetectionView = {
  box: [number, number, number, number];
  confidence: number;
  counted: boolean;
  identity: string | null;
  decision: string | null;
};

export type CheckView = {
  session_id: string;
  phase: "capturing" | "analyzed" | "completed";
  profile: { id: string; reference: string; barcode: string | null };
  card_codes: string[];
  layout: Layout;
  timings: Record<string, number | object>;
  evidence: boolean;
  explanation: { summary: string; steps: string[] } | null;
  advisories: Advisory[];
  review_errors: string[];
  expert_calls: { served_model: string; fallback_used: boolean; seconds: number }[];
  record_id: string | null;
  audit_sequence: number | null;
  learning: { exemplars_added: number; duplicates_skipped: number; observations_recorded: number; tasks_queued: number } | null;
  result: {
    id: string;
    status: string;
    pack_findings: string[];
    has_pack_findings: boolean;
    usable_frames: number;
    capability: string;
    trusted_medications: string[];
    compartments: CompartmentView[];
  } | null;
  detections?: DetectionView[];
};

export type ReviewOutcome = "confirmedCorrect" | "corrected" | "unresolved";

export type BrainView = {
  summary: { embedder_id: string; exemplar_count: number; known_medications: number; trusted_medications: string[] };
  policy: { calibrated: boolean; accept_similarity: number | null; minimum_margin: number; version: number };
  trust_policy: { minimum_predictions: number; minimum_precision_lower_bound: number; minimum_streak: number };
  calibration: { outcome: string; calibrated_at: string; precision_lower_bound: number; coverage: number; trials: number } | null;
  medications: {
    id: string;
    name: string;
    exemplars: number;
    groups: number;
    trust: "learning" | "trusted" | "suspended";
    progress: number;
    correct: number;
    false_identifications: number;
    streak: number;
  }[];
  tasks: {
    id: string;
    compartment: string;
    expected: Record<string, number>;
    priority: number;
    pills: { id: string; crop: string | null; suggestion: string | null }[];
  }[];
};

export type AuditSummary = {
  sequence: number;
  hash: string;
  created_at: string;
  reference: string;
  status: string;
  decision: string;
  pharmacist: string;
  evidence: string[];
};

export type ChartLine = {
  name: string;
  generic_name: string | null;
  strength: string | null;
  form: string | null;
  doses: { time: string; quantity: number }[];
  days: string[];
  schedule_note: string | null;
  appearance: string | null;
  in_pack: boolean;
  unclear: string[];
};

export type Draft = {
  profile: Profile;
  chart: { first_day: string | null; lines: ChartLine[]; warnings: string[] };
  matches: Record<string, string>;
  issues: { kind: string; line: number | null; detail: string }[];
  not_packed: number[];
  profile_issues: ProfileIssue[];
};

export type Settings = {
  station_id: string;
  layout_id: string;
  finder_mode: "markers" | "outline";
  camera_index: number;
  camera_width: number;
  camera_height: number;
  camera_exposure: number | null;
  keep_evidence_images: boolean;
  expert_enabled: boolean;
  language: string;
  expert: {
    model: string;
    fallbacks: boolean;
    chart_effort: string;
    explain_effort: string;
    review_effort: string;
    max_tokens: number;
    timeout_seconds: number;
  };
};
