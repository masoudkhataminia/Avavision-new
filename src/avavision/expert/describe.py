"""Plain-English descriptions of a verification result. Deterministic, so they work offline and are what the
language model is given when it explains a result."""

from __future__ import annotations

from ..core.engine import (
    CompartmentStatus,
    Finding,
    FindingKind,
    PackFinding,
    PackFindingKind,
    VerificationResult,
)
from ..core.models import Catalog, MedicationID, PackLayout


def _name(medication: MedicationID | None, catalog: Catalog) -> str:
    if medication is None:
        return "a dose"
    found = catalog.get(medication)
    return found.display_name if found else medication


def describe_finding(f: Finding, catalog: Catalog) -> str:
    med = _name(f.medication_id, catalog)
    counts = f" (expected {f.expected}, observed {f.observed})" if f.expected is not None else ""
    texts = {
        FindingKind.MISSING: f"fewer doses than expected{counts}",
        FindingKind.EXTRA: f"more doses than expected{counts}",
        FindingKind.WRONG_QUANTITY: f"wrong quantity of {med}{counts}",
        FindingKind.UNEXPECTED_MEDICATION: f"{med} is not expected here (observed {f.observed})",
        FindingKind.BROKEN_DOSE: "a broken or partial tablet",
        FindingKind.FOREIGN_OBJECT: "a foreign object",
        FindingKind.LOW_CONFIDENCE_OBJECT: "an object that could not be recognised with confidence",
        FindingKind.OBJECT_ON_BORDER: "an object on the border between compartments",
        FindingKind.UNSTABLE_ACROSS_FRAMES: "the photos disagree about this compartment",
        FindingKind.IDENTITY_NOT_VERIFIED: "count matches; identity not checked automatically",
        FindingKind.SUSPECTED_MEDICATION: f"a tablet that looks like {med}",
        FindingKind.UNRECOGNISED_PILL: "a tablet the brain does not recognise",
        FindingKind.CONFLICTING_IDENTITY: "the photos disagree about which medication this is",
        FindingKind.ADVISOR_DISAGREES: "the second opinion saw something different",
        FindingKind.NO_EXPECTATION: "the profile does not say what belongs here",
        FindingKind.LAYOUT_UNCALIBRATED: "the pack layout is not calibrated",
        FindingKind.NOT_EVALUATED: "not evaluated",
    }
    text = texts[f.kind]
    if f.count and f.count > 1:
        text += f" ×{f.count}"
    return text


def describe_pack_finding(f: PackFinding) -> str:
    texts = {
        PackFindingKind.INSUFFICIENT_USABLE_FRAMES: f"only {f.usable} of {f.required} usable photos",
        PackFindingKind.CAPTURE_ISSUES: "photo problems: " + ", ".join(i.value for i in f.capture_issues),
        PackFindingKind.REGISTRATION_ISSUES: "pack position problems: "
        + ", ".join(i.value for i in f.registration_issues),
        PackFindingKind.OBJECTS_OUTSIDE_COMPARTMENTS: f"{f.count} object(s) outside the compartments",
        PackFindingKind.MODEL_UNAVAILABLE: "no detection model is available",
        PackFindingKind.PROFILE_LAYOUT_MISMATCH: "the profile belongs to a different pack layout",
    }
    return texts[f.kind]


def describe_result(result: VerificationResult, layout: PackLayout, catalog: Catalog) -> list[str]:
    lines = [f"Pack status: {result.status.value}"]
    lines += [f"Pack: {describe_pack_finding(f)}" for f in result.pack_findings]
    spot = set(result.spot_checks)
    for verdict in result.compartments:
        if verdict.status == CompartmentStatus.VERIFIED and verdict.compartment not in spot:
            continue
        details = "; ".join(describe_finding(f, catalog) for f in verdict.findings) or "random spot check"
        lines.append(f"{layout.label(verdict.compartment)} [{verdict.status.value}]: {details}")
    return lines
