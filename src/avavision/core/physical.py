"""Physical evidence: a pill's real size (mm) and colour, compared with what the pharmacy has confirmed for each
medication. A second channel of evidence, independent of the appearance embedder (as JVM, Omnicell and Eyecon
check size, shape and colour).

It can only escalate: a pill whose size or colour fits none of the expected medications sends the compartment to
review. Fitting the ranges never accepts anything on its own.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from .models import MedicationID


class PhysicalFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    length_mm: float
    width_mm: float
    #: CIE L*a*b* of the pill's face (L* 0–100).
    lightness: float
    a: float
    b: float

    def colour_distance(self, lightness: float, a: float, b: float) -> float:
        """ΔE with lightness weighted down: blister plastic and light change lightness more than hue."""
        return (0.5 * (self.lightness - lightness) ** 2 + (self.a - a) ** 2 + (self.b - b) ** 2) ** 0.5


class PhysicalPolicy(BaseModel):
    #: A medication's ranges are used only after this many confirmed pills from this many packs.
    minimum_samples: int = 8
    minimum_groups: int = 2
    #: Size tolerance beyond the spread seen so far: the larger of an absolute and a relative margin.
    size_margin_mm: float = 0.7
    size_margin_fraction: float = 0.1
    colour_margin: float = 10.0
    #: Robust spread (median absolute deviation) multiples added to the margins.
    spread_factor: float = 4.0
    #: Measurements this far from the median (spread multiples, at least a fraction of the median) do not move the
    #: range's edges: a broken tablet, two touching tablets or a mislabelled pill must not widen what passes as
    #: this medication (D-138). They only make the range tighter, so they can only escalate.
    outlier_factor: float = 6.0
    outlier_floor_fraction: float = 0.15
    outlier_floor_colour: float = 8.0


class PhysicalRange(BaseModel):
    medication_id: MedicationID
    samples: int
    groups: int
    length_mm: tuple[float, float]
    width_mm: tuple[float, float]
    colour: tuple[float, float, float]
    colour_radius: float

    def admits(self, f: PhysicalFeatures) -> bool:
        return (
            self.length_mm[0] <= f.length_mm <= self.length_mm[1]
            and self.width_mm[0] <= f.width_mm <= self.width_mm[1]
            and f.colour_distance(*self.colour) <= self.colour_radius
        )


class PhysicalEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    features: PhysicalFeatures
    #: Medications with established ranges that this pill was compared with.
    known: tuple[MedicationID, ...] = ()
    #: Those whose ranges admit the pill.
    consistent_with: tuple[MedicationID, ...] = ()


def _mad(values: list[float]) -> float:
    centre = statistics.median(values)
    return statistics.median(abs(v - centre) for v in values)


def _typical(values: list[float], limit: float) -> list[float]:
    """The values within ``limit`` of the median (never empty)."""
    centre = statistics.median(values)
    return [v for v in values if abs(v - centre) <= limit] or [centre]


def _interval(values: list[float], policy: PhysicalPolicy) -> tuple[float, float]:
    centre, spread = statistics.median(values), _mad(values)
    typical = _typical(values, max(policy.outlier_factor * spread, policy.outlier_floor_fraction * centre))
    lo, hi = min(typical), max(typical)
    margin = max(policy.size_margin_mm, policy.size_margin_fraction * centre) + policy.spread_factor * spread
    return (min(lo, centre) - margin, max(hi, centre) + margin)


def learn_ranges(
    samples: dict[MedicationID, list[tuple[PhysicalFeatures, str]]], policy: PhysicalPolicy | None = None
) -> dict[MedicationID, PhysicalRange]:
    """Ranges for every medication with enough confirmed pills from enough different packs."""
    policy = policy or PhysicalPolicy()
    ranges = {}
    for medication, rows in samples.items():
        groups = {group for _, group in rows}
        if len(rows) < policy.minimum_samples or len(groups) < policy.minimum_groups:
            continue
        features = [f for f, _ in rows]
        centre = tuple(statistics.median(getattr(f, k) for f in features) for k in ("lightness", "a", "b"))
        distances = [f.colour_distance(*centre) for f in features]
        spread = _mad(distances)
        typical = _typical(distances, max(policy.outlier_factor * spread, policy.outlier_floor_colour))
        radius = max(typical) + policy.colour_margin + policy.spread_factor * spread
        ranges[medication] = PhysicalRange(
            medication_id=medication,
            samples=len(rows),
            groups=len(groups),
            length_mm=_interval([f.length_mm for f in features], policy),
            width_mm=_interval([f.width_mm for f in features], policy),
            colour=centre,
            colour_radius=radius,
        )
    return ranges


def physical_evidence(features: PhysicalFeatures, ranges: dict[MedicationID, PhysicalRange]) -> PhysicalEvidence:
    return PhysicalEvidence(
        features=features,
        known=tuple(sorted(ranges)),
        consistent_with=tuple(sorted(m for m, r in ranges.items() if r.admits(features))),
    )


def fits_none(evidence: PhysicalEvidence | None, expected: set[MedicationID]) -> bool:
    """True when every expected medication has established ranges and the pill fits none of them."""
    if evidence is None or not expected or not expected.issubset(evidence.known):
        return False
    return not expected.intersection(evidence.consistent_with)


def group_samples(rows: list[tuple[MedicationID, PhysicalFeatures, str]]) -> dict[MedicationID, list]:
    grouped: dict[MedicationID, list[tuple[PhysicalFeatures, str]]] = defaultdict(list)
    for medication, features, group in rows:
        grouped[medication].append((features, group))
    return grouped
