"""Holdout evaluation of the whole system, used by the release gate."""

from __future__ import annotations

from pydantic import BaseModel

from .models import MedicationID


class EvaluationSample(BaseModel):
    """One holdout compartment: expectation, pharmacist-confirmed truth and what the system said."""

    sample_id: str
    expected: dict[MedicationID, int]
    truth: dict[MedicationID, int]
    truth_has_broken_or_foreign: bool = False
    predicted_status: str  # a CompartmentStatus value
    predicted_count: int | None
    predicted_medications: dict[MedicationID, int] = {}

    @property
    def true_count(self) -> int:
        return sum(self.truth.values())

    @property
    def expected_count(self) -> int:
        return sum(self.expected.values())

    @property
    def is_erroneous(self) -> bool:
        positive = lambda d: {k: v for k, v in d.items() if v > 0}  # noqa: E731
        return self.truth_has_broken_or_foreign or positive(self.truth) != positive(self.expected)

    @property
    def has_count_error(self) -> bool:
        return self.truth_has_broken_or_foreign or self.true_count != self.expected_count


class MedicationMetrics(BaseModel):
    samples: int
    precision: float
    recall: float


class EvaluationReport(BaseModel):
    dataset_id: str
    total_samples: int
    error_samples: int
    count_accuracy: float
    identity_precision: float
    identity_recall: float
    false_acceptance_rate: float
    count_false_acceptance_rate: float
    review_rate: float
    per_medication: dict[MedicationID, MedicationMetrics]


def _ratio(numerator: int, denominator: int, empty: float = 0.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def evaluate(samples: list[EvaluationSample], dataset_id: str) -> EvaluationReport:
    counted = [s for s in samples if s.predicted_count is not None]
    correct_counts = sum(1 for s in counted if s.predicted_count == s.true_count)
    erroneous = [s for s in samples if s.is_erroneous]
    falsely_verified = sum(1 for s in erroneous if s.predicted_status == "verified")
    count_errors = [s for s in samples if s.has_count_error]
    count_accepted = sum(1 for s in count_errors if s.predicted_status in ("verified", "countMatched"))

    tp = fp = fn = 0
    tallies: dict[MedicationID, list[int]] = {}
    for s in samples:
        for medication in set(s.truth) | set(s.predicted_medications):
            truth = s.truth.get(medication, 0)
            predicted = s.predicted_medications.get(medication, 0)
            t, f, n = min(truth, predicted), max(0, predicted - truth), max(0, truth - predicted)
            tp, fp, fn = tp + t, fp + f, fn + n
            tally = tallies.setdefault(medication, [0, 0, 0, 0])
            tally[0] += 1 if truth > 0 else 0
            tally[1] += t
            tally[2] += f
            tally[3] += n

    return EvaluationReport(
        dataset_id=dataset_id,
        total_samples=len(samples),
        error_samples=len(erroneous),
        count_accuracy=_ratio(correct_counts, len(counted)),
        identity_precision=_ratio(tp, tp + fp),
        identity_recall=_ratio(tp, tp + fn),
        false_acceptance_rate=_ratio(falsely_verified, len(erroneous), empty=1.0),
        count_false_acceptance_rate=_ratio(count_accepted, len(count_errors), empty=1.0),
        review_rate=_ratio(sum(1 for s in samples if s.predicted_status == "needsReview"), len(samples)),
        per_medication={
            m: MedicationMetrics(samples=t[0], precision=_ratio(t[1], t[1] + t[2]), recall=_ratio(t[1], t[1] + t[3]))
            for m, t in tallies.items()
        },
    )
