"""Learning identification thresholds from the brain's own memory by self-testing."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

from .identity import IdentityPolicy
from .knowledge import KnowledgeBase


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval (z = 1.96 → 95%)."""
    if trials <= 0:
        return 0.0
    n, p = float(trials), successes / trials
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    return max(0.0, (centre - spread) / (1 + z2 / n))


class CalibrationOutcome(StrEnum):
    CALIBRATED = "calibrated"
    INSUFFICIENT_DATA = "insufficientData"
    TARGET_UNREACHABLE = "targetUnreachable"


class CalibrationReport(BaseModel):
    outcome: CalibrationOutcome
    calibrated_at: datetime
    accept_similarity: float | None = None
    minimum_margin: float | None = None
    medications: int = 0
    queries: int = 0
    trials: int = 0
    #: Known-medication trials where a medication was named.
    identified: int = 0
    #: Named trials that were wrong, including naming anything for a pill whose medication was hidden.
    false_identifications: int = 0
    precision_lower_bound: float = 0.0
    coverage: float = 0.0


class IdentityCalibrator:
    """Every remembered pill is classified against the other photos (closed-set trial) and against memory
    with its whole medication hidden (open-set trial, where naming anything is an error). Naming is only a
    claim — acceptance additionally needs the trust ledger — so the target must be reachable with a few
    hundred remembered pills."""

    def __init__(
        self,
        required_precision_lower_bound: float = 0.9,
        minimum_medications: int = 2,
        minimum_queries: int = 30,
        maximum_queries: int = 1500,
        margin_grid: tuple[float, ...] = (0, 0.01, 0.02, 0.04, 0.08, 0.15),
    ):
        self.required = required_precision_lower_bound
        self.minimum_medications = minimum_medications
        self.minimum_queries = minimum_queries
        self.maximum_queries = maximum_queries
        self.margin_grid = margin_grid

    def calibrate(
        self, knowledge: KnowledgeBase, base: IdentityPolicy, now: datetime | None = None
    ) -> tuple[CalibrationReport, IdentityPolicy | None]:
        now = now or datetime.now(UTC)
        index = knowledge.index()
        eligible = {
            m
            for m in knowledge.medications
            if knowledge.count(m) >= base.minimum_exemplars and knowledge.group_count(m) >= base.minimum_groups
        }
        queries = sorted((e for e in knowledge.exemplars if e.medication_id in eligible), key=lambda e: e.id)
        if len(queries) > self.maximum_queries:
            step = len(queries) / self.maximum_queries
            queries = [queries[int(i * step)] for i in range(self.maximum_queries)]
        if len(eligible) < self.minimum_medications or len(queries) < self.minimum_queries:
            return (
                CalibrationReport(
                    outcome=CalibrationOutcome.INSUFFICIENT_DATA,
                    calibrated_at=now,
                    medications=len(eligible),
                    queries=len(queries),
                ),
                None,
            )

        truths, best_meds, scores, margins = [], [], [], []
        for query in queries:
            for hide in (False, True):
                candidates = [
                    c
                    for c in index.candidates(
                        query.vector,
                        base.neighbours,
                        excluding_group=query.group_id,
                        excluding_medication=query.medication_id if hide else None,
                    )
                    if c.medication_id in eligible or c.medication_id == query.medication_id
                ]
                truths.append(None if hide else query.medication_id)
                best_meds.append(candidates[0].medication_id if candidates else None)
                scores.append(candidates[0].score if candidates else -np.inf)
                margins.append(candidates[0].score - candidates[1].score if len(candidates) > 1 else np.inf)

        scores_a, margins_a = np.array(scores), np.array(margins)
        closed = np.array([t is not None for t in truths])
        wrong_if_named = np.array([b != t for b, t in zip(best_meds, truths, strict=True)])
        best = None
        for margin in self.margin_grid:
            margin_ok = margins_a >= margin
            for threshold in np.unique(scores_a[np.isfinite(scores_a)]):
                named = margin_ok & (scores_a >= threshold)
                n = int(named.sum())
                if n == 0:
                    continue
                wrong = int((named & wrong_if_named).sum())
                bound = wilson_lower_bound(n - wrong, n)
                if bound < self.required:
                    continue
                named_closed = int((named & closed).sum())
                key = (named_closed, bound)
                if best is None or key > best[0]:
                    best = (key, float(threshold), float(margin), wrong)
        trials = len(truths)
        if best is None:
            return (
                CalibrationReport(
                    outcome=CalibrationOutcome.TARGET_UNREACHABLE,
                    calibrated_at=now,
                    medications=len(eligible),
                    queries=len(queries),
                    trials=trials,
                ),
                None,
            )
        (named_closed, bound), threshold, margin, wrong = best
        policy = base.model_copy(
            update={"accept_similarity": threshold, "minimum_margin": margin, "version": base.version + 1}
        )
        report = CalibrationReport(
            outcome=CalibrationOutcome.CALIBRATED,
            calibrated_at=now,
            accept_similarity=threshold,
            minimum_margin=margin,
            medications=len(eligible),
            queries=len(queries),
            trials=trials,
            identified=named_closed,
            false_identifications=wrong,
            precision_lower_bound=bound,
            coverage=named_closed / max(1, int(closed.sum())),
        )
        return report, policy
