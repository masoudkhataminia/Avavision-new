"""How well each second-opinion advisor would have done on packs the pharmacist already checked.

The audit log is the ground truth: every signed-off check keeps its evidence photo, its profile and the
pharmacist's outcome for each compartment. A compartment the pharmacist *corrected* was wrong in the photo; one
they *confirmed correct* was right. Unresolved and unreviewed compartments are left out.

An advisor only helps when it disagrees with a wrong compartment (caught). Agreeing with it or being unsure is a
miss, and disagreeing with a correct one is a false alarm that costs the pharmacist a look.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from ..core.advisory import AdvisoryOpinion, AdvisoryVerdict
from ..core.audit import CheckRecord
from ..core.models import Catalog, CompartmentIndex, PackLayout, PackProfile
from ..core.signoff import ReviewOutcome
from ..vision.pipeline import crop_compartment
from .claude import ExpertAnswer, ExpertError


class Reviewer(Protocol):
    def review_compartment(
        self, image: np.ndarray, index: CompartmentIndex, layout: PackLayout, profile: PackProfile, catalog: Catalog
    ) -> ExpertAnswer[AdvisoryOpinion]: ...


@dataclass
class Sample:
    pack: str
    index: CompartmentIndex
    label: str
    correct: bool
    image: np.ndarray
    layout: PackLayout
    profile: PackProfile


def samples_from_records(
    records: Iterable[CheckRecord], load_image: Callable[[str], np.ndarray | None]
) -> list[Sample]:
    """One sample per compartment the pharmacist confirmed or corrected, cropped from the check's evidence photo."""
    samples = []
    for record in records:
        canvas = next((image for name in record.evidence_images if (image := load_image(name)) is not None), None)
        if canvas is None:
            continue
        for review in record.sign_off.reviews:
            if review.outcome not in (ReviewOutcome.CONFIRMED_CORRECT, ReviewOutcome.CORRECTED):
                continue
            index = review.compartment
            samples.append(
                Sample(
                    pack=record.profile.reference,
                    index=index,
                    label=record.layout.label(index),
                    correct=review.outcome == ReviewOutcome.CONFIRMED_CORRECT,
                    image=crop_compartment(canvas, record.layout, index),
                    layout=record.layout,
                    profile=record.profile,
                )
            )
    return samples


class AdvisorScore(BaseModel):
    advisor: str
    wrong: int = 0
    caught: int = 0
    correct: int = 0
    false_alarms: int = 0
    unsure: int = 0
    errors: int = 0
    seconds: float = 0.0

    @property
    def answered(self) -> int:
        return self.wrong + self.correct - self.errors

    @property
    def seconds_each(self) -> float:
        return self.seconds / max(1, self.answered)


class SampleResult(BaseModel):
    advisor: str
    pack: str
    compartment: str
    correct: bool
    verdict: str
    observed_count: int | None = None
    note: str = ""
    seconds: float = 0.0


def compare_advisors(
    advisors: dict[str, Reviewer], samples: list[Sample], catalog: Catalog
) -> tuple[list[AdvisorScore], list[SampleResult]]:
    scores, rows = [], []
    for name, advisor in advisors.items():
        score = AdvisorScore(advisor=name)
        for sample in samples:
            if sample.correct:
                score.correct += 1
            else:
                score.wrong += 1
            start = time.perf_counter()
            try:
                opinion = advisor.review_compartment(
                    sample.image, sample.index, sample.layout, sample.profile, catalog
                ).value
            except ExpertError as error:
                score.errors += 1
                rows.append(
                    SampleResult(
                        advisor=name,
                        pack=sample.pack,
                        compartment=sample.label,
                        correct=sample.correct,
                        verdict="error",
                        note=str(error),
                    )
                )
                continue
            seconds = time.perf_counter() - start
            score.seconds += seconds
            disagrees = opinion.verdict == AdvisoryVerdict.DISAGREES
            score.caught += disagrees and not sample.correct
            score.false_alarms += disagrees and sample.correct
            score.unsure += opinion.verdict == AdvisoryVerdict.UNSURE
            rows.append(
                SampleResult(
                    advisor=name,
                    pack=sample.pack,
                    compartment=sample.label,
                    correct=sample.correct,
                    verdict=opinion.verdict.value,
                    observed_count=opinion.observed_count,
                    note=opinion.note,
                    seconds=round(seconds, 2),
                )
            )
        scores.append(score)
    return scores, rows


def report(scores: list[AdvisorScore]) -> str:
    lines = [f"{'advisor':34} {'wrong caught':>14} {'false alarms':>14} {'unsure':>7} {'errors':>7} {'s each':>7}"]
    for s in scores:
        lines.append(
            f"{s.advisor:34} {f'{s.caught} of {s.wrong}':>14} {f'{s.false_alarms} of {s.correct}':>14} "
            f"{s.unsure:>7} {s.errors:>7} {s.seconds_each:>7.1f}"
        )
    return "\n".join(lines)
