"""Advisory opinions: a second look at a compartment from outside the decision engine (today the Claude expert).

Advisors are never part of the acceptance path. An opinion can only escalate: a compartment the advisor
disagrees with goes to pharmacist review, while agreement or uncertainty changes nothing. No opinion can
turn a finding into an accepted result.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .engine import CompartmentStatus, CompartmentVerdict, FindingKind, PackStatus, VerificationResult, finding
from .models import CompartmentIndex


class AdvisoryVerdict(StrEnum):
    AGREES = "agrees"  # the advisor sees what the profile expects
    DISAGREES = "disagrees"  # the advisor sees something different
    UNSURE = "unsure"  # the advisor could not tell


class AdvisoryOpinion(BaseModel):
    model_config = ConfigDict(frozen=True)

    compartment: CompartmentIndex
    verdict: AdvisoryVerdict
    observed_count: int | None = None
    note: str = ""
    #: Who gave the opinion, e.g. the model that served the request.
    source: str


def apply_advisories(result: VerificationResult, opinions: list[AdvisoryOpinion]) -> VerificationResult:
    """The result with every disagreement escalated to review. Statuses only ever get worse."""
    disagreeing = {o.compartment for o in opinions if o.verdict == AdvisoryVerdict.DISAGREES}
    if not disagreeing:
        return result
    verdicts = []
    for verdict in result.compartments:
        if verdict.compartment in disagreeing and not any(
            f.kind == FindingKind.ADVISOR_DISAGREES for f in verdict.findings
        ):
            escalated = CompartmentVerdict.build(
                verdict.compartment,
                [*verdict.findings, finding(FindingKind.ADVISOR_DISAGREES)],
                verdict.expected_count,
                verdict.observed_count,
                verdict.observed_medications,
            )
            worst = max(verdict.status, escalated.status, key=lambda s: s.severity)
            verdict = escalated.model_copy(update={"status": worst})
        verdicts.append(verdict)
    status = result.status
    if status in (PackStatus.VERIFIED, PackStatus.COUNT_MATCHED) and any(
        v.status.severity >= CompartmentStatus.NEEDS_REVIEW.severity for v in verdicts
    ):
        status = PackStatus.NEEDS_REVIEW
    return result.model_copy(update={"compartments": verdicts, "status": status})
