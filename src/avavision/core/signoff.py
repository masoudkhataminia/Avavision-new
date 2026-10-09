"""The pharmacist's accountable decision on a pack."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from .models import CompartmentIndex


class ReviewOutcome(StrEnum):
    CONFIRMED_CORRECT = "confirmedCorrect"  # inspected, content is correct
    CORRECTED = "corrected"  # content was wrong and the pharmacist fixed the pack
    UNRESOLVED = "unresolved"  # wrong or uncertain and not fixed


class CompartmentReview(BaseModel):
    compartment: CompartmentIndex
    outcome: ReviewOutcome
    note: str | None = None


class SignOffDecision(StrEnum):
    RELEASED = "released"
    WITHHELD = "withheld"


class SignOff(BaseModel):
    pharmacist: str  # initials or staff code
    decision: SignOffDecision
    reviews: list[CompartmentReview]
    acknowledged_pack_findings: bool = False
    note: str | None = None
    signed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def review(self, index: CompartmentIndex) -> CompartmentReview | None:
        return next((r for r in reversed(self.reviews) if r.compartment == index), None)


class SignOffErrorKind(StrEnum):
    MISSING_PHARMACIST = "missingPharmacist"
    REVIEW_FOR_UNKNOWN_COMPARTMENT = "reviewForUnknownCompartment"
    UNREVIEWED_COMPARTMENTS = "unreviewedCompartments"
    UNRESOLVED_COMPARTMENTS = "unresolvedCompartments"
    PACK_FINDINGS_NOT_ACKNOWLEDGED = "packFindingsNotAcknowledged"


class SignOffError(ValueError):
    def __init__(self, kind: SignOffErrorKind, compartments: list[CompartmentIndex] | None = None):
        super().__init__(kind.value)
        self.kind = kind
        self.compartments = compartments or []


def validate_sign_off(sign_off: SignOff, result) -> None:
    """A pack can be released only when every compartment requiring review was inspected and is correct or
    corrected, and every pack-level finding was acknowledged. Withholding is always allowed."""
    if not sign_off.pharmacist.strip():
        raise SignOffError(SignOffErrorKind.MISSING_PHARMACIST)
    known = {v.compartment for v in result.compartments}
    for review in sign_off.reviews:
        if review.compartment not in known:
            raise SignOffError(SignOffErrorKind.REVIEW_FOR_UNKNOWN_COMPARTMENT, [review.compartment])
    if sign_off.decision != SignOffDecision.RELEASED:
        return
    unreviewed = [c for c in result.compartments_requiring_review if sign_off.review(c) is None]
    if unreviewed:
        raise SignOffError(SignOffErrorKind.UNREVIEWED_COMPARTMENTS, unreviewed)
    unresolved = [
        v.compartment
        for v in result.compartments
        if (r := sign_off.review(v.compartment)) is not None and r.outcome == ReviewOutcome.UNRESOLVED
    ]
    if unresolved:
        raise SignOffError(SignOffErrorKind.UNRESOLVED_COMPARTMENTS, unresolved)
    if result.pack_findings and not sign_off.acknowledged_pack_findings:
        raise SignOffError(SignOffErrorKind.PACK_FINDINGS_NOT_ACKNOWLEDGED)
