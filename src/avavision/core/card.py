"""The pack's header card: the camera reads its code, which must belong to the profile being checked.

Checking a pack against the wrong person's profile is the most dangerous mistake a station can make, and no
pill-level check can catch it. A card that is in view and is not this profile's sends the whole pack to review.
A card that cannot be seen proves nothing either way, so it changes nothing.
"""

from __future__ import annotations

from .engine import PackFinding, PackFindingKind, PackStatus, VerificationResult
from .models import PackProfile


def check_card(result: VerificationResult, profile: PackProfile, codes_seen: list[str]) -> VerificationResult:
    if not profile.barcode or not codes_seen or profile.barcode in codes_seen:
        return result
    if any(f.kind == PackFindingKind.PACK_CARD_MISMATCH for f in result.pack_findings):
        return result
    finding = PackFinding(kind=PackFindingKind.PACK_CARD_MISMATCH, codes=tuple(sorted(codes_seen)))
    status = result.status
    if status in (PackStatus.VERIFIED, PackStatus.COUNT_MATCHED):
        status = PackStatus.NEEDS_REVIEW
    return result.model_copy(update={"pack_findings": [*result.pack_findings, finding], "status": status})
