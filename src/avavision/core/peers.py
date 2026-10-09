"""Same tablets, same look: compartments the profile fills alike must look alike.

In a weekly pack most compartments repeat (the morning dose of all seven days, say), and a dose given only on
some days still follows from the profile. Compartments with identical expected contents form a group; one whose
content looks unlike its group's consensus goes to the pharmacist. This catches one-off packing errors (a missing,
extra or swapped tablet on one day) without knowing what any tablet looks like.

It only escalates. It cannot see an error repeated in every compartment of a group (a wrong bottle used all
week), which the identity checks and the pharmacist cover, and it compares appearance signatures, so tablets that
look like the pocket floor (white on white) add little to it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from pydantic import BaseModel

from .engine import CompartmentStatus, CompartmentVerdict, FindingKind, PackStatus, VerificationResult, finding
from .models import CompartmentIndex, PackLayout, PackProfile


class PeerPolicy(BaseModel):
    #: Smaller groups have no reliable consensus.
    minimum_group: int = 3
    #: A compartment is unlike its group when its distance from the consensus exceeds both ``spread_factor``
    #: times the group's median distance and the absolute minimum. Set from the first real pack (D-135):
    #: correct compartments stayed within 2.3 times their group's median; a missing, extra or swapped coloured
    #: tablet gave 6 to 7 times.
    minimum_distance: float = 0.025
    spread_factor: float = 3.5


def peer_groups(
    profile: PackProfile, layout: PackLayout, policy: PeerPolicy | None = None
) -> list[list[CompartmentIndex]]:
    """Compartments with identical, non-empty expected contents, in groups large enough to compare."""
    policy = policy or PeerPolicy()
    groups: dict[tuple, list[CompartmentIndex]] = {}
    for index in layout.all_compartments:
        expectation = profile.expectation(index)
        contents = tuple(sorted((m, q) for m, q in (expectation.quantities if expectation else {}).items() if q > 0))
        if contents:
            groups.setdefault(contents, []).append(index)
    return [group for group in groups.values() if len(group) >= policy.minimum_group]


def unlike_peers(
    signatures: dict[CompartmentIndex, Sequence[float]],
    groups: list[list[CompartmentIndex]],
    policy: PeerPolicy | None = None,
) -> dict[CompartmentIndex, float]:
    """Compartments whose signature is far from their group's consensus (the per-component median), with their
    distance. Compartments without a signature (obscured, say) take no part."""
    policy = policy or PeerPolicy()
    result: dict[CompartmentIndex, float] = {}
    for group in groups:
        members = [i for i in group if i in signatures]
        if len(members) < policy.minimum_group:
            continue
        vectors = [list(signatures[i]) for i in members]
        consensus = [statistics.median(values) for values in zip(*vectors, strict=True)]
        distances = [sum(abs(v - c) for v, c in zip(vector, consensus, strict=True)) for vector in vectors]
        limit = max(policy.minimum_distance, policy.spread_factor * statistics.median(distances))
        result.update({i: d for i, d in zip(members, distances, strict=True) if d > limit})
    return result


def apply_peer_check(result: VerificationResult, unlike: dict[CompartmentIndex, float]) -> VerificationResult:
    """The result with every compartment unlike its group escalated to review. Statuses only ever get worse."""
    if not unlike:
        return result
    verdicts = []
    for verdict in result.compartments:
        if verdict.compartment in unlike and not any(f.kind == FindingKind.UNLIKE_PEERS for f in verdict.findings):
            escalated = CompartmentVerdict.build(
                verdict.compartment,
                [*verdict.findings, finding(FindingKind.UNLIKE_PEERS)],
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
