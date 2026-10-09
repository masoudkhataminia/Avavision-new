from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from avavision.core.advisory import AdvisoryOpinion, AdvisoryVerdict, apply_advisories
from avavision.core.engine import CompartmentStatus, FindingKind, PackFinding, PackFindingKind, PackStatus, finding
from avavision.core.models import WEEKLY_7X4, Catalog, Medication, cell
from avavision.core.session import CheckSession, SessionError
from avavision.expert.charts import (
    ChartDose,
    ChartLine,
    DoseTime,
    ImportIssueKind,
    MedicationChart,
    Weekday,
    chart_to_profile,
)
from avavision.expert.claude import (
    FALLBACK_BETA,
    MODEL,
    ClaudeExpert,
    CompartmentReading,
    ExpertIncomplete,
    ExpertRefused,
    Explanation,
    Match,
)
from avavision.expert.describe import describe_finding, describe_pack_finding, describe_result
from conftest import CATALOG, LAYOUT, METFORMIN, ONE, TARGET, engine, frames, full_pack, profile

DAILY = list(Weekday)
CHART_CATALOG = Catalog(
    medications=[
        Medication(id="metformin-500", name="Metformin", strength="500 mg"),
        Medication(id="aspirin-100", name="Aspirin", strength="100mg"),
        Medication(id="alendronate-70", name="Alendronate", strength="70 mg"),
        Medication(id="amlodipine-5", name="Amlodipine", strength="5 mg"),
        Medication(id="amlodipine-10", name="Amlodipine", strength="10 mg"),
    ]
)


def line(name, doses, days=DAILY, strength=None, **kwargs) -> ChartLine:
    return ChartLine(
        name=name,
        strength=strength,
        doses=[ChartDose(time=t, quantity=q) for t, q in doses],
        days=days,
        in_pack=kwargs.pop("in_pack", True),
        unclear=kwargs.pop("unclear", []),
        **kwargs,
    )


def chart(first_day=None) -> MedicationChart:
    return MedicationChart(
        first_day=first_day,
        lines=[
            line("METFORMIN", [(DoseTime.BREAKFAST, 1), (DoseTime.DINNER, 2)], strength="500mg"),
            line("Aspirin", [(DoseTime.BREAKFAST, 1)], strength="100 mg"),
            line("Warfarin", [(DoseTime.DINNER, 1)], strength="1 mg"),
            line("Paracetamol", [(DoseTime.BEDTIME, 2)], in_pack=False),
            line("Alendronate", [(DoseTime.BREAKFAST, 1)], days=[Weekday.MONDAY], strength="70 mg"),
            line("Amlodipine", [(DoseTime.BREAKFAST, 0.5)], unclear=["strength"]),
        ],
        warnings=["line 7 crossed out"],
    )


def kinds(draft):
    return {(i.kind, i.line) for i in draft.issues}


def test_chart_becomes_a_draft_profile_with_every_doubt_listed():
    draft = chart_to_profile(chart(), CHART_CATALOG, WEEKLY_7X4, "PACK-1")
    assert draft.matches == {0: "metformin-500", 1: "aspirin-100", 4: "alendronate-70"}
    assert draft.not_packed == [3]
    assert kinds(draft) >= {
        (ImportIssueKind.CHART_WARNING, None),
        (ImportIssueKind.NOT_MATCHED, 2),
        (ImportIssueKind.UNKNOWN_START_DAY, 4),
        (ImportIssueKind.AMBIGUOUS_MATCH, 5),
        (ImportIssueKind.UNCLEAR, 5),
    }
    assert draft.placed_lines == [0, 1]
    for day in range(7):
        morning = draft.profile.expectation(cell(0, day)).quantities
        assert morning == {"aspirin-100": 1, "metformin-500": 1}
        assert draft.profile.expectation(cell(2, day)).quantities == {"metformin-500": 2}
        assert draft.profile.expectation(cell(3, day)).items == []
    assert draft.profile.issues(WEEKLY_7X4, CHART_CATALOG) == []


def test_weekly_dose_needs_the_first_day_and_halves_are_never_placed():
    draft = chart_to_profile(
        chart(Weekday.WEDNESDAY), CHART_CATALOG, WEEKLY_7X4, "PACK-1", overrides={5: "amlodipine-5"}
    )
    # Monday is the sixth column of a pack that starts on Wednesday.
    assert draft.profile.expectation(cell(0, 5)).quantities["alendronate-70"] == 1
    assert "alendronate-70" not in draft.profile.expectation(cell(0, 4)).quantities
    assert (ImportIssueKind.FRACTIONAL_DOSE, 5) in kinds(draft)
    assert all("amlodipine-5" not in c.quantities for c in draft.profile.compartments)


def test_other_layouts_are_not_guessed():
    draft = chart_to_profile(chart(), CHART_CATALOG, LAYOUT, "PACK-1")
    assert draft.issues[-1].kind == ImportIssueKind.UNSUPPORTED_LAYOUT and draft.profile.total_doses == 0


# --------------------------------------------------------------------------- advisory


def result_with():
    session = CheckSession(LAYOUT, profile(ONE), CATALOG)
    for f in frames(full_pack()):
        session.record(f)
    session.analyze(engine())
    return session


def test_advisories_only_escalate():
    session = result_with()
    result = session.result
    assert result.status == PackStatus.COUNT_MATCHED
    agree = AdvisoryOpinion(compartment=cell(0, 0), verdict=AdvisoryVerdict.AGREES, source="m")
    unsure = AdvisoryOpinion(compartment=cell(0, 2), verdict=AdvisoryVerdict.UNSURE, source="m")
    assert apply_advisories(result, [agree, unsure]) == result

    disagree = AdvisoryOpinion(compartment=TARGET, verdict=AdvisoryVerdict.DISAGREES, observed_count=2, source="m")
    escalated = apply_advisories(result, [disagree])
    assert escalated.status == PackStatus.NEEDS_REVIEW
    assert escalated.verdict(TARGET).status == CompartmentStatus.NEEDS_REVIEW
    assert escalated.verdict(TARGET).findings[-1].kind == FindingKind.ADVISOR_DISAGREES
    assert escalated.verdict(cell(0, 0)) == result.verdict(cell(0, 0))
    assert apply_advisories(escalated, [disagree]) == escalated

    # A mismatch stays a mismatch.
    mismatch = escalated.verdict(TARGET).model_copy(update={"status": CompartmentStatus.MISMATCH})
    worse = escalated.model_copy(
        update={"compartments": [mismatch if v.compartment == TARGET else v for v in escalated.compartments]}
    )
    other = disagree.model_copy(update={"compartment": cell(1, 1)})
    assert apply_advisories(worse, [other]).verdict(TARGET).status == CompartmentStatus.MISMATCH


def test_session_records_advice_in_the_audit_record():
    session = result_with()
    with pytest.raises(SessionError):
        session.advise([AdvisoryOpinion(compartment=cell(5, 5), verdict=AdvisoryVerdict.AGREES, source="m")])
    opinion = AdvisoryOpinion(compartment=TARGET, verdict=AdvisoryVerdict.DISAGREES, source="m")
    assert session.advise([opinion]).status == PackStatus.NEEDS_REVIEW
    assert session.advisories == [opinion]
    session.retake()
    assert session.advisories == []
    with pytest.raises(SessionError):
        session.advise([opinion])


# --------------------------------------------------------------------------- Claude client


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def parse(self, **request):
        self.requests.append(request)
        return self.replies.pop(0)


def reply(parsed=None, stop_reason="end_turn", model=MODEL, iterations=None, stop_details=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=stop_details,
        parsed_output=parsed,
        model=model,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300, iterations=iterations),
        _request_id="req_1",
    )


def expert(*replies):
    messages = FakeMessages(replies)
    return ClaudeExpert(client=SimpleNamespace(beta=SimpleNamespace(messages=messages))), messages


def test_chart_reading_request_shape():
    claude, messages = expert(reply(chart()))
    answer = claude.read_chart(b"%PDF-1.7 fake", "application/pdf")
    assert answer.value == chart() and answer.call.served_model == MODEL and not answer.call.fallback_used
    request = messages.requests[0]
    assert request["model"] == MODEL
    assert request["betas"] == [FALLBACK_BETA] and request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": "high"} and request["output_format"] is MedicationChart
    document = request["messages"][0]["content"][0]
    assert document["type"] == "document" and base64.b64decode(document["source"]["data"]) == b"%PDF-1.7 fake"
    assert "thinking" not in request and "temperature" not in request


def test_refusals_cut_off_answers_and_fallbacks_are_reported():
    details = SimpleNamespace(category="bio", explanation="declined")
    claude, _ = expert(reply(stop_reason="refusal", stop_details=details), reply(stop_reason="max_tokens"), reply())
    with pytest.raises(ExpertRefused) as refused:
        claude.read_chart(b"text", "text/plain")
    assert refused.value.category == "bio"
    with pytest.raises(ExpertIncomplete):
        claude.read_chart(b"text", "text/plain")
    with pytest.raises(ExpertIncomplete):
        claude.read_chart(b"text", "text/plain")

    served = reply(chart(), model="claude-opus-4-8", iterations=[SimpleNamespace(type="fallback_message")])
    claude, _ = expert(served)
    call = claude.read_chart(b"text", "text/plain").call
    assert call.fallback_used and call.served_model == "claude-opus-4-8"


def test_fallbacks_can_be_switched_off():
    claude, messages = expert(reply(Explanation(summary="s", steps=[])))
    claude.settings.fallbacks = False
    session = result_with()
    claude.explain(session.result, LAYOUT, CATALOG)
    assert "fallbacks" not in messages.requests[0] and "betas" not in messages.requests[0]


def test_explanation_is_built_from_engine_facts():
    claude, messages = expert(reply(Explanation(summary="خلاصه", steps=["بررسی کنید"])))
    session = result_with()
    session.advise([AdvisoryOpinion(compartment=TARGET, verdict=AdvisoryVerdict.DISAGREES, source="m")])
    answer = claude.explain(session.result, LAYOUT, CATALOG, language="fa")
    request = messages.requests[0]
    assert "Persian" in request["system"] and request["output_config"] == {"effort": "low"}
    text = request["messages"][0]["content"][0]["text"]
    assert "Pack status: needsReview" in text and LAYOUT.label(TARGET) in text and "second opinion" in text
    assert answer.value.summary == "خلاصه"


def test_compartment_review_becomes_an_advisory_opinion():
    image = np.full((200, 200, 3), 200, np.uint8)
    p = profile(ONE)
    agree = CompartmentReading(pill_count=1, matches_expected=Match.YES, differences=[], note="one white tablet")
    miscount = CompartmentReading(pill_count=2, matches_expected=Match.YES, differences=[], note="")
    unsure = CompartmentReading(pill_count=None, matches_expected=Match.UNSURE, differences=["glare"], note="")
    claude, messages = expert(reply(agree), reply(miscount), reply(unsure))
    opinions = [claude.review_compartment(image, TARGET, LAYOUT, p, CATALOG).value for _ in range(3)]
    assert [o.verdict for o in opinions] == [
        AdvisoryVerdict.AGREES,
        AdvisoryVerdict.DISAGREES,
        AdvisoryVerdict.UNSURE,
    ]
    assert opinions[0].source == MODEL and opinions[2].note == "glare"
    content = messages.requests[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and "1 × Metformin 500 mg" in content[1]["text"]
    assert messages.requests[0]["output_config"] == {"effort": "medium"}


def test_every_finding_has_a_description():
    for kind in FindingKind:
        assert describe_finding(finding(kind, medication_id=METFORMIN, expected=2, observed=1, count=2), CATALOG)
    for kind in PackFindingKind:
        assert describe_pack_finding(PackFinding(kind=kind, usable=1, required=3, count=2))
    lines = describe_result(result_with().result, LAYOUT, CATALOG)
    assert len(lines) == 1 + len(LAYOUT.all_compartments) and "identity not checked automatically" in lines[1]
