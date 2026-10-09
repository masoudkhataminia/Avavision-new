"""The expert language-model layer, backed by Claude.

Three jobs, none of them in the acceptance path:

- ``read_chart``: medication chart (PDF, photo or text) → :class:`MedicationChart` → draft profile.
- ``explain``: a verification result → a short summary and inspection steps for the pharmacist.
- ``review_compartment``: an independent look at one compartment → :class:`AdvisoryOpinion`, which can only
  escalate (see ``core.advisory``).

Credentials come from the environment (``ANTHROPIC_API_KEY``) or an ``ant auth login`` profile. Server-side
refusal fallbacks are on by default, so a request a safety classifier declines is retried on the model
Anthropic recommends instead of failing.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, Literal, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel

from ..core.advisory import AdvisoryOpinion, AdvisoryVerdict
from ..core.engine import VerificationResult
from ..core.models import Catalog, CompartmentIndex, PackLayout, PackProfile
from .charts import MedicationChart
from .describe import describe_result

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

Effort = Literal["low", "medium", "high", "xhigh", "max"]
T = TypeVar("T", bound=BaseModel)


class ExpertSettings(BaseModel):
    model: str = MODEL
    #: Retry requests a safety classifier declines on Anthropic's recommended fallback model.
    fallbacks: bool = True
    chart_effort: Effort = "high"  # accuracy matters most; runs once per profile
    explain_effort: Effort = "low"  # fast; the facts come from the engine
    review_effort: Effort = "medium"
    max_tokens: int = 16000
    timeout_seconds: float = 180.0


class ExpertError(RuntimeError):
    pass


class ExpertUnavailable(ExpertError):
    """No credentials, no connection or an API error. The station keeps working without the expert."""


class ExpertRefused(ExpertError):
    def __init__(self, category: str | None, explanation: str | None):
        super().__init__(f"the request was declined ({category or 'unspecified'})")
        self.category = category
        self.explanation = explanation


class ExpertIncomplete(ExpertError):
    """The answer was cut off or did not match the expected structure."""


class ExpertCall(BaseModel):
    requested_model: str
    served_model: str
    fallback_used: bool
    input_tokens: int
    output_tokens: int
    seconds: float
    request_id: str | None = None


@dataclass
class ExpertAnswer(Generic[T]):
    value: T
    call: ExpertCall


class Explanation(BaseModel):
    summary: str
    steps: list[str]


class Match(StrEnum):
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class CompartmentReading(BaseModel):
    pill_count: int | None
    matches_expected: Match
    differences: list[str]
    note: str


CHART_SYSTEM = """\
You read Australian pharmacy medication charts and dose administration aid (Webster-pak) packing sheets for a \
pharmacist who is setting up a pack check. Transcribe what the document says; never infer a dose, strength or \
schedule that is not printed.

- One line per medication entry, in the order printed. Copy the name and strength exactly as printed.
- Dose times: breakfast (morning), lunch (midday), dinner (evening, tea), bedtime (night).
- quantity is the number of units per dose time, e.g. 0.5 for half a tablet.
- days lists every weekday the medication is given; all seven when it is given daily.
- schedule_note holds any schedule the days and doses cannot express: alternate days, weekly, tapering or \
variable doses, "as directed", start or stop dates.
- in_pack is false for entries the chart marks as not packed and for anything that cannot go in a blister pack \
(PRN, liquids, inhalers, injections, patches, creams, refrigerated items).
- unclear lists every field you are not certain about: illegible, cut off, handwritten changes, conflicting \
entries. Leave a field empty rather than guessing.
- warnings: anything about the whole document a pharmacist should know, such as crossed-out lines, more than \
one person's chart or missing pages.
- first_day only when the document states the weekday the pack starts.

Do not transcribe names, addresses or other personal details."""

EXPLAIN_SYSTEM = """\
You explain the result of an automated Webster-pak check to the pharmacist who inspects and signs off the pack. \
The findings come from the verification engine and are final: do not add, remove, soften or second-guess any of \
them, and never call the pack safe or correct. Summarise what needs attention, then list concrete inspection \
steps, most important first. Write in {language}. Be brief."""

REVIEW_SYSTEM = """\
You give an independent second opinion on one compartment of a medication blister pack, photographed from above \
and perspective-corrected. The compartment in question fills the centre of the image; the edges show parts of \
neighbouring compartments, which you must ignore.

Count the solid oral doses (tablets and capsules; each piece of a broken tablet counts separately) inside the \
centre compartment and say whether the contents are consistent with the expected list. Answer "unsure" whenever \
the image does not let you tell: glare, blur, overlapping tablets or an appearance you cannot match. Your \
opinion is advisory; the pharmacist decides."""

LANGUAGES = {"fa": "Persian (Farsi)", "en": "English"}
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_MAX_IMAGE_SIDE = 3000
_MAX_IMAGE_BYTES = 4_500_000


def _image_block(data: bytes, media_type: str) -> dict:
    if len(data) > _MAX_IMAGE_BYTES or media_type not in IMAGE_TYPES.values():
        data, media_type = _reencode(data), "image/jpeg"
    else:
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is not None and max(decoded.shape[:2]) > _MAX_IMAGE_SIDE:
            data, media_type = _reencode(data), "image/jpeg"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode()},
    }


def _reencode(data: bytes) -> bytes:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable image")
    return encode_jpeg(image)


def encode_jpeg(image: np.ndarray, max_side: int = _MAX_IMAGE_SIDE, quality: int = 90) -> bytes:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the image")
    return buffer.tobytes()


def chart_blocks(data: bytes, media_type: str) -> list[dict]:
    """Message content for one chart document."""
    if media_type == "application/pdf":
        encoded = base64.standard_b64encode(data).decode()
        return [{"type": "document", "source": {"type": "base64", "media_type": media_type, "data": encoded}}]
    if media_type.startswith("image/"):
        return [_image_block(data, media_type)]
    if media_type.startswith("text/"):
        return [{"type": "text", "text": f"<chart>\n{data.decode('utf-8', errors='replace')}\n</chart>"}]
    raise ValueError(f"unsupported chart type {media_type}")


def media_type_of(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in IMAGE_TYPES:
        return IMAGE_TYPES[suffix]
    if suffix in (".txt", ".csv", ".tsv"):
        return "text/plain"
    raise ValueError(f"unsupported chart file {path}")


class ClaudeExpert:
    def __init__(self, settings: ExpertSettings | None = None, client=None, api_key: str | None = None):
        self.settings = settings or ExpertSettings()
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        if self._client is None:
            import anthropic

            try:
                self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
            except anthropic.AnthropicError as error:
                raise ExpertUnavailable(f"Claude is not configured: {error}") from error
        return self._client

    # ------------------------------------------------------------------ jobs

    def read_chart(self, data: bytes, media_type: str) -> ExpertAnswer[MedicationChart]:
        content = [*chart_blocks(data, media_type), {"type": "text", "text": "Transcribe this medication chart."}]
        return self._ask(CHART_SYSTEM, content, MedicationChart, self.settings.chart_effort)

    def explain(
        self,
        result: VerificationResult,
        layout: PackLayout,
        catalog: Catalog,
        language: str = "fa",
    ) -> ExpertAnswer[Explanation]:
        facts = "\n".join(describe_result(result, layout, catalog))
        system = EXPLAIN_SYSTEM.format(language=LANGUAGES.get(language, language))
        content = [{"type": "text", "text": f"<findings>\n{facts}\n</findings>"}]
        return self._ask(system, content, Explanation, self.settings.explain_effort)

    def review_compartment(
        self,
        image: np.ndarray,
        index: CompartmentIndex,
        layout: PackLayout,
        profile: PackProfile,
        catalog: Catalog,
    ) -> ExpertAnswer[AdvisoryOpinion]:
        expectation = profile.expectation(index)
        items = expectation.items if expectation else []
        lines = []
        for item in items:
            medication = catalog.get(item.medication_id)
            name = medication.display_name if medication else item.medication_id
            looks = medication.appearance if medication else None
            described = ", ".join(v for v in (looks.colour, looks.shape, looks.imprint) if v) if looks else ""
            lines.append(f"- {item.quantity} × {name}" + (f" ({described})" if described else ""))
        expected = "\n".join(lines) if lines else "Nothing: the compartment should be empty."
        content = [
            _image_block(encode_jpeg(image, max_side=1024), "image/jpeg"),
            {"type": "text", "text": f"Compartment: {layout.label(index)}\nExpected:\n{expected}"},
        ]
        answer = self._ask(REVIEW_SYSTEM, content, CompartmentReading, self.settings.review_effort)
        total = sum(i.quantity for i in items)
        return ExpertAnswer(value=opinion_from_reading(answer.value, index, total, answer.call), call=answer.call)

    # ------------------------------------------------------------------ transport

    def _ask(self, system: str, content: list[dict], output_format: type[T], effort: Effort) -> ExpertAnswer[T]:
        import anthropic

        settings = self.settings
        request = {
            "model": settings.model,
            "max_tokens": settings.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_format": output_format,
            "output_config": {"effort": effort},
            "timeout": settings.timeout_seconds,
        }
        if settings.fallbacks:
            request |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        start = time.perf_counter()
        try:
            response = self.client.beta.messages.parse(**request)
        except anthropic.AuthenticationError as error:
            raise ExpertUnavailable("the Claude API key is missing or invalid") from error
        except anthropic.APIConnectionError as error:
            raise ExpertUnavailable("no connection to the Claude API") from error
        except anthropic.APIStatusError as error:
            raise ExpertUnavailable(f"Claude API error {error.status_code}: {error.message}") from error
        except anthropic.AnthropicError as error:
            raise ExpertUnavailable(str(error)) from error
        if response.stop_reason == "refusal":
            details = response.stop_details
            raise ExpertRefused(getattr(details, "category", None), getattr(details, "explanation", None))
        if response.stop_reason == "max_tokens":
            raise ExpertIncomplete("the answer was cut off")
        parsed = response.parsed_output
        if parsed is None:
            raise ExpertIncomplete("the answer did not match the expected structure")
        iterations = getattr(response.usage, "iterations", None) or []
        call = ExpertCall(
            requested_model=settings.model,
            served_model=response.model,
            fallback_used=any(getattr(i, "type", None) == "fallback_message" for i in iterations),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            seconds=round(time.perf_counter() - start, 2),
            request_id=getattr(response, "_request_id", None),
        )
        return ExpertAnswer(value=parsed, call=call)


def opinion_from_reading(
    reading: CompartmentReading, index: CompartmentIndex, expected_total: int, call: ExpertCall
) -> AdvisoryOpinion:
    """A "yes" with a count that contradicts the profile is treated as a disagreement."""
    verdict = {Match.YES: AdvisoryVerdict.AGREES, Match.NO: AdvisoryVerdict.DISAGREES}.get(
        reading.matches_expected, AdvisoryVerdict.UNSURE
    )
    if reading.pill_count is not None and reading.pill_count != expected_total:
        verdict = AdvisoryVerdict.DISAGREES
    note = "; ".join([*reading.differences, reading.note] if reading.note else reading.differences)
    return AdvisoryOpinion(
        compartment=index, verdict=verdict, observed_count=reading.pill_count, note=note, source=call.served_model
    )
