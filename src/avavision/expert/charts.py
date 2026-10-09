"""Medication charts (packing sheets, medication lists) and turning them into draft pack profiles.

The language model only reads the chart into :class:`MedicationChart`. Everything after that is
deterministic: matching each line to the catalog and placing doses in the pack grid. The result is a
*draft*; nothing is used for checking until a pharmacist has reviewed and saved it as a profile.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field

from ..core.models import Catalog, ExpectedItem, MedicationID, PackLayout, PackProfile, cell


class DoseTime(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    BEDTIME = "bedtime"


class Weekday(StrEnum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


WEEK = list(Weekday)
#: Position of each dose time: rows from the top in ``WEEKLY_7X4``, columns from the left in the portrait layout.
DOSE_SLOTS = {DoseTime.BREAKFAST: 0, DoseTime.LUNCH: 1, DoseTime.DINNER: 2, DoseTime.BEDTIME: 3}


class ChartDose(BaseModel):
    time: DoseTime
    #: Units per dose, e.g. 1, 2 or 0.5 for half a tablet.
    quantity: float


class ChartLine(BaseModel):
    name: str = Field(description="Medication name exactly as printed (brand or generic).")
    generic_name: str | None = Field(None, description="Active ingredient when printed or unambiguous.")
    strength: str | None = Field(None, description="Strength as printed, e.g. '500 mg'.")
    form: str | None = Field(None, description="tablet, capsule, ...")
    doses: list[ChartDose]
    days: list[Weekday] = Field(description="Days the doses are given; all seven when taken every day.")
    schedule_note: str | None = Field(None, description="Any other schedule wording, e.g. 'alternate days'.")
    appearance: str | None = Field(None, description="Printed tablet description, if any.")
    in_pack: bool = Field(description="False when the chart says the item is not packed (PRN, liquids, ...).")
    unclear: list[str] = Field(description="Every part of this line that could not be read with certainty.")


class MedicationChart(BaseModel):
    first_day: Weekday | None = Field(None, description="Weekday of the pack's first column, if stated.")
    lines: list[ChartLine]
    warnings: list[str] = Field(description="Anything about the whole chart a pharmacist should know.")


class ImportIssueKind(StrEnum):
    UNSUPPORTED_LAYOUT = "unsupportedLayout"
    NOT_MATCHED = "notMatched"  # no catalog medication matches the line
    AMBIGUOUS_MATCH = "ambiguousMatch"  # several catalog medications match
    FRACTIONAL_DOSE = "fractionalDose"  # half tablets are not placed automatically
    UNKNOWN_START_DAY = "unknownStartDay"  # a weekly schedule needs the pack's first weekday
    UNCLEAR = "unclear"  # the reader was not certain about part of the line
    NO_DOSES = "noDoses"
    SCHEDULE_NOTE = "scheduleNote"  # wording the grid cannot express; check by hand
    CHART_WARNING = "chartWarning"


class ImportIssue(BaseModel):
    kind: ImportIssueKind
    line: int | None = None  # index into ``chart.lines``
    detail: str = ""


class ProfileDraft(BaseModel):
    profile: PackProfile
    chart: MedicationChart
    matches: dict[int, MedicationID]
    issues: list[ImportIssue]
    not_packed: list[int]

    @property
    def placed_lines(self) -> list[int]:
        blocked = {i.line for i in self.issues if i.kind in _BLOCKING}
        return [k for k in self.matches if k not in blocked]


_BLOCKING = {
    ImportIssueKind.NOT_MATCHED,
    ImportIssueKind.AMBIGUOUS_MATCH,
    ImportIssueKind.UNKNOWN_START_DAY,
}


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9.]+", "", (text or "").lower())


def match_line(line: ChartLine, catalog: Catalog) -> list[MedicationID]:
    """Catalog medications with the line's name (or generic name) and, when both state one, its strength."""
    names = {n for n in (_norm(line.name), _norm(line.generic_name)) if n}
    strength = _norm(line.strength)
    return [
        m.id
        for m in catalog.medications
        if _norm(m.name) in names and (not strength or not m.strength or _norm(m.strength) == strength)
    ]


def _weekday(label: str) -> Weekday | None:
    text = label.strip().lower()
    return next((d for d in WEEK if len(text) >= 3 and d.value.startswith(text[:3])), None)


def _weekly_grid(layout: PackLayout) -> tuple[bool, list[Weekday] | None] | None:
    """For a weekly 7 × 4 layout: whether the days run down the rows (portrait card), and the weekday of each day
    position when the card prints them (Mon … Sun). ``None`` for any other layout."""
    if layout.rows == len(DOSE_SLOTS) and layout.columns == len(WEEK):
        days_down, labels = False, layout.column_labels
    elif layout.rows == len(WEEK) and layout.columns == len(DOSE_SLOTS):
        days_down, labels = True, layout.row_labels
    else:
        return None
    printed = [_weekday(label) for label in labels]
    return days_down, (printed if None not in printed and len(set(printed)) == len(WEEK) else None)


def chart_to_profile(
    chart: MedicationChart,
    catalog: Catalog,
    layout: PackLayout,
    reference: str,
    overrides: dict[int, MedicationID] | None = None,
) -> ProfileDraft:
    """Places every confidently matched line in a weekly 7 × 4 pack: days across and dose times down, or (portrait)
    days down and dose times across. A card that prints its weekdays fixes where each day goes; otherwise the
    chart's first day does.

    ``overrides`` maps line indices to the catalog medication a pharmacist chose for them.
    """
    overrides = overrides or {}
    profile = PackProfile.empty(reference, layout)
    issues: list[ImportIssue] = [ImportIssue(kind=ImportIssueKind.CHART_WARNING, detail=w) for w in chart.warnings]
    matches: dict[int, MedicationID] = {}
    not_packed: list[int] = []
    grid = _weekly_grid(layout)
    if grid is None:
        issues.append(ImportIssue(kind=ImportIssueKind.UNSUPPORTED_LAYOUT, detail=layout.id))
        return ProfileDraft(profile=profile, chart=chart, matches={}, issues=issues, not_packed=[])
    days_down, printed = grid

    placed: dict[tuple[int, int], dict[MedicationID, int]] = {}
    for k, line in enumerate(chart.lines):
        if not line.in_pack:
            not_packed.append(k)
            continue
        issues.extend(ImportIssue(kind=ImportIssueKind.UNCLEAR, line=k, detail=u) for u in line.unclear)
        if line.schedule_note:
            issues.append(ImportIssue(kind=ImportIssueKind.SCHEDULE_NOTE, line=k, detail=line.schedule_note))
        if k in overrides and overrides[k] in catalog:
            medication = overrides[k]
        else:
            found = match_line(line, catalog)
            if len(found) != 1:
                kind = ImportIssueKind.NOT_MATCHED if not found else ImportIssueKind.AMBIGUOUS_MATCH
                issues.append(ImportIssue(kind=kind, line=k, detail=", ".join(found)))
                continue
            medication = found[0]
        matches[k] = medication
        days = sorted(set(line.days), key=WEEK.index)
        if not days or not line.doses:
            issues.append(ImportIssue(kind=ImportIssueKind.NO_DOSES, line=k))
            continue
        if len(days) < len(WEEK) and chart.first_day is None and printed is None:
            issues.append(ImportIssue(kind=ImportIssueKind.UNKNOWN_START_DAY, line=k))
            continue
        start = WEEK.index(chart.first_day) if chart.first_day else 0
        positions = [printed.index(d) if printed else (WEEK.index(d) - start) % len(WEEK) for d in days]
        for dose in line.doses:
            if dose.quantity <= 0:
                continue
            if dose.quantity != int(dose.quantity):
                issues.append(
                    ImportIssue(kind=ImportIssueKind.FRACTIONAL_DOSE, line=k, detail=f"{dose.time}: {dose.quantity}")
                )
                continue
            for day in positions:
                time = DOSE_SLOTS[dose.time]
                slot = placed.setdefault((day, time) if days_down else (time, day), {})
                slot[medication] = slot.get(medication, 0) + int(dose.quantity)

    for (row, column), items in placed.items():
        profile.set_items(
            cell(row, column), [ExpectedItem(medication_id=m, quantity=q) for m, q in sorted(items.items())]
        )
    return ProfileDraft(profile=profile, chart=chart, matches=matches, issues=issues, not_packed=not_packed)
