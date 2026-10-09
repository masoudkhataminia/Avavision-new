"""Domain model: medications, pack layouts and pack profiles."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from functools import total_ordering

from pydantic import BaseModel, ConfigDict, Field

from .geometry import Point, Rect

MedicationID = str


@total_ordering
class CompartmentIndex(BaseModel):
    """Zero-based position of one blister cell."""

    model_config = ConfigDict(frozen=True)

    row: int
    column: int

    def __lt__(self, other: CompartmentIndex) -> bool:
        return (self.row, self.column) < (other.row, other.column)

    def __str__(self) -> str:
        return f"r{self.row}c{self.column}"


def cell(row: int, column: int) -> CompartmentIndex:
    return CompartmentIndex(row=row, column=column)


class Appearance(BaseModel):
    colour: str | None = None
    shape: str | None = None
    imprint: str | None = None


class Medication(BaseModel):
    """A product the pharmacy packs."""

    id: MedicationID
    name: str
    strength: str = ""
    appearance: Appearance = Field(default_factory=Appearance)

    @property
    def display_name(self) -> str:
        return f"{self.name} {self.strength}".strip()


class Catalog(BaseModel):
    medications: list[Medication] = Field(default_factory=list)

    def get(self, medication_id: MedicationID) -> Medication | None:
        return next((m for m in self.medications if m.id == medication_id), None)

    def __contains__(self, medication_id: MedicationID) -> bool:
        return self.get(medication_id) is not None

    def upsert(self, medication: Medication) -> None:
        self.medications = [m for m in self.medications if m.id != medication.id] + [medication]
        self.medications.sort(key=lambda m: m.name.lower())

    def remove(self, medication_id: MedicationID) -> None:
        self.medications = [m for m in self.medications if m.id != medication_id]


class CellLocationKind(StrEnum):
    INSIDE = "inside"
    BORDER = "border"
    OUTSIDE = "outside"


class CellLocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: CellLocationKind
    compartments: tuple[CompartmentIndex, ...] = ()


class LayoutError(StrEnum):
    EMPTY_IDENTIFIER = "emptyIdentifier"
    INVALID_GRID_SIZE = "invalidGridSize"
    LABEL_COUNT_MISMATCH = "labelCountMismatch"
    INVALID_PHYSICAL_SIZE = "invalidPhysicalSize"
    GRID_REGION_OUTSIDE_PACK = "gridRegionOutsidePack"
    INVALID_BORDER_BAND = "invalidBorderBand"


class PackLayout(BaseModel):
    """Geometry of a multi-dose pack in normalized pack coordinates (0…1 across the registered card)."""

    id: str
    display_name: str
    rows: int
    columns: int
    row_labels: list[str]
    column_labels: list[str]
    width_mm: float
    height_mm: float
    grid_region: Rect
    #: Fraction (0…0.5) of a cell near its edges where ownership is ambiguous.
    border_band: float
    #: Only a layout measured against real packs may produce accepted results.
    is_calibrated: bool

    @property
    def all_compartments(self) -> list[CompartmentIndex]:
        return [cell(r, c) for r in range(max(self.rows, 0)) for c in range(max(self.columns, 0))]

    @property
    def aspect_ratio(self) -> float:
        return self.width_mm / self.height_mm

    def contains(self, index: CompartmentIndex) -> bool:
        return 0 <= index.row < self.rows and 0 <= index.column < self.columns

    def label(self, index: CompartmentIndex) -> str:
        column = self.column_labels[index.column] if index.column < len(self.column_labels) else f"C{index.column + 1}"
        row = self.row_labels[index.row] if index.row < len(self.row_labels) else f"R{index.row + 1}"
        return f"{column} · {row}"

    def cell_rect(self, index: CompartmentIndex) -> Rect:
        w = self.grid_region.width / self.columns
        h = self.grid_region.height / self.rows
        return Rect(x=self.grid_region.x + index.column * w, y=self.grid_region.y + index.row * h, width=w, height=h)

    def locate(self, p: Point) -> CellLocation:
        """Which compartment owns a point; border bands (including the outer edge) are ambiguous."""
        g = self.grid_region
        if not g.contains(p):
            return CellLocation(kind=CellLocationKind.OUTSIDE)
        fx = (p.x - g.x) / g.width * self.columns
        fy = (p.y - g.y) / g.height * self.rows
        column = min(self.columns - 1, max(0, int(fx)))
        row = min(self.rows - 1, max(0, int(fy)))
        u, v = fx - column, fy - row
        columns, rows, outer = [column], [row], False
        if u < self.border_band:
            if column > 0:
                columns.append(column - 1)
            else:
                outer = True
        elif u > 1 - self.border_band:
            if column < self.columns - 1:
                columns.append(column + 1)
            else:
                outer = True
        if v < self.border_band:
            if row > 0:
                rows.append(row - 1)
            else:
                outer = True
        elif v > 1 - self.border_band:
            if row < self.rows - 1:
                rows.append(row + 1)
            else:
                outer = True
        if len(columns) == 1 and len(rows) == 1 and not outer:
            return CellLocation(kind=CellLocationKind.INSIDE, compartments=(cell(row, column),))
        candidates = sorted(cell(r, c) for r in rows for c in columns)
        return CellLocation(kind=CellLocationKind.BORDER, compartments=tuple(candidates))

    def validation_errors(self) -> list[LayoutError]:
        errors = []
        if not self.id.strip():
            errors.append(LayoutError.EMPTY_IDENTIFIER)
        if self.rows < 1 or self.columns < 1:
            errors.append(LayoutError.INVALID_GRID_SIZE)
        if len(self.row_labels) != self.rows or len(self.column_labels) != self.columns:
            errors.append(LayoutError.LABEL_COUNT_MISMATCH)
        if not (self.width_mm > 0 and self.height_mm > 0):
            errors.append(LayoutError.INVALID_PHYSICAL_SIZE)
        g = self.grid_region
        if not (g.width > 0 and g.height > 0 and g.x >= 0 and g.y >= 0 and g.max_x <= 1 and g.max_y <= 1):
            errors.append(LayoutError.GRID_REGION_OUTSIDE_PACK)
        if not (0 <= self.border_band < 0.5):
            errors.append(LayoutError.INVALID_BORDER_BAND)
        return errors


#: Weekly 7-day × 4-dose-time pack. Dimensions are placeholders until measured, so it ships uncalibrated.
WEEKLY_7X4 = PackLayout(
    id="weekly-7x4",
    display_name="Weekly pack · 7 days × 4 times",
    rows=4,
    columns=7,
    row_labels=["Morning", "Midday", "Evening", "Bedtime"],
    column_labels=[f"Day {d}" for d in range(1, 8)],
    width_mm=250,
    height_mm=170,
    grid_region=Rect(x=0.06, y=0.12, width=0.88, height=0.8),
    border_band=0.12,
    is_calibrated=False,
)

#: Portrait weekly card with the days down the side and the dose times across (as on many pharmacy-branded
#: DAA cards). Grid position fitted to a real card's outline (D-132); size not yet measured, so uncalibrated.
WEEKLY_7X4_PORTRAIT = PackLayout(
    id="weekly-7x4-portrait",
    display_name="Weekly pack, portrait · 7 days down × 4 times across",
    rows=7,
    columns=4,
    row_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    column_labels=["Morning", "Noon", "Evening", "Bedtime"],
    width_mm=240,
    height_mm=310,
    grid_region=Rect(x=0.09, y=0.06, width=0.84, height=0.87),
    border_band=0.08,
    is_calibrated=False,
)

BUILTIN_LAYOUTS = (WEEKLY_7X4, WEEKLY_7X4_PORTRAIT)


class ExpectedItem(BaseModel):
    medication_id: MedicationID
    quantity: int


class CompartmentExpectation(BaseModel):
    """Everything a compartment must contain; an empty list means it must be empty."""

    compartment: CompartmentIndex
    items: list[ExpectedItem]

    @property
    def total_quantity(self) -> int:
        return sum(i.quantity for i in self.items)

    @property
    def quantities(self) -> dict[MedicationID, int]:
        result: dict[MedicationID, int] = {}
        for item in self.items:
            result[item.medication_id] = result.get(item.medication_id, 0) + item.quantity
        return result


class ProfileIssueKind(StrEnum):
    EMPTY_REFERENCE = "emptyReference"
    LAYOUT_MISMATCH = "layoutMismatch"
    COMPARTMENT_OUTSIDE_LAYOUT = "compartmentOutsideLayout"
    DUPLICATE_COMPARTMENT = "duplicateCompartment"
    UNSPECIFIED_COMPARTMENT = "unspecifiedCompartment"
    UNKNOWN_MEDICATION = "unknownMedication"
    INVALID_QUANTITY = "invalidQuantity"


class ProfileIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ProfileIssueKind
    compartment: CompartmentIndex | None = None
    medication_id: MedicationID | None = None
    detail: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


class PackProfile(BaseModel):
    """The expected content of one physical pack."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    reference: str
    layout_id: str
    compartments: list[CompartmentExpectation]
    #: Code printed on the pack's header card (QR or barcode), when the packing software prints one.
    barcode: str | None = None
    created_at: datetime = Field(default_factory=_now)

    @classmethod
    def empty(cls, reference: str, layout: PackLayout) -> PackProfile:
        return cls(
            reference=reference,
            layout_id=layout.id,
            compartments=[CompartmentExpectation(compartment=i, items=[]) for i in layout.all_compartments],
        )

    def expectation(self, index: CompartmentIndex) -> CompartmentExpectation | None:
        return next((c for c in self.compartments if c.compartment == index), None)

    def set_items(self, index: CompartmentIndex, items: list[ExpectedItem]) -> None:
        self.compartments = [c for c in self.compartments if c.compartment != index]
        self.compartments.append(CompartmentExpectation(compartment=index, items=items))
        self.compartments.sort(key=lambda c: c.compartment)

    @property
    def total_doses(self) -> int:
        return sum(c.total_quantity for c in self.compartments)

    def issues(self, layout: PackLayout, catalog: Catalog) -> list[ProfileIssue]:
        """Every reason the profile cannot be trusted as the expected state of the pack."""
        issues: list[ProfileIssue] = []
        if not self.reference.strip():
            issues.append(ProfileIssue(kind=ProfileIssueKind.EMPTY_REFERENCE))
        if self.layout_id != layout.id:
            issues.append(ProfileIssue(kind=ProfileIssueKind.LAYOUT_MISMATCH, detail=self.layout_id))
        seen: set[CompartmentIndex] = set()
        unknown: set[MedicationID] = set()
        for expectation in self.compartments:
            index = expectation.compartment
            if not layout.contains(index):
                issues.append(ProfileIssue(kind=ProfileIssueKind.COMPARTMENT_OUTSIDE_LAYOUT, compartment=index))
            if index in seen:
                issues.append(ProfileIssue(kind=ProfileIssueKind.DUPLICATE_COMPARTMENT, compartment=index))
            seen.add(index)
            for item in expectation.items:
                if item.quantity < 1:
                    issues.append(
                        ProfileIssue(
                            kind=ProfileIssueKind.INVALID_QUANTITY, compartment=index, medication_id=item.medication_id
                        )
                    )
                if item.medication_id not in catalog and item.medication_id not in unknown:
                    unknown.add(item.medication_id)
                    issues.append(
                        ProfileIssue(kind=ProfileIssueKind.UNKNOWN_MEDICATION, medication_id=item.medication_id)
                    )
        for index in layout.all_compartments:
            if index not in seen:
                issues.append(ProfileIssue(kind=ProfileIssueKind.UNSPECIFIED_COMPARTMENT, compartment=index))
        return issues
