"""Model manifests and the release gate that decides what a model may do."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .evaluation import EvaluationReport
from .models import MedicationID
from .observation import PILL, LabelMeaning, MeaningKind

MANIFEST_SCHEMA_VERSION = 1


class ModelStage(StrEnum):
    DEVELOPMENT = "development"  # may count, never identifies
    VALIDATION = "validation"  # may count, never identifies
    RELEASED = "released"  # may identify medications its evaluation covers
    REVOKED = "revoked"  # must not be used


class ModelManifest(BaseModel):
    """Description of a detector artefact, checked before use."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    model_id: str
    version: str
    stage: ModelStage
    model_sha256: str
    #: Meaning of every output label, encoded like ``"pill"`` or ``"medication:<id>"``.
    labels: dict[str, str]
    #: Output class names in the model's index order; when empty, the order of ``labels``.
    classes: list[str] = []
    evaluation: EvaluationReport | None = None
    notes: str | None = None

    @property
    def class_names(self) -> list[str]:
        return self.classes or list(self.labels)

    def meaning(self, label: str) -> LabelMeaning:
        """Unknown labels are treated as generic pills: an unexpected class is never ignored."""
        return (LabelMeaning.parse(self.labels[label]) if label in self.labels else None) or PILL

    @property
    def medication_labels(self) -> set[MedicationID]:
        result = set()
        for text in self.labels.values():
            meaning = LabelMeaning.parse(text)
            if meaning and meaning.kind == MeaningKind.MEDICATION:
                result.add(meaning.medication_id)
        return result


class CapabilityKind(StrEnum):
    UNAVAILABLE = "unavailable"
    COUNT_ONLY = "countOnly"
    IDENTITY = "identity"


class ModelCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: CapabilityKind
    medications: frozenset[MedicationID] = frozenset()

    @property
    def can_count(self) -> bool:
        return self.kind != CapabilityKind.UNAVAILABLE

    def can_identify(self, medication: MedicationID) -> bool:
        return self.kind == CapabilityKind.IDENTITY and medication in self.medications

    def resolve(self, meaning: LabelMeaning) -> LabelMeaning:
        """Identity claims the capability does not cover become generic pills."""
        if meaning.kind == MeaningKind.MEDICATION and not self.can_identify(meaning.medication_id):
            return PILL
        return meaning


UNAVAILABLE = ModelCapability(kind=CapabilityKind.UNAVAILABLE)
COUNT_ONLY = ModelCapability(kind=CapabilityKind.COUNT_ONLY)


class ReleaseGatePolicy(BaseModel):
    minimum_samples: int = 1000
    minimum_error_samples: int = 600
    minimum_samples_per_medication: int = 50
    minimum_count_accuracy: float = 0.95
    minimum_identity_precision: float = 0.95
    minimum_identity_recall: float = 0.95
    maximum_false_acceptance_rate: float = 0.005


class BlockerKind(StrEnum):
    MODEL_FILE_MISSING = "modelFileMissing"
    MODEL_HASH_MISMATCH = "modelHashMismatch"
    UNSUPPORTED_MANIFEST_SCHEMA = "unsupportedManifestSchema"
    MODEL_REVOKED = "modelRevoked"
    NO_PILL_LABELS = "noPillLabels"
    STAGE_NOT_RELEASED = "stageNotReleased"
    EVALUATION_MISSING = "evaluationMissing"
    INSUFFICIENT_SAMPLES = "insufficientSamples"
    INSUFFICIENT_ERROR_SAMPLES = "insufficientErrorSamples"
    COUNT_ACCURACY_TOO_LOW = "countAccuracyTooLow"
    IDENTITY_PRECISION_TOO_LOW = "identityPrecisionTooLow"
    IDENTITY_RECALL_TOO_LOW = "identityRecallTooLow"
    FALSE_ACCEPTANCE_TOO_HIGH = "falseAcceptanceTooHigh"
    NO_MEDICATION_LABELS = "noMedicationLabels"
    MEDICATION_UNDERSAMPLED = "medicationUndersampled"
    MEDICATION_BELOW_THRESHOLD = "medicationBelowThreshold"


class GateBlocker(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: BlockerKind
    medication_id: MedicationID | None = None
    have: float | None = None
    need: float | None = None
    detail: str | None = None


class GateDecision(BaseModel):
    capability: ModelCapability
    blockers: list[GateBlocker] = Field(default_factory=list)


def evaluate_gate(
    manifest: ModelManifest, computed_sha256: str | None, policy: ReleaseGatePolicy | None = None
) -> GateDecision:
    """Decides what the model may do. ``computed_sha256`` is the hash of the artefact on disk."""
    policy = policy or ReleaseGatePolicy()
    if computed_sha256 is None:
        return GateDecision(capability=UNAVAILABLE, blockers=[GateBlocker(kind=BlockerKind.MODEL_FILE_MISSING)])
    if computed_sha256.lower() != manifest.model_sha256.lower():
        return GateDecision(capability=UNAVAILABLE, blockers=[GateBlocker(kind=BlockerKind.MODEL_HASH_MISMATCH)])
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        return GateDecision(
            capability=UNAVAILABLE,
            blockers=[GateBlocker(kind=BlockerKind.UNSUPPORTED_MANIFEST_SCHEMA, have=manifest.schema_version)],
        )
    if manifest.stage == ModelStage.REVOKED:
        return GateDecision(capability=UNAVAILABLE, blockers=[GateBlocker(kind=BlockerKind.MODEL_REVOKED)])
    counts_pills = any(
        (m := LabelMeaning.parse(t)) is not None and m.kind in (MeaningKind.PILL, MeaningKind.MEDICATION)
        for t in manifest.labels.values()
    )
    if not counts_pills:
        return GateDecision(capability=UNAVAILABLE, blockers=[GateBlocker(kind=BlockerKind.NO_PILL_LABELS)])

    b: list[GateBlocker] = []
    if manifest.stage != ModelStage.RELEASED:
        b.append(GateBlocker(kind=BlockerKind.STAGE_NOT_RELEASED, detail=manifest.stage.value))
    medications = manifest.medication_labels
    if not medications:
        b.append(GateBlocker(kind=BlockerKind.NO_MEDICATION_LABELS))
    report = manifest.evaluation
    if report is None:
        return GateDecision(capability=COUNT_ONLY, blockers=[*b, GateBlocker(kind=BlockerKind.EVALUATION_MISSING)])

    def below(kind, have, need):
        b.append(GateBlocker(kind=kind, have=have, need=need))

    if report.total_samples < policy.minimum_samples:
        below(BlockerKind.INSUFFICIENT_SAMPLES, report.total_samples, policy.minimum_samples)
    if report.error_samples < policy.minimum_error_samples:
        below(BlockerKind.INSUFFICIENT_ERROR_SAMPLES, report.error_samples, policy.minimum_error_samples)
    if report.count_accuracy < policy.minimum_count_accuracy:
        below(BlockerKind.COUNT_ACCURACY_TOO_LOW, report.count_accuracy, policy.minimum_count_accuracy)
    if report.identity_precision < policy.minimum_identity_precision:
        below(BlockerKind.IDENTITY_PRECISION_TOO_LOW, report.identity_precision, policy.minimum_identity_precision)
    if report.identity_recall < policy.minimum_identity_recall:
        below(BlockerKind.IDENTITY_RECALL_TOO_LOW, report.identity_recall, policy.minimum_identity_recall)
    if report.false_acceptance_rate > policy.maximum_false_acceptance_rate:
        below(BlockerKind.FALSE_ACCEPTANCE_TOO_HIGH, report.false_acceptance_rate, policy.maximum_false_acceptance_rate)
    if b:
        return GateDecision(capability=COUNT_ONLY, blockers=b)

    approved = set()
    for medication in sorted(medications):
        metrics = report.per_medication.get(medication)
        if metrics is None or metrics.samples < policy.minimum_samples_per_medication:
            b.append(
                GateBlocker(
                    kind=BlockerKind.MEDICATION_UNDERSAMPLED,
                    medication_id=medication,
                    have=metrics.samples if metrics else 0,
                    need=policy.minimum_samples_per_medication,
                )
            )
        elif metrics.precision < policy.minimum_identity_precision or metrics.recall < policy.minimum_identity_recall:
            b.append(GateBlocker(kind=BlockerKind.MEDICATION_BELOW_THRESHOLD, medication_id=medication))
        else:
            approved.add(medication)
    capability = (
        ModelCapability(kind=CapabilityKind.IDENTITY, medications=frozenset(approved)) if approved else COUNT_ONLY
    )
    return GateDecision(capability=capability, blockers=b)


class ActiveModel(BaseModel):
    """A detector ready for use, with the capability the gate granted."""

    manifest: ModelManifest
    decision: GateDecision

    @property
    def capability(self) -> ModelCapability:
        return self.decision.capability

    def meaning(self, label: str) -> LabelMeaning:
        return self.capability.resolve(self.manifest.meaning(label))


def builtin_pocket_segmenter() -> ActiveModel:
    """The training-free segmenter built into the station: counts only, never identifies."""
    manifest = ModelManifest(
        model_id="avavision-pocket-segmenter",
        version="1",
        stage=ModelStage.DEVELOPMENT,
        model_sha256="built-in",
        labels={"pill": "pill"},
        notes="Classical segmentation cross-checked across methods and frames",
    )
    return ActiveModel(manifest=manifest, decision=evaluate_gate(manifest, "built-in"))


class EmbedderManifest(BaseModel):
    """A bundled appearance-embedding model (ONNX). It only produces vectors, so it needs integrity, not the gate."""

    schema_version: int = 1
    embedder_id: str
    version: str
    model_sha256: str
    input_size: int = 224
    output_name: str = "embedding"
    base_model_license: str
    notes: str | None = None

    def is_intact(self, computed_sha256: str | None) -> bool:
        return (
            self.schema_version == 1
            and self.input_size > 0
            and computed_sha256 is not None
            and computed_sha256.lower() == self.model_sha256.lower()
        )
