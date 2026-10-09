import Foundation

/// AvaVision's learning brain: memory of pharmacist-confirmed pill appearances, learned thresholds,
/// and an earned track record per medication.
///
/// Safety contract (asymmetric trust):
/// - From day one the brain may only *raise* concerns (a pill that looks like an unexpected medication,
///   or like nothing it knows) — this can never make a result less safe.
/// - It may *grant* acceptance for a medication only once its track record, built exclusively from
///   pills a pharmacist inspected, passes the `TrustPolicy`; one wrong identification suspends that trust.
public struct BrainState: Codable, Hashable, Sendable {
  public static let currentSchemaVersion = 1

  public var schemaVersion: Int
  public private(set) var knowledge: KnowledgeBase
  public private(set) var identityPolicy: IdentityPolicy
  public var trustPolicy: TrustPolicy
  public private(set) var ledger: TrustLedger
  public private(set) var labellingQueue: [LabellingTask]
  public private(set) var lastCalibration: CalibrationReport?
  /// Exemplar count at the last calibration, used to decide when to recalibrate.
  public private(set) var exemplarsAtCalibration: Int

  public init(embedderID: EmbedderID, trustPolicy: TrustPolicy = .standard) {
    schemaVersion = Self.currentSchemaVersion
    knowledge = KnowledgeBase(embedderID: embedderID)
    identityPolicy = .uncalibrated
    self.trustPolicy = trustPolicy
    ledger = TrustLedger(embedderID: embedderID)
    labellingQueue = []
    lastCalibration = nil
    exemplarsAtCalibration = 0
  }

  public var embedderID: EmbedderID { knowledge.embedderID }

  public func classifier() -> IdentityClassifier {
    IdentityClassifier(knowledge: knowledge, policy: identityPolicy)
  }

  public var trustedMedications: Set<MedicationID> {
    ledger.trustedMedications(policy: trustPolicy)
  }

  /// Medications the brain knows well enough to notice a pill that is none of them.
  public var wellKnownMedications: Set<MedicationID> {
    guard identityPolicy.isCalibrated else { return [] }
    return knowledge.medications.filter {
      knowledge.count(for: $0) >= identityPolicy.minimumExemplars
        && knowledge.groupCount(for: $0) >= identityPolicy.minimumGroups
    }
  }

  public func trustStatus(of id: MedicationID) -> TrustStatus {
    ledger.status(of: id, policy: trustPolicy)
  }

  public struct LearningSummary: Hashable, Sendable {
    public var exemplarsAdded = 0
    public var duplicatesSkipped = 0
    public var observationsRecorded = 0
    public var tasksQueued = 0
    public var recalibration: CalibrationReport?
  }

  /// Applies a learning plan, then recalibrates if memory grew enough.
  @discardableResult
  public mutating func learn(_ plan: LearningPlan, at date: Date = Date()) -> LearningSummary {
    var summary = LearningSummary()
    for exemplar in plan.exemplars {
      switch knowledge.add(exemplar) {
      case .added, .addedReplacing: summary.exemplarsAdded += 1
      case .duplicate, .incompatibleEmbedder: summary.duplicatesSkipped += 1
      }
    }
    for observation in plan.observations {
      ledger.observe(observation.decision, truth: observation.truth, at: date)
      summary.observationsRecorded += 1
    }
    labellingQueue.append(contentsOf: plan.labellingTasks)
    labellingQueue.sort { ($0.priority, $0.createdAt) > ($1.priority, $1.createdAt) }
    summary.tasksQueued = plan.labellingTasks.count
    if needsRecalibration {
      summary.recalibration = recalibrate(at: date)
    }
    return summary
  }

  /// Teaches a medication from a photo of pills the pharmacist knows to be that medication.
  @discardableResult
  public mutating func teach(
    _ medication: MedicationID, embeddings: [(embedding: Embedding, cropFile: String?)], groupID: UUID = UUID(),
    at date: Date = Date()
  ) -> LearningSummary {
    var plan = LearningPlan()
    plan.exemplars = embeddings.map {
      PillExemplar(
        medicationID: medication, embedding: $0.embedding, source: .teaching, groupID: groupID, createdAt: date,
        cropFile: $0.cropFile)
    }
    return learn(plan, at: date)
  }

  public enum TaskError: Error, Hashable, Sendable {
    case notFound
  }

  @discardableResult
  public mutating func resolveTask(_ id: UUID, labels: [UUID: MedicationID], at date: Date = Date()) throws
    -> LearningSummary
  {
    guard let task = labellingQueue.first(where: { $0.id == id }) else { throw TaskError.notFound }
    let plan = try task.resolve(labels: labels, at: date)
    labellingQueue.removeAll { $0.id == id }
    return learn(plan, at: date)
  }

  public mutating func discardTask(_ id: UUID) {
    labellingQueue.removeAll { $0.id == id }
  }

  /// Forgets everything about a medication (for example if it was taught with the wrong pills).
  public mutating func forget(medication id: MedicationID) {
    knowledge.forget(medication: id)
    ledger.forget(medication: id)
  }

  /// Memory grew by a quarter, and by at least ten pills, since the last calibration.
  public var needsRecalibration: Bool {
    knowledge.exemplars.count >= exemplarsAtCalibration + max(10, exemplarsAtCalibration / 4)
  }

  /// Re-learns thresholds from memory. If they change, every medication must rebuild its streak.
  @discardableResult
  public mutating func recalibrate(calibrator: IdentityCalibrator = IdentityCalibrator(), at date: Date = Date())
    -> CalibrationReport
  {
    let (report, policy) = calibrator.calibrate(knowledge: knowledge, base: identityPolicy, at: date)
    lastCalibration = report
    exemplarsAtCalibration = knowledge.exemplars.count
    if let policy,
      policy.acceptSimilarity != identityPolicy.acceptSimilarity
        || policy.minimumMargin != identityPolicy.minimumMargin
    {
      identityPolicy = policy
      ledger.resetStreaks()
    }
    return report
  }

  /// Moves the brain to a new embedder (for example after a model upgrade): every exemplar must be
  /// re-embedded from its stored crop. Exemplars without a new embedding are dropped; thresholds and
  /// track records start again because the space changed.
  public func migrated(to embedderID: EmbedderID, embeddings: [UUID: Embedding]) -> BrainState {
    var next = BrainState(embedderID: embedderID, trustPolicy: trustPolicy)
    var plan = LearningPlan()
    for exemplar in knowledge.exemplars {
      guard let embedding = embeddings[exemplar.id], embedding.embedderID == embedderID else { continue }
      var moved = exemplar
      moved.embedding = embedding
      plan.exemplars.append(moved)
    }
    next.learn(plan)
    return next
  }

  public var summary: BrainSummary {
    BrainSummary(
      embedderID: embedderID, policyVersion: identityPolicy.version, exemplarCount: knowledge.exemplars.count,
      knownMedications: knowledge.medications.count, trustedMedications: trustedMedications.sorted())
  }
}

/// Brain state captured in each audit record.
public struct BrainSummary: Codable, Hashable, Sendable {
  public var embedderID: EmbedderID
  public var policyVersion: Int
  public var exemplarCount: Int
  public var knownMedications: Int
  public var trustedMedications: [MedicationID]

  public init(
    embedderID: EmbedderID, policyVersion: Int, exemplarCount: Int, knownMedications: Int,
    trustedMedications: [MedicationID]
  ) {
    self.embedderID = embedderID
    self.policyVersion = policyVersion
    self.exemplarCount = exemplarCount
    self.knownMedications = knownMedications
    self.trustedMedications = trustedMedications
  }
}
