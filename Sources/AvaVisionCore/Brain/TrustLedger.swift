import Foundation

/// When the brain's identification of a medication may be relied on without individual inspection.
public struct TrustPolicy: Codable, Hashable, Sendable {
  /// Pharmacist-checked identifications of the medication needed before trust.
  public var minimumPredictions: Int
  /// Wilson 95% lower bound on precision (named as this medication and really this medication).
  public var minimumPrecisionLowerBound: Double
  /// Consecutive correct identifications needed since the last error or threshold change.
  public var minimumStreak: Int

  public init(minimumPredictions: Int, minimumPrecisionLowerBound: Double, minimumStreak: Int) {
    self.minimumPredictions = minimumPredictions
    self.minimumPrecisionLowerBound = minimumPrecisionLowerBound
    self.minimumStreak = minimumStreak
  }

  public static let standard = TrustPolicy(
    minimumPredictions: 100, minimumPrecisionLowerBound: 0.97, minimumStreak: 50)
}

/// Track record of the brain for one medication, built only from pills a pharmacist inspected.
public struct MedicationTrackRecord: Codable, Hashable, Sendable {
  /// Named as this medication, and it was.
  public var correct = 0
  /// Named as this medication, but it was something else.
  public var falseIdentifications = 0
  /// It was this medication, but the brain named another one.
  public var confusions = 0
  /// It was this medication, and the brain declined to name anything.
  public var abstentions = 0
  /// Correct identifications since the last false identification or threshold change.
  public var streak = 0
  public var lastFalseIdentificationAt: Date?

  public init() {}

  public var namedCount: Int { correct + falseIdentifications }

  public var precisionLowerBound: Double {
    Statistics.wilsonLowerBound(successes: correct, trials: namedCount)
  }
}

public enum TrustStatus: Hashable, Sendable {
  /// Gathering evidence; `progress` is 0…1 towards trust.
  case learning(progress: Double)
  case trusted
  /// Was wrong recently; must rebuild a clean streak.
  case suspended
}

/// Earned, revocable trust per medication. A single wrong identification suspends trust at once.
public struct TrustLedger: Codable, Hashable, Sendable {
  public let embedderID: EmbedderID
  public private(set) var records: [MedicationID: MedicationTrackRecord]

  public init(embedderID: EmbedderID, records: [MedicationID: MedicationTrackRecord] = [:]) {
    self.embedderID = embedderID
    self.records = records
  }

  public func record(for id: MedicationID) -> MedicationTrackRecord {
    records[id] ?? MedicationTrackRecord()
  }

  /// Records the brain's decision for a pill whose medication a pharmacist confirmed.
  public mutating func observe(_ decision: IdentityDecision, truth: MedicationID, at date: Date = Date()) {
    switch decision {
    case .identified(let named) where named == truth:
      records[truth, default: MedicationTrackRecord()].correct += 1
      records[truth, default: MedicationTrackRecord()].streak += 1
    case .identified(let named):
      records[named, default: MedicationTrackRecord()].falseIdentifications += 1
      records[named, default: MedicationTrackRecord()].streak = 0
      records[named, default: MedicationTrackRecord()].lastFalseIdentificationAt = date
      records[truth, default: MedicationTrackRecord()].confusions += 1
    case .ambiguous, .unrecognised, .insufficientKnowledge:
      records[truth, default: MedicationTrackRecord()].abstentions += 1
    }
  }

  /// Thresholds changed: precision history stays, but every medication must rebuild its streak.
  public mutating func resetStreaks() {
    for id in records.keys { records[id]?.streak = 0 }
  }

  public mutating func forget(medication id: MedicationID) {
    records[id] = nil
  }

  public func status(of id: MedicationID, policy: TrustPolicy) -> TrustStatus {
    let record = record(for: id)
    if record.namedCount >= policy.minimumPredictions,
      record.precisionLowerBound >= policy.minimumPrecisionLowerBound,
      record.streak >= policy.minimumStreak
    {
      return .trusted
    }
    if record.falseIdentifications > 0, record.streak < policy.minimumStreak { return .suspended }
    let parts = [
      Double(record.namedCount) / Double(max(1, policy.minimumPredictions)),
      record.precisionLowerBound / max(0.0001, policy.minimumPrecisionLowerBound),
      Double(record.streak) / Double(max(1, policy.minimumStreak)),
    ]
    return .learning(progress: min(1, parts.min() ?? 0))
  }

  public func trustedMedications(policy: TrustPolicy) -> Set<MedicationID> {
    Set(records.keys.filter { status(of: $0, policy: policy) == .trusted })
  }
}
