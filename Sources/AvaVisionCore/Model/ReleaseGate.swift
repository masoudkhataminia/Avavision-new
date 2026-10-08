import Foundation

/// Minimum evidence a model must show on the holdout set before it may identify medications.
public struct ReleaseGatePolicy: Codable, Hashable, Sendable {
  public var minimumSamples: Int
  public var minimumErrorSamples: Int
  public var minimumSamplesPerMedication: Int
  public var minimumCountAccuracy: Double
  public var minimumIdentityPrecision: Double
  public var minimumIdentityRecall: Double
  public var maximumFalseAcceptanceRate: Double

  public init(
    minimumSamples: Int, minimumErrorSamples: Int, minimumSamplesPerMedication: Int,
    minimumCountAccuracy: Double, minimumIdentityPrecision: Double, minimumIdentityRecall: Double,
    maximumFalseAcceptanceRate: Double
  ) {
    self.minimumSamples = minimumSamples
    self.minimumErrorSamples = minimumErrorSamples
    self.minimumSamplesPerMedication = minimumSamplesPerMedication
    self.minimumCountAccuracy = minimumCountAccuracy
    self.minimumIdentityPrecision = minimumIdentityPrecision
    self.minimumIdentityRecall = minimumIdentityRecall
    self.maximumFalseAcceptanceRate = maximumFalseAcceptanceRate
  }

  /// Thresholds recorded in the project decision log.
  public static let standard = ReleaseGatePolicy(
    minimumSamples: 1000,
    minimumErrorSamples: 600,
    minimumSamplesPerMedication: 50,
    minimumCountAccuracy: 0.95,
    minimumIdentityPrecision: 0.95,
    minimumIdentityRecall: 0.95,
    maximumFalseAcceptanceRate: 0.005
  )
}

/// A reason the gate withheld a capability.
public enum GateBlocker: Codable, Hashable, Sendable {
  case modelFileMissing
  case modelHashMismatch
  case unsupportedManifestSchema(Int)
  case modelRevoked
  case noPillLabels
  case stageNotReleased(ModelStage)
  case evaluationMissing
  case insufficientSamples(have: Int, need: Int)
  case insufficientErrorSamples(have: Int, need: Int)
  case countAccuracyTooLow(have: Double, need: Double)
  case identityPrecisionTooLow(have: Double, need: Double)
  case identityRecallTooLow(have: Double, need: Double)
  case falseAcceptanceTooHigh(have: Double, limit: Double)
  case noMedicationLabels
  case medicationUndersampled(MedicationID, have: Int, need: Int)
  case medicationBelowThreshold(MedicationID)
}

public struct GateDecision: Codable, Hashable, Sendable {
  public var capability: ModelCapability
  /// Everything that prevented a stronger capability. Empty only for full identity coverage.
  public var blockers: [GateBlocker]

  public init(capability: ModelCapability, blockers: [GateBlocker]) {
    self.capability = capability
    self.blockers = blockers
  }
}

public struct ReleaseGate: Sendable {
  public var policy: ReleaseGatePolicy

  public init(policy: ReleaseGatePolicy = .standard) {
    self.policy = policy
  }

  /// Decides what the model may do.
  ///
  /// - Parameter computedModelSHA256: Hash of the model artefact actually on disk, or `nil` if missing.
  public func evaluate(_ manifest: ModelManifest, computedModelSHA256: String?) -> GateDecision {
    guard let computed = computedModelSHA256 else {
      return GateDecision(capability: .unavailable, blockers: [.modelFileMissing])
    }
    guard computed.lowercased() == manifest.modelSHA256.lowercased() else {
      return GateDecision(capability: .unavailable, blockers: [.modelHashMismatch])
    }
    guard manifest.schemaVersion == ModelManifest.currentSchemaVersion else {
      return GateDecision(capability: .unavailable, blockers: [.unsupportedManifestSchema(manifest.schemaVersion)])
    }
    guard manifest.stage != .revoked else {
      return GateDecision(capability: .unavailable, blockers: [.modelRevoked])
    }
    let countsPills = manifest.labels.values.contains {
      switch $0 {
      case .pill, .medication: true
      case .broken, .foreign, .ignore: false
      }
    }
    guard countsPills else { return GateDecision(capability: .unavailable, blockers: [.noPillLabels]) }

    var blockers: [GateBlocker] = []
    if manifest.stage != .released { blockers.append(.stageNotReleased(manifest.stage)) }
    let medications = manifest.medicationLabels
    if medications.isEmpty { blockers.append(.noMedicationLabels) }
    guard let report = manifest.evaluation else {
      return GateDecision(capability: .countOnly, blockers: blockers + [.evaluationMissing])
    }

    if report.totalSamples < policy.minimumSamples {
      blockers.append(.insufficientSamples(have: report.totalSamples, need: policy.minimumSamples))
    }
    if report.errorSamples < policy.minimumErrorSamples {
      blockers.append(.insufficientErrorSamples(have: report.errorSamples, need: policy.minimumErrorSamples))
    }
    if report.countAccuracy < policy.minimumCountAccuracy {
      blockers.append(.countAccuracyTooLow(have: report.countAccuracy, need: policy.minimumCountAccuracy))
    }
    if report.identityPrecision < policy.minimumIdentityPrecision {
      blockers.append(
        .identityPrecisionTooLow(have: report.identityPrecision, need: policy.minimumIdentityPrecision))
    }
    if report.identityRecall < policy.minimumIdentityRecall {
      blockers.append(.identityRecallTooLow(have: report.identityRecall, need: policy.minimumIdentityRecall))
    }
    if report.falseAcceptanceRate > policy.maximumFalseAcceptanceRate {
      blockers.append(
        .falseAcceptanceTooHigh(have: report.falseAcceptanceRate, limit: policy.maximumFalseAcceptanceRate))
    }
    guard blockers.isEmpty else { return GateDecision(capability: .countOnly, blockers: blockers) }

    var approved = Set<MedicationID>()
    for id in medications.sorted() {
      guard let metrics = report.perMedication[id], metrics.samples >= policy.minimumSamplesPerMedication else {
        let have = report.perMedication[id]?.samples ?? 0
        blockers.append(.medicationUndersampled(id, have: have, need: policy.minimumSamplesPerMedication))
        continue
      }
      guard metrics.precision >= policy.minimumIdentityPrecision, metrics.recall >= policy.minimumIdentityRecall
      else {
        blockers.append(.medicationBelowThreshold(id))
        continue
      }
      approved.insert(id)
    }
    return GateDecision(capability: approved.isEmpty ? .countOnly : .identity(approved), blockers: blockers)
  }
}
