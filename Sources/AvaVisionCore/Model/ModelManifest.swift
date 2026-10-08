import Foundation

/// Lifecycle stage of a detection model.
public enum ModelStage: String, Codable, Hashable, Sendable, CaseIterable {
  /// Internal experiments; may count, never identifies.
  case development
  /// Being evaluated on the holdout set; may count, never identifies.
  case validation
  /// Passed the release gate; may identify the medications its evaluation covers.
  case released
  /// Withdrawn; must not be used at all.
  case revoked
}

/// Signed description of a model artefact. Shipped next to the model file and checked before use.
public struct ModelManifest: Codable, Hashable, Sendable {
  public static let currentSchemaVersion = 1

  public var schemaVersion: Int
  public var modelID: String
  public var version: String
  public var stage: ModelStage
  /// SHA-256 of the model artefact (see `ContentHasher`), lowercase hex.
  public var modelSHA256: String
  /// Meaning of every output label of the model.
  public var labels: [String: LabelMeaning]
  /// Holdout evaluation; required for identity.
  public var evaluation: EvaluationReport?
  public var notes: String?

  public init(
    schemaVersion: Int = ModelManifest.currentSchemaVersion, modelID: String, version: String,
    stage: ModelStage, modelSHA256: String, labels: [String: LabelMeaning],
    evaluation: EvaluationReport? = nil, notes: String? = nil
  ) {
    self.schemaVersion = schemaVersion
    self.modelID = modelID
    self.version = version
    self.stage = stage
    self.modelSHA256 = modelSHA256
    self.labels = labels
    self.evaluation = evaluation
    self.notes = notes
  }

  /// Unknown labels are treated as generic pills: an unexpected class must never be ignored.
  public func meaning(of label: String) -> LabelMeaning {
    labels[label] ?? .pill
  }

  /// Medications the model claims to recognise.
  public var medicationLabels: Set<MedicationID> {
    Set(
      labels.values.compactMap {
        if case .medication(let id) = $0 { return id }
        return nil
      })
  }
}

/// What the app may do with the currently loaded model.
public enum ModelCapability: Codable, Hashable, Sendable {
  /// No trustworthy model: nothing can be evaluated automatically.
  case unavailable
  /// Counting is allowed; identity is never claimed.
  case countOnly
  /// Counting plus identity for exactly these medications.
  case identity(Set<MedicationID>)

  public var canCount: Bool { self != .unavailable }

  public func canIdentify(_ id: MedicationID) -> Bool {
    if case .identity(let ids) = self { return ids.contains(id) }
    return false
  }

  /// Downgrades identity claims the capability does not cover to a generic pill.
  public func resolve(_ meaning: LabelMeaning) -> LabelMeaning {
    if case .medication(let id) = meaning, !canIdentify(id) { return .pill }
    return meaning
  }
}

/// A model ready for use, after the release gate decided its capability.
public struct ActiveModel: Hashable, Sendable {
  public var manifest: ModelManifest
  public var decision: GateDecision

  public init(manifest: ModelManifest, decision: GateDecision) {
    self.manifest = manifest
    self.decision = decision
  }

  public var capability: ModelCapability { decision.capability }

  /// Final meaning of an output label under the granted capability.
  public func meaning(of label: String) -> LabelMeaning {
    capability.resolve(manifest.meaning(of: label))
  }
}
