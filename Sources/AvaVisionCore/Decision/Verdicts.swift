import Foundation

/// Result for one compartment, ordered from best to worst.
public enum CompartmentStatus: String, Codable, Hashable, Sendable, CaseIterable, Comparable {
  /// Count and identity confirmed by a released model. The pharmacist still signs off the pack.
  case verified
  /// Count matches; identity was not checked automatically.
  case countMatched
  /// Evidence is weak, conflicting or missing; a pharmacist must look.
  case needsReview
  /// Confident evidence that the content is wrong.
  case mismatch

  private var severity: Int {
    switch self {
    case .verified: 0
    case .countMatched: 1
    case .needsReview: 2
    case .mismatch: 3
    }
  }

  public static func < (lhs: CompartmentStatus, rhs: CompartmentStatus) -> Bool {
    lhs.severity < rhs.severity
  }
}

/// A specific reason behind a compartment's status.
public enum Finding: Codable, Hashable, Sendable {
  case missing(expected: Int, observed: Int)
  case extra(expected: Int, observed: Int)
  case wrongQuantity(MedicationID, expected: Int, observed: Int)
  case unexpectedMedication(MedicationID, observed: Int)
  case brokenDose(count: Int)
  case foreignObject(count: Int)
  case lowConfidenceObject(count: Int)
  case objectOnBorder(count: Int)
  case unstableAcrossFrames
  case identityNotVerified
  case noExpectation
  case layoutUncalibrated
  case notEvaluated

  /// The least severe status a compartment with this finding can have.
  public var minimumStatus: CompartmentStatus {
    switch self {
    case .missing, .extra, .wrongQuantity, .unexpectedMedication, .brokenDose, .foreignObject:
      .mismatch
    case .lowConfidenceObject, .objectOnBorder, .unstableAcrossFrames, .noExpectation, .layoutUncalibrated,
      .notEvaluated:
      .needsReview
    case .identityNotVerified:
      .countMatched
    }
  }
}

public struct CompartmentVerdict: Codable, Hashable, Sendable, Identifiable {
  public var compartment: CompartmentIndex
  public var status: CompartmentStatus
  public var findings: [Finding]
  public var expectedCount: Int?
  /// Consensus count across frames; `nil` when frames disagreed or nothing was evaluated.
  public var observedCount: Int?
  /// Medications identified under the granted capability.
  public var observedMedications: [MedicationID: Int]

  public var id: CompartmentIndex { compartment }

  public init(
    compartment: CompartmentIndex, findings: [Finding], expectedCount: Int?, observedCount: Int?,
    observedMedications: [MedicationID: Int] = [:]
  ) {
    self.compartment = compartment
    self.findings = findings
    self.status = findings.map(\.minimumStatus).max() ?? .verified
    self.expectedCount = expectedCount
    self.observedCount = observedCount
    self.observedMedications = observedMedications
  }
}

/// Result for the whole pack, ordered from best to worst.
public enum PackStatus: String, Codable, Hashable, Sendable, CaseIterable {
  case verified
  case countMatched
  case needsReview
  case mismatch
  /// Not enough usable frames; nothing was evaluated.
  case retakeRequired

  init(_ status: CompartmentStatus) {
    switch status {
    case .verified: self = .verified
    case .countMatched: self = .countMatched
    case .needsReview: self = .needsReview
    case .mismatch: self = .mismatch
    }
  }
}

/// A reason that concerns the pack or capture rather than one compartment.
public enum PackFinding: Codable, Hashable, Sendable {
  case insufficientUsableFrames(usable: Int, required: Int)
  case captureIssues([CaptureIssue])
  case registrationIssues([RegistrationIssue])
  case objectsOutsideCompartments(count: Int)
  case modelUnavailable
  case profileLayoutMismatch
}

public struct PackVerificationResult: Codable, Hashable, Sendable, Identifiable {
  public var id: UUID
  public var evaluatedAt: Date
  public var layoutID: String
  public var profileID: UUID
  public var modelID: String?
  public var modelVersion: String?
  public var capability: ModelCapability
  public var status: PackStatus
  public var packFindings: [PackFinding]
  public var compartments: [CompartmentVerdict]
  public var usableFrameCount: Int

  public init(
    id: UUID = UUID(), evaluatedAt: Date, layoutID: String, profileID: UUID, modelID: String?,
    modelVersion: String?, capability: ModelCapability, status: PackStatus, packFindings: [PackFinding],
    compartments: [CompartmentVerdict], usableFrameCount: Int
  ) {
    self.id = id
    self.evaluatedAt = evaluatedAt
    self.layoutID = layoutID
    self.profileID = profileID
    self.modelID = modelID
    self.modelVersion = modelVersion
    self.capability = capability
    self.status = status
    self.packFindings = packFindings
    self.compartments = compartments
    self.usableFrameCount = usableFrameCount
  }

  public func verdict(for index: CompartmentIndex) -> CompartmentVerdict? {
    compartments.first { $0.compartment == index }
  }

  /// Compartments a pharmacist must inspect before the pack can be released.
  public var compartmentsRequiringReview: [CompartmentIndex] {
    compartments.filter { $0.status != .verified }.map(\.compartment)
  }
}
