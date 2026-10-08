import Foundation

/// Tunable thresholds of the decision engine.
public struct DecisionPolicy: Codable, Hashable, Sendable {
  /// Number of most recent usable frames that must agree exactly.
  public var requiredConsistentFrames: Int
  /// Detections below this confidence are never counted; they send the compartment to review.
  public var minimumDetectionConfidence: Double

  public init(requiredConsistentFrames: Int, minimumDetectionConfidence: Double) {
    self.requiredConsistentFrames = requiredConsistentFrames
    self.minimumDetectionConfidence = minimumDetectionConfidence
  }

  public static let standard = DecisionPolicy(requiredConsistentFrames: 3, minimumDetectionConfidence: 0.6)
}

/// Compares the observed pack against its expected profile.
///
/// Safety rules (see docs): weak, missing or conflicting evidence never produces an accepted
/// result; `verified` additionally requires a released model that covers every expected medication.
public struct VerificationEngine: Sendable {
  public var layout: PackLayout
  public var model: ActiveModel?
  public var policy: DecisionPolicy

  public init(layout: PackLayout, model: ActiveModel?, policy: DecisionPolicy = .standard) {
    self.layout = layout
    self.model = model
    self.policy = policy
  }

  public func evaluate(
    profile: PackProfile, frames: [FrameObservation], evaluatedAt: Date = Date()
  ) -> PackVerificationResult {
    let required = max(1, policy.requiredConsistentFrames)
    let usable = frames.filter(\.isUsable)

    func unevaluated(_ status: PackStatus, _ findings: [PackFinding]) -> PackVerificationResult {
      makeResult(
        profile: profile, status: status, findings: findings,
        compartments: layout.allCompartments.map {
          CompartmentVerdict(
            compartment: $0, findings: [.notEvaluated],
            expectedCount: profile.expectation(for: $0)?.totalQuantity, observedCount: nil)
        },
        usableFrameCount: usable.count, evaluatedAt: evaluatedAt)
    }

    guard let model, model.capability.canCount else {
      return unevaluated(.needsReview, [.modelUnavailable])
    }
    guard profile.layoutID == layout.id else {
      return unevaluated(.needsReview, [.profileLayoutMismatch])
    }

    let window = usable.suffix(required).compactMap { frame in
      frame.registration.registration.map { (frame, $0) }
    }
    guard window.count == required else {
      var findings: [PackFinding] = [.insufficientUsableFrames(usable: usable.count, required: required)]
      let rejected = frames.filter { !$0.isUsable }
      let captureIssues = Set(rejected.flatMap(\.quality.issues))
      let registrationIssues = Set(rejected.flatMap(\.registration.issues))
      if !captureIssues.isEmpty {
        findings.append(.captureIssues(CaptureIssue.allCases.filter(captureIssues.contains)))
      }
      if !registrationIssues.isEmpty {
        findings.append(.registrationIssues(RegistrationIssue.allCases.filter(registrationIssues.contains)))
      }
      return unevaluated(.retakeRequired, findings)
    }

    let assignments = window.map { frame, registration in
      CompartmentAssigner.assign(
        frame.detections, registration: registration, layout: layout, meaning: model.meaning(of:))
    }

    let compartments = layout.allCompartments.map {
      verdict(
        for: $0, expectation: profile.expectation(for: $0), assignments: assignments,
        capability: model.capability)
    }

    var findings: [PackFinding] = []
    let outside =
      assignments.map { assignment in
        assignment.outsideGrid.filter { $0.confidence >= policy.minimumDetectionConfidence }.count
      }.max() ?? 0
    if outside > 0 { findings.append(.objectsOutsideCompartments(count: outside)) }

    var status = PackStatus(compartments.map(\.status).max() ?? .verified)
    if !findings.isEmpty, status == .verified || status == .countMatched {
      status = .needsReview
    }
    return makeResult(
      profile: profile, status: status, findings: findings, compartments: compartments,
      usableFrameCount: usable.count, evaluatedAt: evaluatedAt)
  }

  private struct Tally: Equatable {
    var generic = 0
    var medications: [MedicationID: Int] = [:]
    var broken = 0
    var foreign = 0

    var doses: Int { generic + medications.values.reduce(0, +) }
  }

  private func verdict(
    for index: CompartmentIndex, expectation: CompartmentExpectation?, assignments: [FrameAssignment],
    capability: ModelCapability
  ) -> CompartmentVerdict {
    var findings: [Finding] = []
    var tallies: [Tally] = []
    var lowConfidence = 0
    var onBorder = 0
    for assignment in assignments {
      var tally = Tally()
      var low = 0
      for object in assignment.inside[index] ?? [] {
        guard object.confidence >= policy.minimumDetectionConfidence else {
          low += 1
          continue
        }
        switch object.meaning {
        case .pill: tally.generic += 1
        case .medication(let id): tally.medications[id, default: 0] += 1
        case .broken: tally.broken += 1
        case .foreign: tally.foreign += 1
        case .ignore: break
        }
      }
      tallies.append(tally)
      lowConfidence = max(lowConfidence, low)
      onBorder = max(onBorder, assignment.ambiguous[index]?.count ?? 0)
    }

    if lowConfidence > 0 { findings.append(.lowConfidenceObject(count: lowConfidence)) }
    if onBorder > 0 { findings.append(.objectOnBorder(count: onBorder)) }
    if !layout.isCalibrated { findings.append(.layoutUncalibrated) }

    guard let consensus = tallies.first, tallies.allSatisfy({ $0 == consensus }) else {
      findings.append(.unstableAcrossFrames)
      if expectation == nil { findings.append(.noExpectation) }
      return CompartmentVerdict(
        compartment: index, findings: findings, expectedCount: expectation?.totalQuantity, observedCount: nil)
    }

    if consensus.foreign > 0 { findings.append(.foreignObject(count: consensus.foreign)) }
    if consensus.broken > 0 { findings.append(.brokenDose(count: consensus.broken)) }

    guard let expectation else {
      findings.append(.noExpectation)
      return CompartmentVerdict(
        compartment: index, findings: findings, expectedCount: nil, observedCount: consensus.doses,
        observedMedications: consensus.medications)
    }

    let expectedCount = expectation.totalQuantity
    if consensus.doses < expectedCount {
      findings.append(.missing(expected: expectedCount, observed: consensus.doses))
    } else if consensus.doses > expectedCount {
      findings.append(.extra(expected: expectedCount, observed: consensus.doses))
    }

    let expected = expectation.quantities
    for (id, count) in consensus.medications.sorted(by: { $0.key < $1.key }) where expected[id] == nil {
      findings.append(.unexpectedMedication(id, observed: count))
    }

    var identityCovered = false
    if case .identity = capability {
      identityCovered = consensus.generic == 0 && expected.keys.allSatisfy(capability.canIdentify)
    }
    if identityCovered {
      for (id, quantity) in expected.sorted(by: { $0.key < $1.key }) {
        let observed = consensus.medications[id] ?? 0
        if observed != quantity {
          findings.append(.wrongQuantity(id, expected: quantity, observed: observed))
        }
      }
    } else {
      findings.append(.identityNotVerified)
    }

    return CompartmentVerdict(
      compartment: index, findings: findings, expectedCount: expectedCount, observedCount: consensus.doses,
      observedMedications: consensus.medications)
  }

  private func makeResult(
    profile: PackProfile, status: PackStatus, findings: [PackFinding], compartments: [CompartmentVerdict],
    usableFrameCount: Int, evaluatedAt: Date
  ) -> PackVerificationResult {
    PackVerificationResult(
      evaluatedAt: evaluatedAt,
      layoutID: layout.id,
      profileID: profile.id,
      modelID: model?.manifest.modelID,
      modelVersion: model?.manifest.version,
      capability: model?.capability ?? .unavailable,
      status: status,
      packFindings: findings,
      compartments: compartments,
      usableFrameCount: usableFrameCount
    )
  }
}
