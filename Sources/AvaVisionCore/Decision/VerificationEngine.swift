import Foundation

/// Tunable thresholds of the decision engine.
public struct DecisionPolicy: Codable, Hashable, Sendable {
  /// Number of most recent usable frames that must agree exactly.
  public var requiredConsistentFrames: Int
  /// Detections below this confidence are never counted; they send the compartment to review.
  public var minimumDetectionConfidence: Double
  /// Fraction of automatically verified compartments a pharmacist must still inspect, so the
  /// system's accuracy keeps being measured. At least one per pack when any compartment is verified.
  public var spotCheckRate: Double

  public init(requiredConsistentFrames: Int, minimumDetectionConfidence: Double, spotCheckRate: Double) {
    self.requiredConsistentFrames = requiredConsistentFrames
    self.minimumDetectionConfidence = minimumDetectionConfidence
    self.spotCheckRate = spotCheckRate
  }

  public static let standard = DecisionPolicy(
    requiredConsistentFrames: 3, minimumDetectionConfidence: 0.6, spotCheckRate: 0.1)
}

/// What the learning brain contributes to a decision.
public struct BrainContext: Hashable, Sendable {
  /// Medications whose identification has earned trust (see `TrustLedger`).
  public var trustedMedications: Set<MedicationID>
  /// Medications the brain knows well enough to notice a pill that is none of them.
  public var wellKnownMedications: Set<MedicationID>

  public init(trustedMedications: Set<MedicationID>, wellKnownMedications: Set<MedicationID>) {
    self.trustedMedications = trustedMedications
    self.wellKnownMedications = wellKnownMedications
  }

  public init(_ brain: BrainState) {
    self.init(trustedMedications: brain.trustedMedications, wellKnownMedications: brain.wellKnownMedications)
  }
}

/// Compares the observed pack against its expected profile.
///
/// Safety rules (see docs): weak, missing or conflicting evidence never produces an accepted
/// result; `verified` additionally requires identity authority (a released model or earned brain
/// trust) for every expected medication. The brain can always escalate, never relax.
public struct VerificationEngine: Sendable {
  public var layout: PackLayout
  public var model: ActiveModel?
  public var brain: BrainContext?
  public var policy: DecisionPolicy

  public init(layout: PackLayout, model: ActiveModel?, brain: BrainContext? = nil, policy: DecisionPolicy = .standard) {
    self.layout = layout
    self.model = model
    self.brain = brain
    self.policy = policy
  }

  /// True when the medication may be counted by identity.
  func canIdentify(_ id: MedicationID) -> Bool {
    (model?.capability.canIdentify(id) ?? false) || (brain?.trustedMedications.contains(id) ?? false)
  }

  var hasIdentityAuthority: Bool {
    if case .identity = model?.capability { return true }
    return !(brain?.trustedMedications.isEmpty ?? true)
  }

  public func evaluate(
    profile: PackProfile, frames: [FrameObservation], evaluatedAt: Date = Date(), resultID: UUID = UUID()
  ) -> PackVerificationResult {
    let required = max(1, policy.requiredConsistentFrames)
    let usable = frames.filter(\.isUsable)

    func unevaluated(_ status: PackStatus, _ findings: [PackFinding]) -> PackVerificationResult {
      makeResult(
        id: resultID, profile: profile, status: status, findings: findings,
        compartments: layout.allCompartments.map {
          CompartmentVerdict(
            compartment: $0, findings: [.notEvaluated],
            expectedCount: profile.expectation(for: $0)?.totalQuantity, observedCount: nil)
        },
        spotChecks: [], usableFrameCount: usable.count, evaluatedAt: evaluatedAt)
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
      verdict(for: $0, expectation: profile.expectation(for: $0), assignments: assignments)
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
      id: resultID, profile: profile, status: status, findings: findings, compartments: compartments,
      spotChecks: spotChecks(among: compartments, seed: resultID), usableFrameCount: usable.count,
      evaluatedAt: evaluatedAt)
  }

  /// Deterministic per result, so the audit record can show why each spot check was chosen.
  private func spotChecks(among compartments: [CompartmentVerdict], seed: UUID) -> [CompartmentIndex] {
    let verified = compartments.filter { $0.status == .verified }.map(\.compartment)
    guard !verified.isEmpty, policy.spotCheckRate > 0 else { return [] }
    var generator = SeededGenerator(seed: seed)
    var chosen = verified.filter { _ in Double.random(in: 0..<1, using: &generator) < policy.spotCheckRate }
    if chosen.isEmpty, let any = verified.randomElement(using: &generator) { chosen = [any] }
    return chosen.sorted()
  }

  private struct Tally: Equatable {
    var generic = 0
    var medications: [MedicationID: Int] = [:]
    var broken = 0
    var foreign = 0

    var doses: Int { generic + medications.values.reduce(0, +) }
  }

  /// Final meaning of an object once the brain's trusted identification is taken into account.
  private func resolvedMeaning(_ object: PlacedObject) -> (meaning: LabelMeaning, conflict: Bool) {
    let named = object.identity?.decision.medicationID.flatMap { canIdentify($0) ? $0 : nil }
    switch object.meaning {
    case .pill:
      return (named.map(LabelMeaning.medication) ?? .pill, false)
    case .medication(let id):
      if let named, named != id { return (.pill, true) }
      return (object.meaning, false)
    case .broken, .foreign, .ignore:
      return (object.meaning, false)
    }
  }

  private func verdict(
    for index: CompartmentIndex, expectation: CompartmentExpectation?, assignments: [FrameAssignment]
  ) -> CompartmentVerdict {
    let expected = expectation?.quantities.filter { $0.value > 0 } ?? [:]
    var findings: [Finding] = []
    var tallies: [Tally] = []
    var lowConfidence = 0
    var onBorder = 0
    var conflicts = 0
    var suspected: Set<MedicationID> = []
    var unrecognised = 0
    let noticesStrangers =
      !expected.isEmpty && expected.keys.allSatisfy { brain?.wellKnownMedications.contains($0) ?? false }

    for assignment in assignments {
      var tally = Tally()
      var low = 0
      var conflictsInFrame = 0
      var unrecognisedInFrame = 0
      for object in assignment.inside[index] ?? [] {
        guard object.confidence >= policy.minimumDetectionConfidence else {
          low += 1
          continue
        }
        let (meaning, conflict) = resolvedMeaning(object)
        if conflict { conflictsInFrame += 1 }
        switch meaning {
        case .pill:
          tally.generic += 1
          switch object.identity?.decision {
          case .identified(let id) where expected[id] == nil: suspected.insert(id)
          case .unrecognised where noticesStrangers: unrecognisedInFrame += 1
          default: break
          }
        case .medication(let id): tally.medications[id, default: 0] += 1
        case .broken: tally.broken += 1
        case .foreign: tally.foreign += 1
        case .ignore: break
        }
      }
      tallies.append(tally)
      lowConfidence = max(lowConfidence, low)
      onBorder = max(onBorder, assignment.ambiguous[index]?.count ?? 0)
      conflicts = max(conflicts, conflictsInFrame)
      unrecognised = max(unrecognised, unrecognisedInFrame)
    }

    if lowConfidence > 0 { findings.append(.lowConfidenceObject(count: lowConfidence)) }
    if onBorder > 0 { findings.append(.objectOnBorder(count: onBorder)) }
    if !layout.isCalibrated { findings.append(.layoutUncalibrated) }
    if conflicts > 0 { findings.append(.conflictingIdentity(count: conflicts)) }
    for id in suspected.sorted() { findings.append(.suspectedMedication(id)) }
    if unrecognised > 0 { findings.append(.unrecognisedPill(count: unrecognised)) }

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

    for (id, count) in consensus.medications.sorted(by: { $0.key < $1.key }) where expected[id] == nil {
      findings.append(.unexpectedMedication(id, observed: count))
    }

    let identityCovered =
      hasIdentityAuthority && consensus.generic == 0 && expected.keys.allSatisfy(canIdentify)
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
    id: UUID, profile: PackProfile, status: PackStatus, findings: [PackFinding], compartments: [CompartmentVerdict],
    spotChecks: [CompartmentIndex], usableFrameCount: Int, evaluatedAt: Date
  ) -> PackVerificationResult {
    PackVerificationResult(
      id: id,
      evaluatedAt: evaluatedAt,
      layoutID: layout.id,
      profileID: profile.id,
      modelID: model?.manifest.modelID,
      modelVersion: model?.manifest.version,
      capability: model?.capability ?? .unavailable,
      trustedMedications: (brain?.trustedMedications).map { $0.sorted() } ?? [],
      status: status,
      packFindings: findings,
      compartments: compartments,
      spotChecks: spotChecks,
      usableFrameCount: usableFrameCount
    )
  }
}

/// Small deterministic generator (SplitMix64) seeded from a UUID.
struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64

  init(seed: UUID) {
    let bytes = withUnsafeBytes(of: seed.uuid) { Array($0) }
    state = bytes.prefix(8).reduce(0) { $0 << 8 | UInt64($1) } ^ bytes.suffix(8).reduce(0) { $0 << 8 | UInt64($1) }
  }

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}
