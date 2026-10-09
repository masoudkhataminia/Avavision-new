import Foundation
import XCTest

@testable import AvaVisionCore

/// Deterministic random numbers for reproducible synthetic embeddings.
struct SplitMix64: RandomNumberGenerator {
  var state: UInt64
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}

/// Synthetic "appearance space": each medication has a prototype; pills are noisy copies of it.
struct SyntheticPills {
  static let embedder: EmbedderID = "test-embedder"
  static let dimension = 32
  var rng = SplitMix64(state: 42)
  var prototypes: [MedicationID: [Float]] = [:]

  mutating func gaussian() -> Float {
    let u1 = Float.random(in: Float.ulpOfOne..<1, using: &rng)
    let u2 = Float.random(in: 0..<1, using: &rng)
    return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
  }

  mutating func prototype(_ id: MedicationID) -> [Float] {
    if let existing = prototypes[id] { return existing }
    let vector = (0..<Self.dimension).map { _ in gaussian() }
    prototypes[id] = vector
    return vector
  }

  mutating func pill(_ id: MedicationID, noise: Float = 0.25) -> Embedding {
    let base = Embedding(embedderID: Self.embedder, raw: prototype(id))!.vector
    let raw = base.map { $0 + noise * gaussian() / Float(Self.dimension).squareRoot() }
    return Embedding(embedderID: Self.embedder, raw: raw)!
  }

  mutating func unknownPill() -> Embedding {
    Embedding(embedderID: Self.embedder, raw: (0..<Self.dimension).map { _ in gaussian() })!
  }

  /// `groups` photos of `perGroup` pills for each medication.
  mutating func knowledge(_ medications: [MedicationID], groups: Int = 3, perGroup: Int = 8) -> KnowledgeBase {
    var knowledge = KnowledgeBase(embedderID: Self.embedder)
    for id in medications {
      for _ in 0..<groups {
        let group = UUID()
        for _ in 0..<perGroup {
          knowledge.add(PillExemplar(medicationID: id, embedding: pill(id), source: .teaching, groupID: group))
        }
      }
    }
    return knowledge
  }
}

final class EmbeddingAndKnowledgeTests: XCTestCase {
  func testEmbeddingNormalizesAndRoundTrips() throws {
    let embedding = try XCTUnwrap(Embedding(embedderID: "e", raw: [3, 4]))
    XCTAssertEqual(embedding.vector, [0.6, 0.8])
    XCTAssertNil(Embedding(embedderID: "e", raw: [0, 0]))
    XCTAssertNil(Embedding(embedderID: "e", raw: [.nan, 1]))
    XCTAssertNil(Embedding(embedderID: "e", raw: []))
    XCTAssertEqual(embedding.similarity(to: embedding) ?? 0, 1, accuracy: 1e-6)
    XCTAssertNil(embedding.similarity(to: Embedding(embedderID: "other", raw: [3, 4])!))

    let data = try AvaVisionJSON.encoder().encode(embedding)
    XCTAssertTrue(String(decoding: data, as: UTF8.self).contains(#""dimension":2"#))
    XCTAssertEqual(try AvaVisionJSON.decoder().decode(Embedding.self, from: data), embedding)
    let corrupt = Data(#"{"embedderID":"e","dimension":3,"vector":"AAAAAA=="}"#.utf8)
    XCTAssertThrowsError(try AvaVisionJSON.decoder().decode(Embedding.self, from: corrupt))
  }

  func testKnowledgeSkipsDuplicatesRejectsForeignEmbeddersAndForgetsRedundantPills() {
    var synthetic = SyntheticPills()
    var knowledge = KnowledgeBase(
      embedderID: SyntheticPills.embedder,
      policy: KnowledgePolicy(maximumExemplarsPerMedication: 5, duplicateSimilarity: 0.999))
    let group = UUID()
    let first = synthetic.pill(Fixtures.metformin)
    XCTAssertEqual(
      knowledge.add(
        PillExemplar(medicationID: Fixtures.metformin, embedding: first, source: .teaching, groupID: group)), .added)
    XCTAssertEqual(
      knowledge.add(
        PillExemplar(medicationID: Fixtures.metformin, embedding: first, source: .teaching, groupID: group)), .duplicate
    )
    XCTAssertEqual(
      knowledge.add(
        PillExemplar(
          medicationID: Fixtures.metformin, embedding: Embedding(embedderID: "other", raw: first.vector)!,
          source: .teaching, groupID: group)),
      .incompatibleEmbedder)

    for _ in 0..<10 {
      knowledge.add(
        PillExemplar(
          medicationID: Fixtures.metformin, embedding: synthetic.pill(Fixtures.metformin), source: .teaching,
          groupID: UUID()))
    }
    XCTAssertEqual(knowledge.count(for: Fixtures.metformin), 5)
    knowledge.add(
      PillExemplar(
        medicationID: Fixtures.aspirin, embedding: synthetic.pill(Fixtures.aspirin), source: .teaching, groupID: group))
    XCTAssertEqual(knowledge.medications, [Fixtures.metformin, Fixtures.aspirin])
    knowledge.forget(medication: Fixtures.metformin)
    XCTAssertEqual(knowledge.medications, [Fixtures.aspirin])
  }
}

final class IdentityTests: XCTestCase {
  private let meds: [MedicationID] = ["med-a", "med-b", "med-c", "med-d"]

  func testUncalibratedBrainNeverNamesAMedication() {
    var synthetic = SyntheticPills()
    let knowledge = synthetic.knowledge(meds)
    let evidence = IdentityClassifier(knowledge: knowledge, policy: .uncalibrated).classify(synthetic.pill("med-a"))
    XCTAssertEqual(evidence.decision, .insufficientKnowledge)
    XCTAssertEqual(evidence.candidates.first?.medicationID, "med-a")
    XCTAssertEqual(evidence.candidates.count, 3)
  }

  func testCalibrationLearnsThresholdsThatIdentifyAndRejectUnknownPills() throws {
    var synthetic = SyntheticPills()
    let knowledge = synthetic.knowledge(meds)
    let (report, learned) = IdentityCalibrator().calibrate(knowledge: knowledge, base: .uncalibrated)
    let policy = try XCTUnwrap(learned, "\(report)")
    XCTAssertEqual(policy.version, 1)
    XCTAssertGreaterThanOrEqual(report.precisionLowerBound, 0.9)
    XCTAssertEqual(report.falseIdentifications, 0)
    XCTAssertGreaterThan(report.coverage, 0.9)

    let classifier = IdentityClassifier(knowledge: knowledge, policy: policy)
    var named = 0
    for id in meds {
      for _ in 0..<25 {
        let decision = classifier.classify(synthetic.pill(id)).decision
        if case .identified(let found) = decision {
          XCTAssertEqual(found, id)
          named += 1
        }
      }
    }
    XCTAssertGreaterThan(named, 90)
    for _ in 0..<50 {
      XCTAssertNil(classifier.classify(synthetic.unknownPill()).decision.medicationID)
    }
  }

  func testIndistinguishableMedicationsAreNeverNamed() {
    var synthetic = SyntheticPills()
    synthetic.prototypes["twin-b"] = synthetic.prototype("twin-a")
    let knowledge = synthetic.knowledge(["twin-a", "twin-b"], groups: 4)
    let (report, policy) = IdentityCalibrator().calibrate(knowledge: knowledge, base: .uncalibrated)
    if let policy {
      let classifier = IdentityClassifier(knowledge: knowledge, policy: policy)
      let named = (0..<40).compactMap { _ in classifier.classify(synthetic.pill("twin-a")).decision.medicationID }
      XCTAssertLessThan(named.count, 4, "\(report)")
    } else {
      XCTAssertEqual(report.outcome, .targetUnreachable)
    }
  }

  func testCalibrationNeedsEnoughKnowledge() {
    var synthetic = SyntheticPills()
    let (report, policy) = IdentityCalibrator().calibrate(
      knowledge: synthetic.knowledge(["only-one"]), base: .uncalibrated)
    XCTAssertNil(policy)
    XCTAssertEqual(report.outcome, .insufficientData(medications: 1, queries: 24))
  }

  func testPoorlyKnownMedicationIsNotNamedOrOverruled() {
    var synthetic = SyntheticPills()
    var knowledge = synthetic.knowledge(["well-known"])
    knowledge.add(
      PillExemplar(medicationID: "new-med", embedding: synthetic.pill("new-med"), source: .teaching, groupID: UUID()))
    var policy = IdentityPolicy.uncalibrated
    policy.acceptSimilarity = 0.5
    let classifier = IdentityClassifier(knowledge: knowledge, policy: policy)
    XCTAssertEqual(classifier.classify(synthetic.pill("new-med")).decision, .insufficientKnowledge)
    XCTAssertEqual(classifier.classify(synthetic.pill("well-known")).decision, .identified("well-known"))
  }

  func testAmbiguityMargin() {
    var synthetic = SyntheticPills()
    let knowledge = synthetic.knowledge(["med-a", "med-b"])
    var policy = IdentityPolicy.uncalibrated
    policy.acceptSimilarity = -1
    policy.minimumMargin = 3
    let classifier = IdentityClassifier(knowledge: knowledge, policy: policy)
    XCTAssertEqual(classifier.classify(synthetic.pill("med-a")).decision, .ambiguous)
  }

  func testWilsonLowerBound() {
    XCTAssertEqual(Statistics.wilsonLowerBound(successes: 0, trials: 0), 0)
    XCTAssertEqual(Statistics.wilsonLowerBound(successes: 81, trials: 100), 0.7222, accuracy: 0.001)
    XCTAssertEqual(Statistics.wilsonLowerBound(successes: 100, trials: 100), 0.9630, accuracy: 0.001)
  }
}

final class TrustLedgerTests: XCTestCase {
  private let policy = TrustPolicy(minimumPredictions: 20, minimumPrecisionLowerBound: 0.8, minimumStreak: 10)

  func testTrustIsEarnedAndLostImmediately() {
    var ledger = TrustLedger(embedderID: "e")
    for _ in 0..<19 { ledger.observe(.identified("m"), truth: "m") }
    guard case .learning(let progress) = ledger.status(of: "m", policy: policy) else {
      return XCTFail("must still be learning")
    }
    XCTAssertEqual(progress, 0.95, accuracy: 1e-9)
    ledger.observe(.identified("m"), truth: "m")
    XCTAssertEqual(ledger.status(of: "m", policy: policy), .trusted)
    XCTAssertEqual(ledger.trustedMedications(policy: policy), ["m"])

    ledger.observe(.identified("m"), truth: "other")
    XCTAssertEqual(ledger.status(of: "m", policy: policy), .suspended)
    XCTAssertEqual(ledger.record(for: "other").confusions, 1)
    for _ in 0..<9 { ledger.observe(.identified("m"), truth: "m") }
    XCTAssertEqual(ledger.status(of: "m", policy: policy), .suspended)
    ledger.observe(.identified("m"), truth: "m")
    XCTAssertEqual(ledger.status(of: "m", policy: policy), .trusted)

    ledger.resetStreaks()
    XCTAssertNotEqual(ledger.status(of: "m", policy: policy), .trusted)
  }

  func testAbstentionsNeitherBuildNorBreakTrust() {
    var ledger = TrustLedger(embedderID: "e")
    for decision in [IdentityDecision.ambiguous, .unrecognised, .insufficientKnowledge] {
      ledger.observe(decision, truth: "m")
    }
    XCTAssertEqual(ledger.record(for: "m").abstentions, 3)
    XCTAssertEqual(ledger.record(for: "m").namedCount, 0)
    XCTAssertEqual(ledger.status(of: "m", policy: policy), .learning(progress: 0))
  }
}

final class LearningTests: XCTestCase {
  private var synthetic = SyntheticPills()
  private let checkID = UUID()

  private func sightings(_ index: CompartmentIndex, _ count: Int, _ id: MedicationID, identified: Bool = true)
    -> [PillSighting]
  {
    (0..<count).map { _ in
      PillSighting(
        compartment: index, embedding: synthetic.pill(id),
        identity: IdentityEvidence(
          embedderID: SyntheticPills.embedder, policyVersion: 1,
          decision: identified ? .identified(id) : .unrecognised, candidates: []),
        cropFile: "crop.jpg")
    }
  }

  private func signOff(_ reviews: [CompartmentReview]) -> PharmacistSignOff {
    PharmacistSignOff(pharmacistIdentifier: "MK", decision: .released, reviews: reviews, acknowledgedPackFindings: true)
  }

  func testLearnsOnlyFromInspectedConfirmedCompartments() {
    let single = CompartmentIndex.at(0, 0)
    let corrected = CompartmentIndex.at(0, 1)
    let miscounted = CompartmentIndex.at(0, 2)
    let mixed = CompartmentIndex.at(1, 0)
    let unreviewed = CompartmentIndex.at(1, 1)
    let profile = Fixtures.profile(
      default: [ExpectedItem(medicationID: Fixtures.metformin, quantity: 2)],
      overrides: [
        mixed: [
          ExpectedItem(medicationID: Fixtures.metformin, quantity: 1),
          ExpectedItem(medicationID: Fixtures.aspirin, quantity: 1),
        ]
      ])
    let seen =
      sightings(single, 2, Fixtures.metformin) + sightings(corrected, 2, Fixtures.metformin)
      + sightings(miscounted, 1, Fixtures.metformin) + sightings(mixed, 1, Fixtures.metformin, identified: false)
      + sightings(mixed, 1, Fixtures.aspirin) + sightings(unreviewed, 2, Fixtures.metformin)
    let plan = LearningPlanner.plan(
      checkID: checkID,
      signOff: signOff([
        CompartmentReview(compartment: single, outcome: .confirmedCorrect),
        CompartmentReview(compartment: corrected, outcome: .corrected),
        CompartmentReview(compartment: miscounted, outcome: .confirmedCorrect),
        CompartmentReview(compartment: mixed, outcome: .confirmedCorrect),
      ]), profile: profile, layout: Fixtures.layout, sightings: seen)

    XCTAssertEqual(plan.exemplars.count, 2)
    XCTAssertTrue(plan.exemplars.allSatisfy { $0.medicationID == Fixtures.metformin && $0.groupID == checkID })
    XCTAssertEqual(plan.observations.count, 2)
    XCTAssertEqual(plan.labellingTasks.count, 1)
    let task = plan.labellingTasks[0]
    XCTAssertEqual(task.compartmentLabel, "D1 · PM")
    XCTAssertEqual(task.expected, [Fixtures.metformin: 1, Fixtures.aspirin: 1])
    XCTAssertEqual(task.priority, 0.5)

    let ids = task.sightings.map(\.id)
    XCTAssertThrowsError(try task.resolve(labels: [ids[0]: Fixtures.metformin]))
    XCTAssertThrowsError(try task.resolve(labels: [ids[0]: Fixtures.metformin, ids[1]: Fixtures.metformin])) {
      XCTAssertEqual($0 as? LabellingTask.LabellingError, .countsDoNotMatchExpected)
    }
    let resolved = try? task.resolve(labels: [ids[0]: Fixtures.metformin, ids[1]: Fixtures.aspirin])
    XCTAssertEqual(resolved?.exemplars.map(\.medicationID), [Fixtures.metformin, Fixtures.aspirin])
    XCTAssertEqual(resolved?.exemplars.map(\.source), [.labelled, .labelled])
  }

  func testBrainTeachesCalibratesAndEarnsTrust() throws {
    var brain = BrainState(
      embedderID: SyntheticPills.embedder,
      trustPolicy: TrustPolicy(minimumPredictions: 30, minimumPrecisionLowerBound: 0.85, minimumStreak: 20))
    XCTAssertTrue(brain.wellKnownMedications.isEmpty)
    let meds: [MedicationID] = ["med-a", "med-b", "med-c"]
    for id in meds {
      for _ in 0..<3 {
        brain.teach(id, embeddings: (0..<8).map { _ in (synthetic.pill(id), nil) })
      }
    }
    XCTAssertTrue(brain.identityPolicy.isCalibrated, "\(String(describing: brain.lastCalibration))")
    XCTAssertEqual(brain.wellKnownMedications, Set(meds))
    XCTAssertTrue(brain.trustedMedications.isEmpty)

    // Shadow mode: the brain names pills, pharmacists confirm them, trust builds per medication.
    for _ in 0..<40 {
      let evidence = brain.classifier().classify(synthetic.pill("med-a"))
      var plan = LearningPlan()
      plan.observations.append(.init(decision: evidence.decision, truth: "med-a"))
      brain.learn(plan)
    }
    XCTAssertEqual(brain.trustStatus(of: "med-a"), .trusted)
    XCTAssertEqual(brain.trustedMedications, ["med-a"])
    XCTAssertEqual(brain.summary.trustedMedications, ["med-a"])

    let taskPlan = LearningPlanner.plan(
      checkID: checkID,
      signOff: signOff([CompartmentReview(compartment: .at(0, 0), outcome: .confirmedCorrect)]),
      profile: Fixtures.profile(default: [
        ExpectedItem(medicationID: "med-b", quantity: 1), ExpectedItem(medicationID: "med-c", quantity: 1),
      ]), layout: Fixtures.layout, sightings: sightings(.at(0, 0), 1, "med-b") + sightings(.at(0, 0), 1, "med-c"))
    brain.learn(taskPlan)
    let task = try XCTUnwrap(brain.labellingQueue.first)
    let before = brain.knowledge.exemplars.count
    try brain.resolveTask(
      task.id, labels: [task.sightings[0].id: "med-b", task.sightings[1].id: "med-c"])
    XCTAssertTrue(brain.labellingQueue.isEmpty)
    XCTAssertEqual(brain.knowledge.exemplars.count, before + 2)
    XCTAssertThrowsError(try brain.resolveTask(task.id, labels: [:]))

    brain.forget(medication: "med-a")
    XCTAssertFalse(brain.knowledge.medications.contains("med-a"))
    XCTAssertTrue(brain.trustedMedications.isEmpty)
  }

  func testMigrationKeepsOnlyReembeddedPillsAndStartsTrustAgain() {
    var brain = BrainState(embedderID: SyntheticPills.embedder)
    brain.teach("med-a", embeddings: (0..<3).map { _ in (synthetic.pill("med-a"), "a.jpg") })
    let reembedded = Dictionary(
      uniqueKeysWithValues: brain.knowledge.exemplars.prefix(2).map {
        ($0.id, Embedding(embedderID: "v2", raw: $0.embedding.vector.map { $0 * 2 })!)
      })
    let migrated = brain.migrated(to: "v2", embeddings: reembedded)
    XCTAssertEqual(migrated.embedderID, "v2")
    XCTAssertEqual(migrated.knowledge.exemplars.count, 2)
    XCTAssertEqual(migrated.identityPolicy.version, 0)
    XCTAssertEqual(migrated.knowledge.exemplars.map(\.cropFile), ["a.jpg", "a.jpg"])
  }

  func testBrainStateRoundTrips() throws {
    var brain = BrainState(embedderID: SyntheticPills.embedder)
    brain.teach("med-a", embeddings: (0..<3).map { _ in (synthetic.pill("med-a"), nil) }, at: Fixtures.fixedDate)
    let data = try AvaVisionJSON.encoder().encode(brain)
    XCTAssertEqual(try AvaVisionJSON.decoder().decode(BrainState.self, from: data), brain)
  }
}
