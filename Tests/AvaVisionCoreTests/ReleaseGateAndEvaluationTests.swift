import XCTest

@testable import AvaVisionCore

final class ReleaseGateTests: XCTestCase {
  private let gate = ReleaseGate()

  private func report(
    total: Int = 1200, errors: Int = 700, countAccuracy: Double = 0.99, precision: Double = 0.99,
    recall: Double = 0.99, falseAcceptance: Double = 0.001,
    perMedication: [MedicationID: MedicationMetrics] = [
      Fixtures.metformin: MedicationMetrics(samples: 80, precision: 0.99, recall: 0.98),
      Fixtures.atorvastatin: MedicationMetrics(samples: 20, precision: 1, recall: 1),
      Fixtures.aspirin: MedicationMetrics(samples: 90, precision: 0.9, recall: 0.99),
    ]
  ) -> EvaluationReport {
    EvaluationReport(
      datasetID: "holdout-v1", totalSamples: total, errorSamples: errors, countAccuracy: countAccuracy,
      identityPrecision: precision, identityRecall: recall, falseAcceptanceRate: falseAcceptance,
      countFalseAcceptanceRate: 0.001, reviewRate: 0.1, perMedication: perMedication)
  }

  func testIntegrityFailuresDisableTheModel() {
    let manifest = Fixtures.manifest()
    XCTAssertEqual(gate.evaluate(manifest, computedModelSHA256: nil).blockers, [.modelFileMissing])
    XCTAssertEqual(gate.evaluate(manifest, computedModelSHA256: "different").capability, .unavailable)
    XCTAssertEqual(gate.evaluate(manifest, computedModelSHA256: "ABC123").capability, .countOnly)

    var revoked = manifest
    revoked.stage = .revoked
    XCTAssertEqual(gate.evaluate(revoked, computedModelSHA256: "abc123").blockers, [.modelRevoked])

    var future = manifest
    future.schemaVersion = 99
    XCTAssertEqual(gate.evaluate(future, computedModelSHA256: "abc123").capability, .unavailable)

    let noPills = Fixtures.manifest(labels: ["foreign": .foreign])
    XCTAssertEqual(gate.evaluate(noPills, computedModelSHA256: "abc123").blockers, [.noPillLabels])
  }

  func testDevelopmentModelOnlyCounts() {
    let decision = gate.evaluate(Fixtures.manifest(), computedModelSHA256: "abc123")
    XCTAssertEqual(decision.capability, .countOnly)
    XCTAssertEqual(decision.blockers, [.stageNotReleased(.development), .noMedicationLabels, .evaluationMissing])
  }

  func testReleasedModelIdentifiesOnlyMedicationsThatPassIndividually() {
    let manifest = Fixtures.manifest(stage: .released, labels: Fixtures.identityLabels, evaluation: report())
    let decision = gate.evaluate(manifest, computedModelSHA256: "abc123")
    XCTAssertEqual(decision.capability, .identity([Fixtures.metformin]))
    XCTAssertEqual(
      decision.blockers,
      [
        .medicationBelowThreshold(Fixtures.aspirin),
        .medicationUndersampled(Fixtures.atorvastatin, have: 20, need: 50),
      ])
  }

  func testGlobalMetricsBelowThresholdKeepCountOnly() {
    let manifest = Fixtures.manifest(
      stage: .released, labels: Fixtures.identityLabels,
      evaluation: report(
        total: 900, errors: 100, countAccuracy: 0.9, precision: 0.9, recall: 0.8, falseAcceptance: 0.02))
    let decision = gate.evaluate(manifest, computedModelSHA256: "abc123")
    XCTAssertEqual(decision.capability, .countOnly)
    XCTAssertEqual(
      decision.blockers,
      [
        .insufficientSamples(have: 900, need: 1000), .insufficientErrorSamples(have: 100, need: 600),
        .countAccuracyTooLow(have: 0.9, need: 0.95), .identityPrecisionTooLow(have: 0.9, need: 0.95),
        .identityRecallTooLow(have: 0.8, need: 0.95), .falseAcceptanceTooHigh(have: 0.02, limit: 0.005),
      ])
  }

  func testCapabilityResolution() {
    let capability = ModelCapability.identity([Fixtures.metformin])
    XCTAssertEqual(capability.resolve(.medication(Fixtures.metformin)), .medication(Fixtures.metformin))
    XCTAssertEqual(capability.resolve(.medication(Fixtures.aspirin)), .pill)
    XCTAssertEqual(ModelCapability.countOnly.resolve(.medication(Fixtures.metformin)), .pill)
    XCTAssertFalse(ModelCapability.unavailable.canCount)
    XCTAssertEqual(Fixtures.manifest().meaning(of: "never-seen"), .pill)
  }
}

final class EvaluatorTests: XCTestCase {
  func testMetrics() {
    let m = Fixtures.metformin
    let a = Fixtures.aspirin
    let samples = [
      // Correct pack, verified.
      EvaluationSample(
        sampleID: "1", expected: [m: 1], truth: [m: 1], predictedStatus: .verified, predictedCount: 1,
        predictedMedications: [m: 1]),
      // Wrong medication accepted: a false acceptance.
      EvaluationSample(
        sampleID: "2", expected: [m: 1], truth: [a: 1], predictedStatus: .verified, predictedCount: 1,
        predictedMedications: [m: 1]),
      // Missing dose caught.
      EvaluationSample(
        sampleID: "3", expected: [m: 2], truth: [m: 1], predictedStatus: .mismatch, predictedCount: 1,
        predictedMedications: [m: 1]),
      // Unstable, sent to review.
      EvaluationSample(
        sampleID: "4", expected: [m: 1], truth: [m: 1], truthHasBrokenOrForeign: true, predictedStatus: .needsReview,
        predictedCount: nil),
    ]
    let report = Evaluator.evaluate(samples, datasetID: "unit")
    XCTAssertEqual(report.totalSamples, 4)
    XCTAssertEqual(report.errorSamples, 3)
    XCTAssertEqual(report.countAccuracy, 1)
    XCTAssertEqual(report.falseAcceptanceRate, 1.0 / 3.0, accuracy: 1e-12)
    XCTAssertEqual(report.countFalseAcceptanceRate, 0)
    XCTAssertEqual(report.reviewRate, 0.25)
    // TP: 1 + 0 + 1 + 0 = 2, FP: 1 (sample 2), FN: 1 (aspirin) + 1 (sample 4)
    XCTAssertEqual(report.identityPrecision, 2.0 / 3.0, accuracy: 1e-12)
    XCTAssertEqual(report.identityRecall, 2.0 / 4.0, accuracy: 1e-12)
    XCTAssertEqual(report.perMedication[a], MedicationMetrics(samples: 1, precision: 0, recall: 0))
    XCTAssertEqual(report.perMedication[m]?.samples, 3)
  }

  func testEmptyErrorSetReportsWorstCaseFalseAcceptance() {
    let report = Evaluator.evaluate([], datasetID: "empty")
    XCTAssertEqual(report.falseAcceptanceRate, 1)
    XCTAssertEqual(report.countFalseAcceptanceRate, 1)
  }
}

final class EmbedderManifestTests: XCTestCase {
  func testIntegrityAndBuiltInSegmenter() throws {
    let manifest = EmbedderManifest(
      embedderID: "avavision-dinov2s-v1", version: "1", modelSHA256: "ABC", inputSize: 224, outputName: "embedding",
      baseModelLicense: "Apache-2.0")
    XCTAssertTrue(manifest.isIntact(computedModelSHA256: "abc"))
    XCTAssertFalse(manifest.isIntact(computedModelSHA256: "abd"))
    XCTAssertFalse(manifest.isIntact(computedModelSHA256: nil))
    let data = try AvaVisionJSON.encoder().encode(manifest)
    XCTAssertEqual(try AvaVisionJSON.decoder().decode(EmbedderManifest.self, from: data), manifest)

    let segmenter = ActiveModel.builtInPocketSegmenter
    XCTAssertEqual(segmenter.capability, .countOnly)
    XCTAssertEqual(segmenter.meaning(of: "pill"), .pill)
  }
}
