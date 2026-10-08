import XCTest

@testable import AvaVisionCore

final class VerificationEngineTests: XCTestCase {
  private let one = [ExpectedItem(medicationID: Fixtures.metformin, quantity: 1)]
  private let target = CompartmentIndex.at(0, 1)

  /// Every compartment gets one metformin detection.
  private var fullPack: [Detection] {
    Fixtures.layout.allCompartments.flatMap { Fixtures.detections("pill", count: 1, in: $0) }
  }

  private func evaluate(
    _ detections: [Detection], profile: PackProfile? = nil, engine: VerificationEngine = Fixtures.engine()
  ) -> PackVerificationResult {
    engine.evaluate(
      profile: profile ?? Fixtures.profile(default: one), frames: Fixtures.frames(detections),
      evaluatedAt: Fixtures.fixedDate)
  }

  func testCountOnlyModelNeverVerifiesEvenWhenCountsMatch() {
    let result = evaluate(fullPack)
    XCTAssertEqual(result.status, .countMatched)
    XCTAssertEqual(result.compartments.count, 6)
    for verdict in result.compartments {
      XCTAssertEqual(verdict.status, .countMatched)
      XCTAssertEqual(verdict.findings, [.identityNotVerified])
      XCTAssertEqual(verdict.observedCount, 1)
    }
    XCTAssertEqual(result.compartmentsRequiringReview.count, 6)
  }

  func testMissingDoseIsMismatch() {
    let detections = Fixtures.layout.allCompartments.filter { $0 != target }.flatMap {
      Fixtures.detections("pill", count: 1, in: $0)
    }
    let result = evaluate(detections)
    XCTAssertEqual(result.status, .mismatch)
    let verdict = result.verdict(for: target)
    XCTAssertEqual(verdict?.status, .mismatch)
    XCTAssertTrue(verdict?.findings.contains(.missing(expected: 1, observed: 0)) == true)
  }

  func testExtraDoseIsMismatch() {
    let result = evaluate(
      fullPack
        + Fixtures.detections("pill", count: 1, in: target).map {
          var shifted = $0
          shifted.boundingBox.x += 0.03
          return shifted
        })
    XCTAssertEqual(result.verdict(for: target)?.status, .mismatch)
    XCTAssertTrue(result.verdict(for: target)?.findings.contains(.extra(expected: 1, observed: 2)) == true)
  }

  func testFramesThatDisagreeRequireReview() {
    let frames = [
      Fixtures.frame(fullPack),
      Fixtures.frame(fullPack + Fixtures.detections("pill", count: 2, in: target).suffix(1)),
      Fixtures.frame(fullPack),
    ]
    let result = Fixtures.engine().evaluate(profile: Fixtures.profile(default: one), frames: frames)
    let verdict = result.verdict(for: target)
    XCTAssertEqual(verdict?.status, .needsReview)
    XCTAssertTrue(verdict?.findings.contains(.unstableAcrossFrames) == true)
    XCTAssertNil(verdict?.observedCount)
    XCTAssertEqual(result.status, .needsReview)
  }

  func testLowConfidenceObjectIsNeverCountedSilently() {
    let result = evaluate(fullPack + Fixtures.detections("pill", count: 2, in: target, confidence: 0.4).suffix(1))
    let verdict = result.verdict(for: target)
    XCTAssertEqual(verdict?.status, .needsReview)
    XCTAssertTrue(verdict?.findings.contains(.lowConfidenceObject(count: 1)) == true)
    XCTAssertEqual(verdict?.observedCount, 1)
  }

  func testObjectOnBorderFlagsBothCompartments() {
    let left = CompartmentIndex.at(0, 0)
    let cell = Fixtures.layout.cellRect(left)
    let onBorder = Fixtures.detection("pill", at: Point2D(x: cell.maxX - 0.01, y: cell.center.y))
    let result = evaluate(fullPack + [onBorder])
    for index in [left, target] {
      XCTAssertEqual(result.verdict(for: index)?.status, .needsReview, "\(index)")
      XCTAssertTrue(result.verdict(for: index)?.findings.contains(.objectOnBorder(count: 1)) == true)
    }
  }

  func testForeignObjectAndBrokenDoseAreMismatches() {
    let other = CompartmentIndex.at(1, 2)
    let result = evaluate(
      fullPack + Fixtures.detections("foreign", count: 1, in: target)
        + Fixtures.detections("broken", count: 1, in: other))
    XCTAssertTrue(result.verdict(for: target)?.findings.contains(.foreignObject(count: 1)) == true)
    XCTAssertEqual(result.verdict(for: target)?.status, .mismatch)
    XCTAssertTrue(result.verdict(for: other)?.findings.contains(.brokenDose(count: 1)) == true)
    XCTAssertEqual(result.verdict(for: other)?.status, .mismatch)
  }

  func testIgnoredLabelsDoNotCount() {
    let result = evaluate(fullPack + Fixtures.detections("background", count: 1, in: target))
    XCTAssertEqual(result.verdict(for: target)?.status, .countMatched)
  }

  func testUnknownLabelsCountAsPills() {
    let result = evaluate(
      fullPack
        + Fixtures.detections("mystery", count: 1, in: target).map {
          var shifted = $0
          shifted.boundingBox.x += 0.03
          return shifted
        })
    XCTAssertEqual(result.verdict(for: target)?.status, .mismatch)
  }

  func testTooFewUsableFramesRequiresRetake() {
    let blurry = CaptureQualityAssessment(issues: [.blurry, .glare], metrics: nil)
    let lost = FrameObservation(
      capturedAt: Fixtures.fixedDate, quality: .accepted, registration: .rejected([.packTooSmall]), detections: [])
    let frames = [Fixtures.frame(fullPack), Fixtures.frame(fullPack, quality: blurry), lost, Fixtures.frame(fullPack)]
    let result = Fixtures.engine().evaluate(profile: Fixtures.profile(default: one), frames: frames)
    XCTAssertEqual(result.status, .retakeRequired)
    XCTAssertEqual(
      result.packFindings,
      [
        .insufficientUsableFrames(usable: 2, required: 3), .captureIssues([.blurry, .glare]),
        .registrationIssues([.packTooSmall]),
      ])
    XCTAssertTrue(result.compartments.allSatisfy { $0.findings == [.notEvaluated] && $0.status == .needsReview })
  }

  func testOnlyTheMostRecentUsableFramesAreEvaluated() {
    let stale = Fixtures.frame(fullPack + Fixtures.detections("pill", count: 2, in: target).suffix(1))
    let frames = [stale] + Fixtures.frames(fullPack)
    let result = Fixtures.engine().evaluate(profile: Fixtures.profile(default: one), frames: frames)
    XCTAssertEqual(result.status, .countMatched)
  }

  func testWithoutModelNothingIsEvaluated() {
    let result = evaluate(fullPack, engine: Fixtures.engine(model: nil))
    XCTAssertEqual(result.status, .needsReview)
    XCTAssertEqual(result.packFindings, [.modelUnavailable])
    XCTAssertEqual(result.capability, .unavailable)
  }

  func testProfileForAnotherLayoutIsNotEvaluated() {
    var profile = Fixtures.profile(default: one)
    profile.layoutID = "other"
    let result = evaluate(fullPack, profile: profile)
    XCTAssertEqual(result.status, .needsReview)
    XCTAssertEqual(result.packFindings, [.profileLayoutMismatch])
  }

  func testUncalibratedLayoutCannotProduceAcceptedResults() {
    var layout = Fixtures.layout
    layout.isCalibrated = false
    let result = evaluate(fullPack, engine: Fixtures.engine(layout: layout))
    XCTAssertEqual(result.status, .needsReview)
    XCTAssertTrue(result.compartments.allSatisfy { $0.findings.contains(.layoutUncalibrated) })

    let missing = evaluate(Array(fullPack.dropFirst()), engine: Fixtures.engine(layout: layout))
    XCTAssertEqual(missing.status, .mismatch)
  }

  func testObjectsOutsideTheGridSendPackToReview() {
    var layout = Fixtures.layout
    layout.gridRegion = Rect2D(x: 0.1, y: 0.1, width: 0.8, height: 0.8)
    let engine = Fixtures.engine(layout: layout)
    let inside = layout.allCompartments.map { index in
      Fixtures.detection("pill", at: layout.cellRect(index).center)
    }
    let loose = Fixtures.detection("pill", at: Point2D(x: 0.03, y: 0.5))
    let result = engine.evaluate(profile: Fixtures.profile(default: one), frames: Fixtures.frames(inside + [loose]))
    XCTAssertEqual(result.packFindings, [.objectsOutsideCompartments(count: 1)])
    XCTAssertEqual(result.status, .needsReview)
    XCTAssertTrue(result.compartments.allSatisfy { $0.status == .countMatched })
  }

  func testExpectedEmptyCompartmentsMustBeEmpty() {
    let profile = Fixtures.profile(default: [], overrides: [target: one])
    let ok = evaluate(Fixtures.detections("pill", count: 1, in: target), profile: profile)
    XCTAssertEqual(ok.status, .countMatched)

    let stray = evaluate(fullPack, profile: profile)
    XCTAssertEqual(stray.status, .mismatch)
    XCTAssertEqual(stray.verdict(for: .at(0, 0))?.findings.first, .extra(expected: 0, observed: 1))
  }

  // MARK: - Identity

  private var identityProfile: PackProfile {
    Fixtures.profile(
      default: [],
      overrides: [
        target: [
          ExpectedItem(medicationID: Fixtures.metformin, quantity: 1),
          ExpectedItem(medicationID: Fixtures.atorvastatin, quantity: 1),
        ]
      ])
  }

  private var identityEngine: VerificationEngine {
    Fixtures.engine(model: Fixtures.identityModel([Fixtures.metformin, Fixtures.atorvastatin]))
  }

  func testReleasedIdentityModelVerifiesMatchingCompartment() {
    let cell = Fixtures.layout.cellRect(target)
    let detections = [
      Fixtures.detection("metformin", at: Point2D(x: cell.center.x - 0.03, y: cell.center.y)),
      Fixtures.detection("atorvastatin", at: Point2D(x: cell.center.x + 0.03, y: cell.center.y)),
    ]
    let result = evaluate(detections, profile: identityProfile, engine: identityEngine)
    XCTAssertEqual(result.verdict(for: target)?.status, .verified)
    XCTAssertEqual(result.verdict(for: target)?.findings, [])
    XCTAssertEqual(result.verdict(for: target)?.observedMedications, [Fixtures.metformin: 1, Fixtures.atorvastatin: 1])
    XCTAssertEqual(result.status, .verified)
    XCTAssertEqual(result.compartmentsRequiringReview, [])
  }

  func testWrongMedicationWithRightCountIsMismatch() {
    let cell = Fixtures.layout.cellRect(target)
    let detections = [
      Fixtures.detection("metformin", at: Point2D(x: cell.center.x - 0.03, y: cell.center.y)),
      Fixtures.detection("metformin", at: Point2D(x: cell.center.x + 0.03, y: cell.center.y)),
    ]
    let verdict = evaluate(detections, profile: identityProfile, engine: identityEngine).verdict(for: target)
    XCTAssertEqual(verdict?.status, .mismatch)
    XCTAssertEqual(
      verdict?.findings,
      [
        .wrongQuantity(Fixtures.atorvastatin, expected: 1, observed: 0),
        .wrongQuantity(Fixtures.metformin, expected: 1, observed: 2),
      ])
  }

  func testIdentifiedUnexpectedMedicationIsMismatch() {
    let model = Fixtures.identityModel([Fixtures.metformin, Fixtures.atorvastatin, Fixtures.aspirin])
    let cell = Fixtures.layout.cellRect(target)
    let detections = [
      Fixtures.detection("metformin", at: Point2D(x: cell.center.x - 0.03, y: cell.center.y)),
      Fixtures.detection("aspirin", at: Point2D(x: cell.center.x + 0.03, y: cell.center.y)),
    ]
    let verdict = evaluate(detections, profile: identityProfile, engine: Fixtures.engine(model: model))
      .verdict(for: target)
    XCTAssertEqual(verdict?.status, .mismatch)
    XCTAssertTrue(verdict?.findings.contains(.unexpectedMedication(Fixtures.aspirin, observed: 1)) == true)
  }

  func testMedicationOutsideGrantedCapabilityIsTreatedAsGenericPill() {
    let model = Fixtures.identityModel([Fixtures.metformin])
    let cell = Fixtures.layout.cellRect(target)
    let detections = [
      Fixtures.detection("metformin", at: Point2D(x: cell.center.x - 0.03, y: cell.center.y)),
      Fixtures.detection("atorvastatin", at: Point2D(x: cell.center.x + 0.03, y: cell.center.y)),
    ]
    let verdict = evaluate(detections, profile: identityProfile, engine: Fixtures.engine(model: model))
      .verdict(for: target)
    XCTAssertEqual(verdict?.status, .countMatched)
    XCTAssertEqual(verdict?.findings, [.identityNotVerified])
    XCTAssertEqual(verdict?.observedMedications, [Fixtures.metformin: 1])
  }
}
