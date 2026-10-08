import Foundation
import XCTest

@testable import AvaVisionCore

final class SignOffTests: XCTestCase {
  private let one = [ExpectedItem(medicationID: Fixtures.metformin, quantity: 1)]

  private var countMatchedResult: PackVerificationResult {
    let detections = Fixtures.layout.allCompartments.flatMap { Fixtures.detections("pill", count: 1, in: $0) }
    return Fixtures.engine().evaluate(
      profile: Fixtures.profile(default: one), frames: Fixtures.frames(detections), evaluatedAt: Fixtures.fixedDate)
  }

  private func signOff(
    _ decision: SignOffDecision, reviews: [CompartmentReview], pharmacist: String = "MK", acknowledged: Bool = false
  ) -> PharmacistSignOff {
    PharmacistSignOff(
      pharmacistIdentifier: pharmacist, decision: decision, reviews: reviews, acknowledgedPackFindings: acknowledged,
      signedAt: Fixtures.fixedDate)
  }

  private func allReviewed(_ outcome: CompartmentReviewOutcome = .confirmedCorrect) -> [CompartmentReview] {
    Fixtures.layout.allCompartments.map { CompartmentReview(compartment: $0, outcome: outcome) }
  }

  func testReleaseRequiresEveryUnverifiedCompartmentToBeReviewed() {
    let result = countMatchedResult
    XCTAssertThrowsError(
      try SignOffValidator.validate(signOff(.released, reviews: Array(allReviewed().dropLast())), for: result)
    ) {
      XCTAssertEqual($0 as? SignOffError, .unreviewedCompartments([.at(1, 2)]))
    }
    XCTAssertNoThrow(try SignOffValidator.validate(signOff(.released, reviews: allReviewed()), for: result))
  }

  func testUnresolvedCompartmentBlocksRelease() {
    var reviews = allReviewed()
    reviews[0].outcome = .unresolved
    XCTAssertThrowsError(try SignOffValidator.validate(signOff(.released, reviews: reviews), for: countMatchedResult)) {
      XCTAssertEqual($0 as? SignOffError, .unresolvedCompartments([.at(0, 0)]))
    }
    XCTAssertNoThrow(try SignOffValidator.validate(signOff(.withheld, reviews: reviews), for: countMatchedResult))
  }

  func testPharmacistIdentifierIsMandatory() {
    XCTAssertThrowsError(
      try SignOffValidator.validate(signOff(.withheld, reviews: [], pharmacist: "  "), for: countMatchedResult)
    ) {
      XCTAssertEqual($0 as? SignOffError, .missingPharmacistIdentifier)
    }
  }

  func testReviewsMustReferToExistingCompartments() {
    let stray = [CompartmentReview(compartment: .at(9, 9), outcome: .confirmedCorrect)]
    XCTAssertThrowsError(try SignOffValidator.validate(signOff(.withheld, reviews: stray), for: countMatchedResult)) {
      XCTAssertEqual($0 as? SignOffError, .reviewForUnknownCompartment(.at(9, 9)))
    }
  }

  func testPackFindingsMustBeAcknowledged() {
    var result = countMatchedResult
    result.packFindings = [.objectsOutsideCompartments(count: 1)]
    XCTAssertThrowsError(try SignOffValidator.validate(signOff(.released, reviews: allReviewed()), for: result)) {
      XCTAssertEqual($0 as? SignOffError, .packFindingsNotAcknowledged)
    }
    XCTAssertNoThrow(
      try SignOffValidator.validate(signOff(.released, reviews: allReviewed(), acknowledged: true), for: result))
  }
}

final class CheckSessionTests: XCTestCase {
  private let one = [ExpectedItem(medicationID: Fixtures.metformin, quantity: 1)]

  private func makeSession() throws -> CheckSession {
    try CheckSession(layout: Fixtures.layout, profile: Fixtures.profile(default: one), catalog: Fixtures.catalog)
  }

  func testFullLifecycleProducesAuditRecord() throws {
    var session = try makeSession()
    let detections = Fixtures.layout.allCompartments.flatMap { Fixtures.detections("pill", count: 1, in: $0) }
    for var frame in Fixtures.frames(detections) {
      frame.imageSHA256 = "img"
      try session.record(frame)
    }
    XCTAssertTrue(session.isReadyToAnalyze())
    let result = try session.analyze(with: Fixtures.engine())
    XCTAssertEqual(session.phase, .analyzed)
    XCTAssertThrowsError(try session.record(Fixtures.frame([])))

    let reviews = result.compartmentsRequiringReview.map {
      CompartmentReview(compartment: $0, outcome: .confirmedCorrect)
    }
    let signOff = PharmacistSignOff(
      pharmacistIdentifier: "MK", decision: .released, reviews: reviews, acknowledgedPackFindings: false)
    let record = try session.complete(
      with: signOff, appVersion: "1.0 (1)", deviceIdentifier: "device", model: ModelSummary(Fixtures.countOnlyModel))
    XCTAssertEqual(session.phase, .completed)
    XCTAssertEqual(record.result, result)
    XCTAssertEqual(record.frameImageSHA256s, ["img", "img", "img"])
    XCTAssertEqual(record.model?.capability, .countOnly)
  }

  func testInvalidSignOffKeepsSessionOpen() throws {
    var session = try makeSession()
    try session.analyze(with: Fixtures.engine())
    let incomplete = PharmacistSignOff(
      pharmacistIdentifier: "MK", decision: .released, reviews: [], acknowledgedPackFindings: false)
    XCTAssertThrowsError(try session.complete(with: incomplete, appVersion: "1", deviceIdentifier: "d", model: nil))
    XCTAssertEqual(session.phase, .analyzed)
  }

  func testRetakeClearsFramesAndResult() throws {
    var session = try makeSession()
    try session.record(Fixtures.frame([]))
    try session.analyze(with: Fixtures.engine())
    try session.retake()
    XCTAssertEqual(session.phase, .capturing)
    XCTAssertTrue(session.frames.isEmpty)
    XCTAssertNil(session.result)
    XCTAssertThrowsError(try session.retake())
  }

  func testInvalidProfileCannotStartSession() {
    var profile = Fixtures.profile(default: one)
    profile.compartments.removeLast()
    XCTAssertThrowsError(try CheckSession(layout: Fixtures.layout, profile: profile, catalog: Fixtures.catalog)) {
      XCTAssertEqual($0 as? CheckSession.SessionError, .invalidProfile([.unspecifiedCompartment(.at(1, 2))]))
    }
  }

  func testFrameRetentionIsBounded() throws {
    var session = try makeSession()
    for _ in 0..<(CheckSession.maximumRetainedFrames + 5) { try session.record(Fixtures.frame([])) }
    XCTAssertEqual(session.frames.count, CheckSession.maximumRetainedFrames)
  }
}

final class AuditTests: XCTestCase {
  private var temporaryDirectory: URL!

  override func setUpWithError() throws {
    temporaryDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: temporaryDirectory)
  }

  private func record(_ pharmacist: String = "MK") -> CheckRecord {
    let profile = Fixtures.profile()
    let result = Fixtures.engine().evaluate(profile: profile, frames: [], evaluatedAt: Fixtures.fixedDate)
    return CheckRecord(
      createdAt: Fixtures.fixedDate, appVersion: "1.0", deviceIdentifier: "device", layout: Fixtures.layout,
      profile: profile, model: nil, result: result,
      signOff: PharmacistSignOff(
        pharmacistIdentifier: pharmacist, decision: .withheld, reviews: [], acknowledgedPackFindings: false,
        signedAt: Fixtures.fixedDate),
      frameImageSHA256s: [])
  }

  func testChainVerifiesAndDetectsTampering() throws {
    var chain = AuditChain()
    try chain.append(record("A"))
    try chain.append(record("B"))
    try chain.append(record("C"))
    XCTAssertEqual(chain.verify(), .intact(entries: 3))
    XCTAssertEqual(try chain.records().map(\.signOff.pharmacistIdentifier), ["A", "B", "C"])

    var entries = chain.entries
    let tampered = entries[1].payload.replacingOccurrences(of: "\"B\"", with: "\"Z\"")
    entries[1] = AuditEntry(
      sequence: 1, previousHash: entries[1].previousHash, payload: tampered, hash: entries[1].hash)
    XCTAssertEqual(AuditChain(entries: entries).verify(), .broken(sequence: 1, defect: .hashMismatch))

    var removed = chain.entries
    removed.remove(at: 1)
    XCTAssertEqual(AuditChain(entries: removed).verify(), .broken(sequence: 1, defect: .sequenceGap))
  }

  func testStorePersistsAppendsAndRefusesBrokenChain() async throws {
    let url = temporaryDirectory.appendingPathComponent("audit/log.jsonl")
    let store = AuditLogStore(fileURL: url)
    let firstRecord = record("A")
    try await store.append(firstRecord)
    try await store.append(record("B"))

    let reopened = AuditLogStore(fileURL: url)
    let verification = try await reopened.verify()
    XCTAssertEqual(verification, .intact(entries: 2))
    let names = try await reopened.records().map(\.signOff.pharmacistIdentifier)
    XCTAssertEqual(names, ["A", "B"])
    let first = try await reopened.records().first
    XCTAssertEqual(first, firstRecord)

    let text = try String(contentsOf: url, encoding: .utf8)
    try text.replacingOccurrences(of: "\\\"A\\\"", with: "\\\"Q\\\"").write(to: url, atomically: true, encoding: .utf8)
    let tampered = AuditLogStore(fileURL: url)
    do {
      try await tampered.append(record("C"))
      XCTFail("Appending to a broken chain must fail")
    } catch {
      XCTAssertEqual(error as? AuditStoreError, .chainBroken(sequence: 0, defect: .hashMismatch))
    }
  }

  func testJSONFileStoreRoundTrip() async throws {
    let store = JSONFileStore<MedicationCatalog>(fileURL: temporaryDirectory.appendingPathComponent("a/catalog.json"))
    let empty = try await store.load()
    XCTAssertNil(empty)
    try await store.save(Fixtures.catalog)
    let loaded = try await store.load()
    XCTAssertEqual(loaded, Fixtures.catalog)
  }
}
