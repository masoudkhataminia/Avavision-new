import Foundation
import XCTest

@testable import AvaVisionCore

final class CodingAndHashingTests: XCTestCase {
  func testKnownSHA256Vector() {
    XCTAssertEqual(
      ContentHasher.sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  }

  func testDirectoryHashIsStableAndContentSensitive() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("Model.mlmodelc")
    try FileManager.default.createDirectory(
      at: model.appendingPathComponent("weights"), withIntermediateDirectories: true)
    try Data("spec".utf8).write(to: model.appendingPathComponent("model.mil"))
    try Data("weights".utf8).write(to: model.appendingPathComponent("weights/weight.bin"))

    let first = try ContentHasher.sha256Hex(contentsOf: model)
    try Data("junk".utf8).write(to: model.appendingPathComponent(".DS_Store"))
    XCTAssertEqual(try ContentHasher.sha256Hex(contentsOf: model), first)

    try Data("weights-2".utf8).write(to: model.appendingPathComponent("weights/weight.bin"))
    XCTAssertNotEqual(try ContentHasher.sha256Hex(contentsOf: model), first)

    XCTAssertThrowsError(try ContentHasher.sha256Hex(contentsOf: root.appendingPathComponent("missing")))
  }

  func testLabelMeaningCoding() throws {
    let meanings: [LabelMeaning] = [.pill, .broken, .foreign, .ignore, .medication("metformin-500")]
    let data = try AvaVisionJSON.encoder().encode(meanings)
    XCTAssertEqual(
      String(decoding: data, as: UTF8.self), #"["pill","broken","foreign","ignore","medication:metformin-500"]"#)
    XCTAssertEqual(try AvaVisionJSON.decoder().decode([LabelMeaning].self, from: data), meanings)
    XCTAssertThrowsError(try AvaVisionJSON.decoder().decode([LabelMeaning].self, from: Data(#"["medication:"]"#.utf8)))
  }

  func testManifestDecodesFromHandWrittenJSON() throws {
    let json = """
      {
        "schemaVersion": 1,
        "modelID": "avavision-detector",
        "version": "2026.10.0",
        "stage": "development",
        "modelSHA256": "abc",
        "labels": { "pill": "pill", "metformin": "medication:metformin-500", "bg": "ignore" }
      }
      """
    let manifest = try AvaVisionJSON.decoder().decode(ModelManifest.self, from: Data(json.utf8))
    XCTAssertEqual(manifest.meaning(of: "metformin"), .medication("metformin-500"))
    XCTAssertEqual(manifest.medicationLabels, ["metformin-500"])
    XCTAssertNil(manifest.evaluation)
  }

  func testMedicationKeyedDictionariesEncodeAsObjects() throws {
    let data = try AvaVisionJSON.encoder().encode([MedicationID("b"): 2, MedicationID("a"): 1])
    XCTAssertEqual(String(decoding: data, as: UTF8.self), #"{"a":1,"b":2}"#)
  }

  func testDatesKeepFractionalSeconds() throws {
    let date = Date(timeIntervalSince1970: 1_800_000_000.25)
    let data = try AvaVisionJSON.encoder().encode([date])
    XCTAssertEqual(try AvaVisionJSON.decoder().decode([Date].self, from: data), [date])
    let legacy = Data(#"["2027-01-15T08:00:00Z"]"#.utf8)
    XCTAssertNoThrow(try AvaVisionJSON.decoder().decode([Date].self, from: legacy))
  }

  func testResultRoundTrip() throws {
    let detections = Fixtures.layout.allCompartments.flatMap { Fixtures.detections("pill", count: 1, in: $0) }
    let result = Fixtures.engine().evaluate(
      profile: Fixtures.profile(default: [ExpectedItem(medicationID: Fixtures.metformin, quantity: 1)]),
      frames: Fixtures.frames(detections), evaluatedAt: Fixtures.fixedDate)
    let data = try AvaVisionJSON.encoder().encode(result)
    XCTAssertEqual(try AvaVisionJSON.decoder().decode(PackVerificationResult.self, from: data), result)
  }
}
