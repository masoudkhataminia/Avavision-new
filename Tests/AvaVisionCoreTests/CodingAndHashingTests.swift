import Foundation
import XCTest

@testable import AvaVisionCore

final class CodingAndHashingTests: XCTestCase {
  func testKnownSHA256Vector() {
    XCTAssertEqual(
      ContentHasher.sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  }

  func testSHA256MatchesReferenceVectors() {
    let expected: [Int: String] = [
      0: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      55: "e7313d333c272e639f790978283f9eb392e843d0f29b7016828bb1daa4aac70b",
      56: "4324d65f3c103567f5589c710bc08f8523f929a9272e3af36fc968e52abc6c27",
      63: "81c80242132f230c3bd41b3e63bbcff16107339549214a99614ff26664625055",
      64: "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241",
      65: "aacca6ff74fdbb296d165a45cecfa04e5127bc008770fbbdd48006f2d2fae95e",
      119: "9ce7368e4daf32341631b492e80359dc9f594b48453cd0dd5bf0b19279cc177e",
      120: "7836b787757e95e58b3ca5aec90b1b004e8deba1e50e9675af9cabf1a13a04b5",
      1000: "1e9bc38cbf860b9ec31918b065f9b52476c549a782e0e7990bed8ce3868d2371",
    ]
    for (length, digest) in expected {
      let data = Data((0..<length).map { UInt8(($0 * 7 + 3) % 256) })
      XCTAssertEqual(ContentHasher.sha256Hex(data), digest, "length \(length)")

      // Feeding the same bytes in uneven chunks must give the same digest.
      var hasher = SHA256Hasher()
      var offset = 0
      var chunk = 1
      while offset < data.count {
        let end = min(data.count, offset + chunk)
        hasher.update(data[offset..<end])
        offset = end
        chunk = chunk * 3 % 97 + 1
      }
      let streamed = hasher.finalize().map { String(format: "%02x", $0) }.joined()
      XCTAssertEqual(streamed, digest, "streamed length \(length)")
    }
    XCTAssertEqual(
      ContentHasher.sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
    XCTAssertEqual(
      ContentHasher.sha256Hex(Data(repeating: UInt8(ascii: "a"), count: 1_000_000)),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
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
