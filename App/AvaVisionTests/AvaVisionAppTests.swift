import AvaVisionCore
import CoreGraphics
import XCTest

@testable import AvaVision

final class FrameAnalyzerTests: XCTestCase {
  /// A light card with dark dots on a dark table, like a pack photographed from above.
  private func syntheticPack(width: Int = 1200, height: Int = 900) throws -> CGImage {
    let context = try XCTUnwrap(
      CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(CGColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let card = CGRect(x: 180, y: 180, width: 840, height: 560)
    context.setFillColor(CGColor(red: 0.92, green: 0.93, blue: 0.9, alpha: 1))
    context.fill(card)
    context.setFillColor(CGColor(red: 0.3, green: 0.35, blue: 0.4, alpha: 1))
    for row in 0..<4 {
      for column in 0..<7 {
        let x = card.minX + 60 + CGFloat(column) * 108
        let y = card.minY + 60 + CGFloat(row) * 120
        context.fillEllipse(in: CGRect(x: x, y: y, width: 40, height: 28))
      }
    }
    return try XCTUnwrap(context.makeImage())
  }

  func testFindsThePackOutlineWithoutAModel() throws {
    let analyzer = FrameAnalyzer(layout: .weekly7x4, orientation: .upright, visionModel: nil)
    let frame = analyzer.analyze(try syntheticPack())
    let registration = try XCTUnwrap(
      frame.observation.registration.registration, "registration: \(frame.observation.registration)")
    XCTAssertEqual(registration.quad.boundingBox.width, 0.7, accuracy: 0.05)
    XCTAssertEqual(registration.quad.boundingBox.height, 560.0 / 900.0, accuracy: 0.05)
    XCTAssertTrue(frame.observation.detections.isEmpty)
    XCTAssertNotNil(frame.observation.imageSHA256)
    XCTAssertNotNil(frame.observation.quality.metrics)
  }

  func testAutomaticOrientationTurnsPortraitOutlineForLandscapePack() {
    let quad = Quad(
      topLeft: Point2D(x: 0.2, y: 0.1), topRight: Point2D(x: 0.8, y: 0.1), bottomRight: Point2D(x: 0.8, y: 0.9),
      bottomLeft: Point2D(x: 0.2, y: 0.9))
    XCTAssertEqual(
      PackOrientation.automatic.quarterTurns(for: quad, imageWidth: 1000, imageHeight: 2000, layout: .weekly7x4), 1)
    XCTAssertEqual(
      PackOrientation.automatic.quarterTurns(for: quad, imageWidth: 3000, imageHeight: 1000, layout: .weekly7x4), 0)
    XCTAssertEqual(
      PackOrientation.halfTurn.quarterTurns(for: quad, imageWidth: 3000, imageHeight: 1000, layout: .weekly7x4), 2)
  }

  func testGrayConversionKeepsRowOrder() throws {
    let context = try XCTUnwrap(
      CGContext(
        data: nil, width: 1, height: 2, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 1, height: 1))  // bottom row
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 1, width: 1, height: 1))  // top row
    let gray = try XCTUnwrap(ImageConversion.grayImage(from: try XCTUnwrap(context.makeImage())))
    XCTAssertEqual(gray.pixels, [255, 0])
  }

  func testNoBundledModelMeansNotInstalled() {
    guard case .notInstalled = ModelLoader.load(bundle: Bundle(for: Self.self)) else {
      return XCTFail("Test bundle must not contain a model")
    }
  }
}

@MainActor
final class AppModelTests: XCTestCase {
  func testCatalogProfilesAndLayoutsPersist() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }

    let app = AppModel(storage: Storage(root: root))
    await app.load()
    app.upsertMedication(Medication(id: "metformin-500", name: "Metformin", strength: "500 mg"))
    // Whole-second date: stored dates keep millisecond precision.
    var profile = PackProfile.empty(reference: "PACK-1", layout: .weekly7x4)
    profile.createdAt = Date(timeIntervalSince1970: 1_800_000_000)
    profile.setItems(
      [ExpectedItem(medicationID: "metformin-500", quantity: 2)], for: CompartmentIndex(row: 0, column: 0))
    app.saveProfile(profile)
    var layout = PackLayout.weekly7x4
    layout.widthMillimetres = 260
    app.saveLayout(layout)
    await app.flushWrites()

    let reloaded = AppModel(storage: Storage(root: root))
    await reloaded.load()
    XCTAssertEqual(reloaded.catalog, app.catalog)
    XCTAssertEqual(reloaded.profiles, [profile])
    XCTAssertEqual(reloaded.layout(id: layout.id)?.widthMillimetres, 260)
    XCTAssertEqual(reloaded.issues(for: profile), [])
    XCTAssertTrue(reloaded.isMedicationInUse("metformin-500"))
  }

  func testWithoutModelTheEngineEvaluatesNothing() throws {
    let app = AppModel(storage: nil)
    let profile = PackProfile.empty(reference: "PACK-1", layout: .weekly7x4)
    let result = app.engine(for: .weekly7x4).evaluate(profile: profile, frames: [])
    XCTAssertEqual(result.packFindings, [.modelUnavailable])
    XCTAssertEqual(result.status, .needsReview)
  }
}
