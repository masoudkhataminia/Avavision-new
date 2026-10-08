import XCTest

@testable import AvaVisionCore

final class GeometryAndLayoutTests: XCTestCase {
  private let square = [Point2D(x: 0, y: 0), Point2D(x: 1, y: 0), Point2D(x: 1, y: 1), Point2D(x: 0, y: 1)]

  func testHomographyMapsCornersAndInverts() throws {
    let skewed = [
      Point2D(x: 0.12, y: 0.2), Point2D(x: 0.85, y: 0.1), Point2D(x: 0.95, y: 0.9), Point2D(x: 0.05, y: 0.8),
    ]
    let h = try XCTUnwrap(Homography(mapping: skewed, to: square))
    for (source, target) in zip(skewed, square) {
      let mapped = try XCTUnwrap(h.apply(source))
      XCTAssertEqual(mapped.x, target.x, accuracy: 1e-9)
      XCTAssertEqual(mapped.y, target.y, accuracy: 1e-9)
    }
    let inverse = try XCTUnwrap(h.inverse)
    let point = Point2D(x: 0.37, y: 0.61)
    let roundTrip = try XCTUnwrap(inverse.apply(try XCTUnwrap(h.apply(point))))
    XCTAssertEqual(roundTrip.x, point.x, accuracy: 1e-9)
    XCTAssertEqual(roundTrip.y, point.y, accuracy: 1e-9)
  }

  func testDegenerateCorrespondencesAreRejected() {
    let collinear = [Point2D(x: 0, y: 0), Point2D(x: 0.5, y: 0.5), Point2D(x: 1, y: 1), Point2D(x: 0.2, y: 0.2)]
    XCTAssertNil(Homography(mapping: collinear, to: square))
    XCTAssertNil(Homography(mapping: Array(square.prefix(3)), to: square))
  }

  func testQuadProperties() {
    let quad = Quad(
      topLeft: Point2D(x: 0, y: 0), topRight: Point2D(x: 2, y: 0), bottomRight: Point2D(x: 2, y: 1),
      bottomLeft: Point2D(x: 0, y: 1))
    XCTAssertEqual(quad.area, 2, accuracy: 1e-12)
    XCTAssertTrue(quad.isConvex)
    for angle in quad.interiorAngles { XCTAssertEqual(angle, 90, accuracy: 1e-9) }
    let bowtie = Quad(
      topLeft: Point2D(x: 0, y: 0), topRight: Point2D(x: 1, y: 1), bottomRight: Point2D(x: 1, y: 0),
      bottomLeft: Point2D(x: 0, y: 1))
    XCTAssertFalse(bowtie.isConvex)
  }

  func testQuadRotationRelabelsCorners() {
    let quad = Fixtures.quad
    let turned = quad.rotated(quarterTurns: 1)
    XCTAssertEqual(turned.topLeft, quad.bottomLeft)
    XCTAssertEqual(turned.topRight, quad.topLeft)
    XCTAssertEqual(turned.bottomRight, quad.topRight)
    XCTAssertEqual(turned.bottomLeft, quad.bottomRight)
    XCTAssertEqual(quad.rotated(quarterTurns: 4), quad)
    XCTAssertEqual(quad.rotated(quarterTurns: -1), quad.rotated(quarterTurns: 3))
    XCTAssertEqual(quad.rotated(quarterTurns: 2).topLeft, quad.bottomRight)
  }

  func testWeeklyLayoutIsValidButUncalibrated() {
    let layout = PackLayout.weekly7x4
    XCTAssertEqual(layout.validationErrors(), [])
    XCTAssertEqual(layout.allCompartments.count, 28)
    XCTAssertFalse(layout.isCalibrated)
    XCTAssertEqual(layout.label(for: .at(3, 6)), "Day 7 · Bedtime")
  }

  func testLocateDistinguishesInsideBorderAndOutside() {
    let layout = Fixtures.layout
    XCTAssertEqual(layout.locate(layout.cellRect(.at(1, 1)).center), .inside(.at(1, 1)))
    let boundary = Point2D(x: 1.0 / 3.0 + 0.01, y: 0.25)
    XCTAssertEqual(layout.locate(boundary), .border([.at(0, 0), .at(0, 1)]))
    let corner = Point2D(x: 2.0 / 3.0 - 0.01, y: 0.5 + 0.01)
    XCTAssertEqual(layout.locate(corner), .border([.at(0, 1), .at(0, 2), .at(1, 1), .at(1, 2)]))
    XCTAssertEqual(layout.locate(Point2D(x: 0.01, y: 0.25)), .border([.at(0, 0)]))
    XCTAssertEqual(layout.locate(Point2D(x: 1.2, y: 0.5)), .outsideGrid)
  }

  func testLayoutValidation() {
    var layout = Fixtures.layout
    layout.rows = 0
    layout.rowLabels = ["x"]
    layout.gridRegion = Rect2D(x: 0.5, y: 0, width: 0.7, height: 1)
    layout.borderBand = 0.5
    XCTAssertEqual(
      layout.validationErrors(), [.invalidGridSize, .labelCountMismatch, .gridRegionOutsidePack, .invalidBorderBand])
  }

  func testProfileIssues() {
    var profile = Fixtures.profile(default: [ExpectedItem(medicationID: Fixtures.metformin, quantity: 1)])
    XCTAssertEqual(profile.issues(layout: Fixtures.layout, catalog: Fixtures.catalog), [])

    profile.reference = " "
    profile.compartments.removeAll { $0.compartment == .at(1, 2) }
    profile.compartments.append(CompartmentExpectation(compartment: .at(5, 5), items: []))
    profile.compartments.append(
      CompartmentExpectation(compartment: .at(0, 0), items: [ExpectedItem(medicationID: "unknown", quantity: 0)]))
    XCTAssertEqual(
      profile.issues(layout: Fixtures.layout, catalog: Fixtures.catalog),
      [
        .emptyReference, .compartmentOutsideLayout(.at(5, 5)), .duplicateCompartment(.at(0, 0)),
        .invalidQuantity(.at(0, 0), "unknown"), .unknownMedication("unknown"), .unspecifiedCompartment(.at(1, 2)),
      ])
  }

  func testEmptyProfileCoversEveryCompartment() {
    let profile = PackProfile.empty(reference: "P-1", layout: .weekly7x4)
    XCTAssertEqual(profile.issues(layout: .weekly7x4, catalog: MedicationCatalog()), [])
    XCTAssertEqual(profile.expectation(for: .at(2, 3))?.totalQuantity, 0)
  }

  func testCatalogUpsertReplacesAndSorts() {
    var catalog = Fixtures.catalog
    catalog.upsert(Medication(id: Fixtures.aspirin, name: "Aspirin", strength: "300 mg"))
    XCTAssertEqual(catalog.medications.map(\.name), ["Aspirin", "Atorvastatin", "Metformin"])
    XCTAssertEqual(catalog.medication(Fixtures.aspirin)?.displayName, "Aspirin 300 mg")
    catalog.remove(Fixtures.aspirin)
    XCTAssertFalse(catalog.contains(Fixtures.aspirin))
  }
}

final class PackRegistrationTests: XCTestCase {
  private func register(_ quad: Quad?, confidence: Double = 0.95, layout: PackLayout = Fixtures.layout)
    -> RegistrationOutcome
  {
    PackRegistrar.register(
      quad: quad, detectorConfidence: confidence, imageWidth: 1500, imageHeight: 1000, layout: layout)
  }

  private func quad(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> Quad {
    Quad(
      topLeft: Point2D(x: x0, y: y0), topRight: Point2D(x: x1, y: y0), bottomRight: Point2D(x: x1, y: y1),
      bottomLeft: Point2D(x: x0, y: y1))
  }

  func testWellPlacedPackRegisters() throws {
    let registration = try XCTUnwrap(register(Fixtures.quad).registration)
    let center = try XCTUnwrap(registration.imageToPack.apply(Point2D(x: 0.5, y: 0.5)))
    XCTAssertEqual(center.x, 0.5, accuracy: 1e-9)
    XCTAssertEqual(center.y, 0.5, accuracy: 1e-9)
  }

  func testRejections() {
    XCTAssertEqual(register(nil), .rejected([.packNotFound]))
    XCTAssertEqual(register(Fixtures.quad, confidence: 0.2), .rejected([.lowDetectorConfidence]))
    XCTAssertEqual(register(quad(0.4, 0.4, 0.6, 0.6)).issues, [.packTooSmall])
    XCTAssertEqual(register(quad(0.0, 0.1, 0.8, 0.9)).issues, [.packTouchesFrameEdge])
    let steep = Quad(
      topLeft: Point2D(x: 0.3, y: 0.1), topRight: Point2D(x: 0.7, y: 0.1), bottomRight: Point2D(x: 0.95, y: 0.9),
      bottomLeft: Point2D(x: 0.05, y: 0.9))
    XCTAssertTrue(register(steep).issues.contains(.perspectiveTooSteep))
    let bowtie = Quad(
      topLeft: Point2D(x: 0.1, y: 0.1), topRight: Point2D(x: 0.9, y: 0.9), bottomRight: Point2D(x: 0.9, y: 0.1),
      bottomLeft: Point2D(x: 0.1, y: 0.9))
    XCTAssertEqual(register(bowtie), .rejected([.packNotConvex]))
  }

  func testAspectRatioIsCheckedOnlyForCalibratedLayouts() {
    let portrait = quad(0.3, 0.05, 0.7, 0.95)
    XCTAssertEqual(register(portrait).issues, [.aspectRatioMismatch])
    var uncalibrated = Fixtures.layout
    uncalibrated.isCalibrated = false
    XCTAssertNotNil(register(portrait, layout: uncalibrated).registration)
  }
}
