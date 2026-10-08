import Foundation
import XCTest

@testable import AvaVisionCore

/// Shared builders for engine-level tests.
enum Fixtures {
  static let metformin: MedicationID = "metformin-500"
  static let atorvastatin: MedicationID = "atorvastatin-20"
  static let aspirin: MedicationID = "aspirin-100"

  static let catalog = MedicationCatalog(medications: [
    Medication(id: metformin, name: "Metformin", strength: "500 mg"),
    Medication(id: atorvastatin, name: "Atorvastatin", strength: "20 mg"),
    Medication(id: aspirin, name: "Aspirin", strength: "100 mg"),
  ])

  /// A small 2×3 layout covering the whole card, calibrated so results can be accepted.
  static let layout = PackLayout(
    id: "test-2x3", displayName: "Test 2×3", rows: 2, columns: 3,
    rowLabels: ["AM", "PM"], columnLabels: ["D1", "D2", "D3"],
    widthMillimetres: 150, heightMillimetres: 100,
    gridRegion: Rect2D(x: 0, y: 0, width: 1, height: 1), borderBand: 0.1, isCalibrated: true)

  static let fixedDate = Date(timeIntervalSince1970: 1_800_000_000)

  /// Pack outline occupying the centre of a 1500×1000 image, square-on.
  static let quad = Quad(
    topLeft: Point2D(x: 0.1, y: 0.1), topRight: Point2D(x: 0.9, y: 0.1),
    bottomRight: Point2D(x: 0.9, y: 0.9), bottomLeft: Point2D(x: 0.1, y: 0.9))

  static var registration: PackRegistration {
    guard
      case .registered(let registration) = PackRegistrar.register(
        quad: quad, detectorConfidence: 0.95, imageWidth: 1500, imageHeight: 1000, layout: layout)
    else { fatalError("fixture quad must register") }
    return registration
  }

  /// A detection whose centre lies at `point` in pack coordinates.
  static func detection(_ label: String, at point: Point2D, confidence: Double = 0.9) -> Detection {
    let packToImage = registration.imageToPack.inverse!
    let center = packToImage.apply(point)!
    return Detection(
      label: label, confidence: confidence,
      boundingBox: Rect2D(x: center.x - 0.01, y: center.y - 0.01, width: 0.02, height: 0.02))
  }

  /// `count` detections spread inside the centre of a compartment.
  static func detections(_ label: String, count: Int, in index: CompartmentIndex, confidence: Double = 0.9)
    -> [Detection]
  {
    let cell = layout.cellRect(index)
    return (0..<count).map { i in
      let offset = (Double(i) - Double(count - 1) / 2) * cell.width * 0.12
      return detection(label, at: Point2D(x: cell.center.x + offset, y: cell.center.y), confidence: confidence)
    }
  }

  static func frame(_ detections: [Detection], quality: CaptureQualityAssessment = .accepted) -> FrameObservation {
    FrameObservation(
      capturedAt: fixedDate, quality: quality, registration: .registered(registration), detections: detections)
  }

  static func frames(_ detections: [Detection], count: Int = 3) -> [FrameObservation] {
    Array(repeating: frame(detections), count: count)
  }

  /// Profile expecting `items` in every compartment unless overridden.
  static func profile(
    default items: [ExpectedItem] = [], overrides: [CompartmentIndex: [ExpectedItem]] = [:]
  ) -> PackProfile {
    var profile = PackProfile(reference: "PACK-001", layoutID: layout.id, compartments: [], createdAt: fixedDate)
    for index in layout.allCompartments {
      profile.setItems(overrides[index] ?? items, for: index)
    }
    return profile
  }

  static let genericLabels: [String: LabelMeaning] = [
    "pill": .pill, "broken": .broken, "foreign": .foreign, "background": .ignore,
  ]

  static let identityLabels: [String: LabelMeaning] = genericLabels.merging([
    "metformin": .medication(metformin),
    "atorvastatin": .medication(atorvastatin),
    "aspirin": .medication(aspirin),
  ]) { $1 }

  static func manifest(
    stage: ModelStage = .development, labels: [String: LabelMeaning] = genericLabels,
    evaluation: EvaluationReport? = nil
  ) -> ModelManifest {
    ModelManifest(
      modelID: "test-detector", version: "1.0.0", stage: stage, modelSHA256: "abc123", labels: labels,
      evaluation: evaluation)
  }

  static var countOnlyModel: ActiveModel {
    let manifest = manifest()
    return ActiveModel(manifest: manifest, decision: ReleaseGate().evaluate(manifest, computedModelSHA256: "abc123"))
  }

  /// A model granted identity for exactly `medications`, bypassing the gate.
  static func identityModel(_ medications: Set<MedicationID>) -> ActiveModel {
    ActiveModel(
      manifest: manifest(stage: .released, labels: identityLabels),
      decision: GateDecision(capability: .identity(medications), blockers: []))
  }

  static func engine(model: ActiveModel? = countOnlyModel, layout: PackLayout = layout) -> VerificationEngine {
    VerificationEngine(layout: layout, model: model)
  }
}

extension CompartmentIndex {
  static func at(_ row: Int, _ column: Int) -> CompartmentIndex {
    CompartmentIndex(row: row, column: column)
  }
}
