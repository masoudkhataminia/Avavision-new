import Foundation

/// A detection after label resolution and mapping into pack coordinates.
public struct PlacedObject: Hashable, Sendable {
  /// Position of the detection in the frame's `detections`.
  public var detectionIndex: Int
  public var meaning: LabelMeaning
  public var confidence: Double
  public var identity: IdentityEvidence?
  /// `nil` when the point could not be mapped onto the pack plane.
  public var packCenter: Point2D?
  public var location: CellLocation
}

/// Detections of one frame, grouped by compartment.
public struct FrameAssignment: Sendable {
  /// Objects whose centre is clearly inside each compartment.
  public var inside: [CompartmentIndex: [PlacedObject]] = [:]
  /// Objects in a border band, listed under every compartment they might belong to.
  public var ambiguous: [CompartmentIndex: [PlacedObject]] = [:]
  /// Objects on the pack card but outside the grid, or that could not be mapped.
  public var outsideGrid: [PlacedObject] = []
}

public enum CompartmentAssigner {
  /// Maps each detection's box centre through the registration and assigns it to a compartment.
  /// Labels resolving to `.ignore` are dropped; everything else is kept, whatever its confidence,
  /// so low-confidence objects can trigger review instead of silently disappearing.
  public static func assign(
    _ detections: [Detection], registration: PackRegistration, layout: PackLayout,
    meaning: (String) -> LabelMeaning
  ) -> FrameAssignment {
    var assignment = FrameAssignment()
    for (detectionIndex, detection) in detections.enumerated() {
      let resolved = meaning(detection.label)
      if resolved == .ignore { continue }
      guard let center = registration.imageToPack.apply(detection.boundingBox.center) else {
        assignment.outsideGrid.append(
          PlacedObject(
            detectionIndex: detectionIndex, meaning: resolved, confidence: detection.confidence,
            identity: detection.identity, packCenter: nil, location: .outsideGrid))
        continue
      }
      let location = layout.locate(center)
      let object = PlacedObject(
        detectionIndex: detectionIndex, meaning: resolved, confidence: detection.confidence,
        identity: detection.identity, packCenter: center, location: location)
      switch location {
      case .inside(let index):
        assignment.inside[index, default: []].append(object)
      case .border(let candidates):
        for index in candidates { assignment.ambiguous[index, default: []].append(object) }
      case .outsideGrid:
        if Rect2D.unit.contains(center) { assignment.outsideGrid.append(object) }
      }
    }
    return assignment
  }

  /// Confident whole doses clearly inside one compartment, keyed by detection index.
  /// These are the pills the brain may learn from once a pharmacist confirms the compartment.
  public static func pillCompartments(
    in frame: FrameObservation, layout: PackLayout, minimumConfidence: Double, meaning: (String) -> LabelMeaning
  ) -> [Int: CompartmentIndex] {
    guard let registration = frame.registration.registration else { return [:] }
    let assignment = assign(frame.detections, registration: registration, layout: layout, meaning: meaning)
    var result: [Int: CompartmentIndex] = [:]
    for (index, objects) in assignment.inside {
      for object in objects where object.confidence >= minimumConfidence {
        switch object.meaning {
        case .pill, .medication: result[object.detectionIndex] = index
        case .broken, .foreign, .ignore: break
        }
      }
    }
    return result
  }
}
