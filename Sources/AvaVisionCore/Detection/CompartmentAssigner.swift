import Foundation

/// A detection after label resolution and mapping into pack coordinates.
public struct PlacedObject: Hashable, Sendable {
  public var meaning: LabelMeaning
  public var confidence: Double
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
    for detection in detections {
      let resolved = meaning(detection.label)
      if resolved == .ignore { continue }
      guard let center = registration.imageToPack.apply(detection.boundingBox.center) else {
        assignment.outsideGrid.append(
          PlacedObject(
            meaning: resolved, confidence: detection.confidence,
            packCenter: nil, location: .outsideGrid))
        continue
      }
      let location = layout.locate(center)
      let object = PlacedObject(
        meaning: resolved, confidence: detection.confidence, packCenter: center, location: location)
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
}
