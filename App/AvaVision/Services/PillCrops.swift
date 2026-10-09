import AvaVisionCore
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Pill image crops: cut from frames for embedding, and stored so the brain can re-learn them
/// with a newer embedder. Crops show a single pill and never the pack label.
enum PillCrops {
  /// A square crop around `box` (normalized, origin top-left), padded so the pill's outline is kept.
  static func crop(_ image: CGImage, box: Rect2D, padding: Double = 0.2) -> CGImage? {
    let width = Double(image.width)
    let height = Double(image.height)
    let side = max(box.width * width, box.height * height) * (1 + 2 * padding)
    let center = box.center
    let rect = CGRect(
      x: center.x * width - side / 2, y: center.y * height - side / 2, width: side, height: side
    ).integral.intersection(CGRect(x: 0, y: 0, width: width, height: height))
    guard rect.width >= 4, rect.height >= 4 else { return nil }
    return image.cropping(to: rect)
  }

  static func jpegData(_ image: CGImage, quality: Double = 0.85) -> Data? {
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil)
    else { return nil }
    CGImageDestinationAddImage(
      destination, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    return CGImageDestinationFinalize(destination) ? data as Data : nil
  }

  static func image(contentsOf url: URL) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
  }
}

/// A pill found in a frame, with its crop and the brain's opinion.
struct IdentifiedPill: @unchecked Sendable {
  let detectionIndex: Int
  let compartment: CompartmentIndex
  let crop: CGImage
  let embedding: Embedding
  let identity: IdentityEvidence
}

/// Asks the brain about every confident pill clearly inside a compartment.
struct PillIdentifier: @unchecked Sendable {
  let embedder: PillEmbedder
  let classifier: IdentityClassifier

  func identify(
    _ observation: FrameObservation, image: CGImage, layout: PackLayout, model: ActiveModel,
    minimumConfidence: Double
  ) -> (observation: FrameObservation, pills: [IdentifiedPill]) {
    let located = CompartmentAssigner.pillCompartments(
      in: observation, layout: layout, minimumConfidence: minimumConfidence, meaning: model.meaning(of:))
    var updated = observation
    var pills: [IdentifiedPill] = []
    for (index, compartment) in located.sorted(by: { $0.key < $1.key }) {
      guard let crop = PillCrops.crop(image, box: observation.detections[index].boundingBox),
        let embedding = try? embedder.embed(crop)
      else { continue }
      let identity = classifier.classify(embedding)
      updated.detections[index].identity = identity
      pills.append(
        IdentifiedPill(
          detectionIndex: index, compartment: compartment, crop: crop, embedding: embedding, identity: identity))
    }
    return (updated, pills)
  }
}
