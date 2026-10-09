import AvaVisionCore
import AvaVisionImaging
import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// Finds objects (pills, fragments, foreign bodies) in an upright image of a registered pack.
protocol ObjectDetector: Sendable {
  func detect(in image: CGImage, registration: PackRegistration, layout: PackLayout) throws -> [Detection]
}

/// A trained Core ML object-detection model run on the whole image.
final class CoreMLObjectDetector: ObjectDetector, @unchecked Sendable {
  /// Detections below this confidence are model noise and are not reported. The release gate
  /// evaluates the model at this same operating point.
  static let noiseFloor: Float = 0.25

  private let model: VNCoreMLModel

  init(model: VNCoreMLModel) {
    self.model = model
  }

  func detect(in image: CGImage, registration: PackRegistration, layout: PackLayout) throws -> [Detection] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFit
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    let objects = request.results?.compactMap { $0 as? VNRecognizedObjectObservation } ?? []
    return objects.compactMap { object in
      guard let label = object.labels.first, label.confidence >= Self.noiseFloor else { return nil }
      let box = object.boundingBox
      return Detection(
        label: label.identifier, confidence: Double(label.confidence),
        boundingBox: Rect2D(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height))
    }
  }
}

/// Training-free pill finder used until a trained detector ships: every compartment is cropped and
/// segmented by Apple's class-agnostic instance segmentation, cross-checked with the classical
/// segmenter. Disagreement yields low-confidence pills, which go to review.
final class PocketSegmenter: ObjectDetector, @unchecked Sendable {
  /// Fraction of each compartment trimmed on every side so blister walls stay out of the crop.
  static let inset = 0.06

  private let combiner = PocketSegmentation()
  private let classical = BlobSegmenter()

  func detect(in image: CGImage, registration: PackRegistration, layout: PackLayout) throws -> [Detection] {
    guard let packToImage = registration.imageToPack.inverse else { return [] }
    let width = Double(image.width)
    let height = Double(image.height)
    let compartments = layout.allCompartments
    var results = [[Detection]](repeating: [], count: compartments.count)
    let lock = NSLock()

    DispatchQueue.concurrentPerform(iterations: compartments.count) { index in
      let cell = layout.cellRect(compartments[index])
      let dx = cell.width * Self.inset
      let dy = cell.height * Self.inset
      let corners = [
        Point2D(x: cell.minX + dx, y: cell.minY + dy), Point2D(x: cell.maxX - dx, y: cell.minY + dy),
        Point2D(x: cell.maxX - dx, y: cell.maxY - dy), Point2D(x: cell.minX + dx, y: cell.maxY - dy),
      ].compactMap(packToImage.apply)
      guard corners.count == 4 else { return }
      let xs = corners.map { $0.x * width }
      let ys = corners.map { $0.y * height }
      let pixelRect = CGRect(
        x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!
      ).integral.intersection(CGRect(x: 0, y: 0, width: width, height: height))
      guard pixelRect.width >= 8, pixelRect.height >= 8, let crop = image.cropping(to: pixelRect) else { return }

      let learned = try? Self.instanceRegions(in: crop)
      let classicalRegions = ImageConversion.grayImage(from: crop, maxDimension: 200).map(classical.pills) ?? []
      let detections = combiner.combine(learned: learned, classical: classicalRegions).map { pocket in
        Detection(
          label: "pill", confidence: pocket.confidence,
          boundingBox: Rect2D(
            x: (pixelRect.minX + pocket.box.x * pixelRect.width) / width,
            y: (pixelRect.minY + pocket.box.y * pixelRect.height) / height,
            width: pocket.box.width * pixelRect.width / width,
            height: pocket.box.height * pixelRect.height / height))
      }
      lock.lock()
      results[index] = detections
      lock.unlock()
    }
    return results.flatMap { $0 }
  }

  /// Instance regions from Apple's foreground instance segmentation, in normalized crop coordinates.
  private static func instanceRegions(in crop: CGImage) throws -> [MaskRegion] {
    let request = VNGenerateForegroundInstanceMaskRequest()
    try VNImageRequestHandler(cgImage: crop, orientation: .up).perform([request])
    guard let observation = request.results?.first else { return [] }
    let buffer = observation.instanceMask
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return [] }
    let maskWidth = CVPixelBufferGetWidth(buffer)
    let maskHeight = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    var labels = [UInt8](repeating: 0, count: maskWidth * maskHeight)
    for y in 0..<maskHeight {
      let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
      for x in 0..<maskWidth { labels[y * maskWidth + x] = row[x] }
    }
    return MaskAnalysis.regions(labels: labels, width: maskWidth, height: maskHeight)
  }
}

/// Finds individual pills lying loose on a plain surface (teaching photos), with no pack involved.
enum LoosePillFinder {
  static func pills(in image: CGImage) -> [CGImage] {
    let request = VNGenerateForegroundInstanceMaskRequest()
    var boxes: [Rect2D] = []
    let filter = PillRegionFilter(minimumAreaFraction: 0.0005, maximumAreaFraction: 0.2, rejectBorderRegions: true)
    if (try? VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])) != nil,
      let observation = request.results?.first
    {
      let buffer = observation.instanceMask
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      if let base = CVPixelBufferGetBaseAddress(buffer) {
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        var labels = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
          let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
          for x in 0..<w { labels[y * w + x] = row[x] }
        }
        boxes = filter.pills(MaskAnalysis.regions(labels: labels, width: w, height: h)).map(\.box)
      }
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    if boxes.isEmpty, let gray = ImageConversion.grayImage(from: image, maxDimension: 400) {
      boxes = BlobSegmenter(analysisMaxDimension: 400, filter: filter).pills(in: gray).map(\.box)
    }
    return boxes.compactMap { PillCrops.crop(image, box: $0) }
  }
}
