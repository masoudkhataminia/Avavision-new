import AvaVisionCore
import AvaVisionImaging
import CoreGraphics
import Foundation
import Vision

/// How the pack lies in the camera image. The grid overlay shows the result so staff can confirm
/// that "Day 1 · Morning" sits on the correct compartment.
enum PackOrientation: String, CaseIterable, Identifiable {
  /// Rotate a landscape pack seen in a portrait image by one quarter turn; otherwise upright.
  case automatic
  case upright
  case quarterTurn
  case halfTurn
  case threeQuarterTurn

  static let storageKey = "packOrientation"

  var id: String { rawValue }

  var title: String {
    switch self {
    case .automatic: "Automatic"
    case .upright: "Upright"
    case .quarterTurn: "Rotated 90°"
    case .halfTurn: "Rotated 180°"
    case .threeQuarterTurn: "Rotated 270°"
    }
  }

  func quarterTurns(for quad: Quad, imageWidth: Int, imageHeight: Int, layout: PackLayout) -> Int {
    switch self {
    case .upright: return 0
    case .quarterTurn: return 1
    case .halfTurn: return 2
    case .threeQuarterTurn: return 3
    case .automatic:
      let pixels = quad.scaled(x: Double(imageWidth), y: Double(imageHeight))
      let width = pixels.topLeft.distance(to: pixels.topRight)
      let height = pixels.topLeft.distance(to: pixels.bottomLeft)
      let quadIsLandscape = width >= height
      let packIsLandscape = layout.aspectRatio >= 1
      return quadIsLandscape == packIsLandscape ? 0 : 1
    }
  }
}

/// One analysed image together with the pixels it came from (kept in memory only).
struct AnalyzedFrame: @unchecked Sendable {
  let observation: FrameObservation
  let image: CGImage
}

/// Turns an upright image into a `FrameObservation`: pack outline (Vision rectangle detection),
/// objects (the Core ML model, if one may run) and capture quality. All stored state is immutable and
/// Vision requests are created per call, so one analyzer can be used from any queue.
final class FrameAnalyzer: @unchecked Sendable {
  /// Detections below this confidence are model noise and are not reported. The release gate
  /// evaluates the model at this same operating point.
  static let detectionNoiseFloor: Float = 0.25

  let layout: PackLayout
  let orientation: PackOrientation
  let visionModel: VNCoreMLModel?
  private let quality = CaptureQualityAnalyzer()

  init(layout: PackLayout, orientation: PackOrientation, visionModel: VNCoreMLModel?) {
    self.layout = layout
    self.orientation = orientation
    self.visionModel = visionModel
  }

  func analyze(_ image: CGImage, capturedAt: Date = Date()) -> AnalyzedFrame {
    let rectangles = VNDetectRectanglesRequest()
    rectangles.maximumObservations = 1
    rectangles.minimumConfidence = 0.5
    rectangles.minimumSize = 0.3
    rectangles.minimumAspectRatio = 0.4
    rectangles.maximumAspectRatio = 1.0
    rectangles.quadratureTolerance = 30

    var requests: [VNRequest] = [rectangles]
    var objectRequest: VNCoreMLRequest?
    if let visionModel {
      let request = VNCoreMLRequest(model: visionModel)
      request.imageCropAndScaleOption = .scaleFit
      requests.append(request)
      objectRequest = request
    }

    var issues: [CaptureIssue] = []
    var quad: Quad?
    var rectangleConfidence = 0.0
    var detections: [Detection] = []
    do {
      try VNImageRequestHandler(cgImage: image, orientation: .up).perform(requests)
      if let rectangle = rectangles.results?.first {
        let detected = Quad(
          topLeft: Self.topLeftOrigin(rectangle.topLeft), topRight: Self.topLeftOrigin(rectangle.topRight),
          bottomRight: Self.topLeftOrigin(rectangle.bottomRight), bottomLeft: Self.topLeftOrigin(rectangle.bottomLeft))
        let turns = orientation.quarterTurns(
          for: detected, imageWidth: image.width, imageHeight: image.height, layout: layout)
        quad = detected.rotated(quarterTurns: turns)
        rectangleConfidence = Double(rectangle.confidence)
      }
      let objects = objectRequest?.results?.compactMap { $0 as? VNRecognizedObjectObservation } ?? []
      detections = objects.compactMap { object in
        guard let label = object.labels.first, label.confidence >= Self.detectionNoiseFloor else { return nil }
        let box = object.boundingBox
        return Detection(
          label: label.identifier, confidence: Double(label.confidence),
          boundingBox: Rect2D(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height))
      }
    } catch {
      issues.append(.analysisFailed)
    }

    let gray = ImageConversion.grayImage(from: image)
    var assessment = CaptureQualityAssessment(issues: [.analysisFailed], metrics: nil)
    if let gray {
      assessment = quality.assess(gray, region: quad?.boundingBox)
    }
    assessment.issues = Array(Set(assessment.issues + issues)).sorted { $0.rawValue < $1.rawValue }

    let registration = PackRegistrar.register(
      quad: quad, detectorConfidence: rectangleConfidence, imageWidth: image.width, imageHeight: image.height,
      layout: layout)
    let observation = FrameObservation(
      capturedAt: capturedAt, quality: assessment, registration: registration, detections: detections,
      imageSHA256: gray.map { ContentHasher.sha256Hex(Data($0.pixels)) })
    return AnalyzedFrame(observation: observation, image: image)
  }

  /// Vision uses a bottom-left origin; the core uses top-left.
  private static func topLeftOrigin(_ point: CGPoint) -> Point2D {
    Point2D(x: Double(point.x), y: 1 - Double(point.y))
  }
}
