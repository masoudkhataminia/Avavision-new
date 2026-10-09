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
  var observation: FrameObservation
  let image: CGImage
}

/// Turns an upright image into a `FrameObservation` in two steps:
/// `locate` (pack outline and capture quality — fast enough for the live camera) and
/// `detectObjects` (pills in every compartment — run on the few frames that are evaluated).
/// All stored state is immutable and Vision requests are created per call, so one analyzer can be
/// used from any queue.
final class FrameAnalyzer: @unchecked Sendable {
  let layout: PackLayout
  let orientation: PackOrientation
  let detector: ObjectDetector?
  private let quality = CaptureQualityAnalyzer()

  init(layout: PackLayout, orientation: PackOrientation, detector: ObjectDetector?) {
    self.layout = layout
    self.orientation = orientation
    self.detector = detector
  }

  func locate(_ image: CGImage, capturedAt: Date = Date()) -> AnalyzedFrame {
    let rectangles = VNDetectRectanglesRequest()
    rectangles.maximumObservations = 1
    rectangles.minimumConfidence = 0.5
    rectangles.minimumSize = 0.3
    rectangles.minimumAspectRatio = 0.4
    rectangles.maximumAspectRatio = 1.0
    rectangles.quadratureTolerance = 30

    var issues: [CaptureIssue] = []
    var quad: Quad?
    var rectangleConfidence = 0.0
    do {
      try VNImageRequestHandler(cgImage: image, orientation: .up).perform([rectangles])
      if let rectangle = rectangles.results?.first {
        let detected = Quad(
          topLeft: Self.topLeftOrigin(rectangle.topLeft), topRight: Self.topLeftOrigin(rectangle.topRight),
          bottomRight: Self.topLeftOrigin(rectangle.bottomRight), bottomLeft: Self.topLeftOrigin(rectangle.bottomLeft))
        let turns = orientation.quarterTurns(
          for: detected, imageWidth: image.width, imageHeight: image.height, layout: layout)
        quad = detected.rotated(quarterTurns: turns)
        rectangleConfidence = Double(rectangle.confidence)
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
      capturedAt: capturedAt, quality: assessment, registration: registration, detections: [],
      imageSHA256: gray.map { ContentHasher.sha256Hex(Data($0.pixels)) })
    return AnalyzedFrame(observation: observation, image: image)
  }

  /// Fills in the objects of a usable frame. If detection fails, the frame is marked unusable.
  func detectObjects(in frame: AnalyzedFrame) -> AnalyzedFrame {
    guard frame.observation.isUsable, let registration = frame.observation.registration.registration else {
      return frame
    }
    var result = frame
    guard let detector else { return result }
    do {
      result.observation.detections = try detector.detect(
        in: frame.image, registration: registration, layout: layout)
    } catch {
      result.observation.quality.issues.append(.analysisFailed)
    }
    return result
  }

  /// Vision uses a bottom-left origin; the core uses top-left.
  private static func topLeftOrigin(_ point: CGPoint) -> Point2D {
    Point2D(x: Double(point.x), y: 1 - Double(point.y))
  }
}
