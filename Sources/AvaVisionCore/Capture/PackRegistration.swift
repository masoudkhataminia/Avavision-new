import Foundation

/// A reason the pack could not be located reliably in a frame.
public enum RegistrationIssue: String, Codable, Hashable, Sendable, CaseIterable {
  case packNotFound
  case lowDetectorConfidence
  case packNotConvex
  case packTooSmall
  case packTouchesFrameEdge
  case perspectiveTooSteep
  case aspectRatioMismatch
}

/// Thresholds for accepting a detected pack outline.
public struct RegistrationPolicy: Codable, Hashable, Sendable {
  /// Minimum fraction of the image covered by the pack.
  public var minimumAreaFraction: Double
  public var minimumDetectorConfidence: Double
  /// Maximum deviation of any corner angle from 90°, in degrees.
  public var maximumCornerAngleDeviation: Double
  /// Allowed relative difference between observed and physical aspect ratio (calibrated layouts only).
  public var aspectRatioTolerance: Double
  /// Minimum normalized distance between the pack outline and the image border.
  public var minimumEdgeMargin: Double

  public init(
    minimumAreaFraction: Double, minimumDetectorConfidence: Double,
    maximumCornerAngleDeviation: Double, aspectRatioTolerance: Double, minimumEdgeMargin: Double
  ) {
    self.minimumAreaFraction = minimumAreaFraction
    self.minimumDetectorConfidence = minimumDetectorConfidence
    self.maximumCornerAngleDeviation = maximumCornerAngleDeviation
    self.aspectRatioTolerance = aspectRatioTolerance
    self.minimumEdgeMargin = minimumEdgeMargin
  }

  public static let standard = RegistrationPolicy(
    minimumAreaFraction: 0.25,
    minimumDetectorConfidence: 0.7,
    maximumCornerAngleDeviation: 20,
    aspectRatioTolerance: 0.12,
    minimumEdgeMargin: 0.005
  )
}

/// The pack outline found in a frame and the transform from image to pack coordinates.
public struct PackRegistration: Codable, Hashable, Sendable {
  /// Outline in normalized image coordinates (origin top-left).
  public let quad: Quad
  public let imageToPack: Homography
  public let detectorConfidence: Double
}

public enum RegistrationOutcome: Codable, Hashable, Sendable {
  case registered(PackRegistration)
  case rejected([RegistrationIssue])

  public var registration: PackRegistration? {
    if case .registered(let registration) = self { return registration }
    return nil
  }

  public var issues: [RegistrationIssue] {
    if case .rejected(let issues) = self { return issues }
    return []
  }
}

public enum PackRegistrar {
  /// Validates a detected outline and builds the image→pack transform.
  ///
  /// - Parameters:
  ///   - quad: Detected outline in normalized image coordinates, or `nil` if nothing was found.
  ///   - imageWidth/imageHeight: Pixel size of the analysed image, used for true angles and aspect.
  public static func register(
    quad: Quad?, detectorConfidence: Double, imageWidth: Int, imageHeight: Int,
    layout: PackLayout, policy: RegistrationPolicy = .standard
  ) -> RegistrationOutcome {
    guard let quad, imageWidth > 0, imageHeight > 0 else { return .rejected([.packNotFound]) }
    guard quad.isConvex else { return .rejected([.packNotConvex]) }

    var issues: [RegistrationIssue] = []
    if detectorConfidence < policy.minimumDetectorConfidence { issues.append(.lowDetectorConfidence) }
    if quad.area < policy.minimumAreaFraction { issues.append(.packTooSmall) }

    let margin = policy.minimumEdgeMargin
    if quad.corners.contains(where: { $0.x < margin || $0.y < margin || $0.x > 1 - margin || $0.y > 1 - margin }) {
      issues.append(.packTouchesFrameEdge)
    }

    let pixels = quad.scaled(x: Double(imageWidth), y: Double(imageHeight))
    if pixels.interiorAngles.contains(where: { abs($0 - 90) > policy.maximumCornerAngleDeviation }) {
      issues.append(.perspectiveTooSteep)
    }

    if layout.isCalibrated {
      let width =
        (pixels.topLeft.distance(to: pixels.topRight) + pixels.bottomLeft.distance(to: pixels.bottomRight)) / 2
      let height =
        (pixels.topLeft.distance(to: pixels.bottomLeft) + pixels.topRight.distance(to: pixels.bottomRight)) / 2
      if height <= 0 || abs((width / height) / layout.aspectRatio - 1) > policy.aspectRatioTolerance {
        issues.append(.aspectRatioMismatch)
      }
    }

    let packCorners = [Point2D(x: 0, y: 0), Point2D(x: 1, y: 0), Point2D(x: 1, y: 1), Point2D(x: 0, y: 1)]
    guard issues.isEmpty else { return .rejected(issues) }
    guard let transform = Homography(mapping: quad.corners, to: packCorners) else {
      return .rejected([.packNotConvex])
    }
    return .registered(
      PackRegistration(quad: quad, imageToPack: transform, detectorConfidence: detectorConfidence))
  }
}
