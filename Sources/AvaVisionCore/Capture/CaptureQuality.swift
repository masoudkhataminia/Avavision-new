import Foundation

/// A reason an image cannot be used as evidence.
public enum CaptureIssue: String, Codable, Hashable, Sendable, CaseIterable {
  case blurry
  case tooDark
  case tooBright
  case glare
  /// The image could not be analysed at all (for example the model failed to run).
  case analysisFailed
}

/// Raw measurements behind a capture-quality decision, kept for audit and calibration.
public struct CaptureQualityMetrics: Codable, Hashable, Sendable {
  /// Variance of the Laplacian on the analysed region; higher is sharper.
  public var sharpness: Double
  /// Mean luminance, 0…255.
  public var meanLuminance: Double
  /// Fraction of pixels at or below the shadow clip level.
  public var shadowClipFraction: Double
  /// Fraction of pixels at or above the specular glare level.
  public var glareFraction: Double

  public init(sharpness: Double, meanLuminance: Double, shadowClipFraction: Double, glareFraction: Double) {
    self.sharpness = sharpness
    self.meanLuminance = meanLuminance
    self.shadowClipFraction = shadowClipFraction
    self.glareFraction = glareFraction
  }
}

/// Outcome of the capture-quality check for one frame.
public struct CaptureQualityAssessment: Codable, Hashable, Sendable {
  public var issues: [CaptureIssue]
  public var metrics: CaptureQualityMetrics?

  public init(issues: [CaptureIssue], metrics: CaptureQualityMetrics?) {
    self.issues = issues
    self.metrics = metrics
  }

  public var isAcceptable: Bool { issues.isEmpty }

  /// Used by tests and tools that feed already-validated images.
  public static let accepted = CaptureQualityAssessment(issues: [], metrics: nil)
}
