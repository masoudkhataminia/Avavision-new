import AvaVisionCore

/// Thresholds for accepting an image as evidence. Defaults are conservative starting points and
/// must be calibrated on the real station (lighting, camera distance) before a pilot.
public struct CaptureQualityPolicy: Codable, Hashable, Sendable {
  /// Images are downsampled to this size before measuring, so thresholds do not depend on resolution.
  public var analysisMaxDimension: Int
  public var minimumSharpness: Double
  public var minimumMeanLuminance: Double
  public var maximumMeanLuminance: Double
  public var shadowClipLevel: UInt8
  public var maximumShadowClipFraction: Double
  public var glareLevel: UInt8
  public var maximumGlareFraction: Double

  public init(
    analysisMaxDimension: Int, minimumSharpness: Double, minimumMeanLuminance: Double,
    maximumMeanLuminance: Double, shadowClipLevel: UInt8, maximumShadowClipFraction: Double,
    glareLevel: UInt8, maximumGlareFraction: Double
  ) {
    self.analysisMaxDimension = analysisMaxDimension
    self.minimumSharpness = minimumSharpness
    self.minimumMeanLuminance = minimumMeanLuminance
    self.maximumMeanLuminance = maximumMeanLuminance
    self.shadowClipLevel = shadowClipLevel
    self.maximumShadowClipFraction = maximumShadowClipFraction
    self.glareLevel = glareLevel
    self.maximumGlareFraction = maximumGlareFraction
  }

  public static let standard = CaptureQualityPolicy(
    analysisMaxDimension: 640,
    minimumSharpness: 60,
    minimumMeanLuminance: 50,
    maximumMeanLuminance: 215,
    shadowClipLevel: 12,
    maximumShadowClipFraction: 0.35,
    glareLevel: 250,
    maximumGlareFraction: 0.03
  )
}

/// Measures sharpness, exposure and glare of a frame (or the pack region of it).
public struct CaptureQualityAnalyzer: Sendable {
  public var policy: CaptureQualityPolicy

  public init(policy: CaptureQualityPolicy = .standard) {
    self.policy = policy
  }

  /// - Parameter region: Normalized area to analyse, usually the bounding box of the detected pack.
  public func assess(_ image: GrayImage, region: Rect2D? = nil) -> CaptureQualityAssessment {
    let source = region.flatMap { image.cropped(to: $0) } ?? image
    let sample = source.downsampled(maxDimension: policy.analysisMaxDimension)
    guard let metrics = Self.measure(sample, policy: policy) else {
      return CaptureQualityAssessment(issues: [.blurry], metrics: nil)
    }
    var issues: [CaptureIssue] = []
    if metrics.sharpness < policy.minimumSharpness { issues.append(.blurry) }
    if metrics.meanLuminance < policy.minimumMeanLuminance
      || metrics.shadowClipFraction > policy.maximumShadowClipFraction
    {
      issues.append(.tooDark)
    }
    if metrics.meanLuminance > policy.maximumMeanLuminance { issues.append(.tooBright) }
    if metrics.glareFraction > policy.maximumGlareFraction { issues.append(.glare) }
    return CaptureQualityAssessment(issues: issues, metrics: metrics)
  }

  /// Returns `nil` for images too small to measure (less than 3×3).
  static func measure(_ image: GrayImage, policy: CaptureQualityPolicy) -> CaptureQualityMetrics? {
    guard image.width >= 3, image.height >= 3 else { return nil }
    let count = Double(image.pixels.count)
    var sum = 0
    var shadows = 0
    var glare = 0
    for value in image.pixels {
      sum += Int(value)
      if value <= policy.shadowClipLevel { shadows += 1 }
      if value >= policy.glareLevel { glare += 1 }
    }

    // Variance of the 4-neighbour Laplacian over interior pixels.
    var laplacianSum = 0.0
    var laplacianSquares = 0.0
    let w = image.width
    let p = image.pixels
    for y in 1..<(image.height - 1) {
      for x in 1..<(w - 1) {
        let i = y * w + x
        let value = Double(Int(p[i - 1]) + Int(p[i + 1]) + Int(p[i - w]) + Int(p[i + w]) - 4 * Int(p[i]))
        laplacianSum += value
        laplacianSquares += value * value
      }
    }
    let interior = Double((image.width - 2) * (image.height - 2))
    let laplacianMean = laplacianSum / interior

    return CaptureQualityMetrics(
      sharpness: laplacianSquares / interior - laplacianMean * laplacianMean,
      meanLuminance: Double(sum) / count,
      shadowClipFraction: Double(shadows) / count,
      glareFraction: Double(glare) / count
    )
  }
}
