import AvaVisionCore

/// One connected region of a mask.
public struct MaskRegion: Hashable, Sendable {
  /// Instance label for label masks; component number for binary masks.
  public var label: Int
  public var area: Int
  /// Bounding box in normalized mask coordinates (origin top-left).
  public var box: Rect2D
  /// The region reaches the outermost row or column, so it may be a wall or a cut-off object.
  public var touchesBorder: Bool
  /// Area as a fraction of the whole mask.
  public var areaFraction: Double
}

public enum MaskAnalysis {
  /// Regions of a label mask in which 0 is background and every other value is one instance.
  public static func regions(labels: [UInt8], width: Int, height: Int) -> [MaskRegion] {
    guard width > 0, height > 0, labels.count >= width * height else { return [] }
    var bounds: [Int: (minX: Int, minY: Int, maxX: Int, maxY: Int, area: Int)] = [:]
    for y in 0..<height {
      for x in 0..<width {
        let label = Int(labels[y * width + x])
        guard label != 0 else { continue }
        if let b = bounds[label] {
          bounds[label] = (min(b.minX, x), min(b.minY, y), max(b.maxX, x), max(b.maxY, y), b.area + 1)
        } else {
          bounds[label] = (x, y, x, y, 1)
        }
      }
    }
    return bounds.keys.sorted().map { label in
      region(label: label, bounds: bounds[label]!, width: width, height: height)
    }
  }

  /// 4-connected components of a binary mask.
  public static func components(mask: [Bool], width: Int, height: Int) -> [MaskRegion] {
    guard width > 0, height > 0, mask.count >= width * height else { return [] }
    var visited = [Bool](repeating: false, count: width * height)
    var result: [MaskRegion] = []
    var stack: [Int] = []
    for start in 0..<(width * height) where mask[start] && !visited[start] {
      visited[start] = true
      stack.append(start)
      var b = (minX: width, minY: height, maxX: 0, maxY: 0, area: 0)
      while let index = stack.popLast() {
        let x = index % width
        let y = index / width
        b = (min(b.minX, x), min(b.minY, y), max(b.maxX, x), max(b.maxY, y), b.area + 1)
        for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        where nx >= 0 && ny >= 0 && nx < width && ny < height {
          let neighbour = ny * width + nx
          if mask[neighbour] && !visited[neighbour] {
            visited[neighbour] = true
            stack.append(neighbour)
          }
        }
      }
      result.append(region(label: result.count + 1, bounds: b, width: width, height: height))
    }
    return result
  }

  private static func region(
    label: Int, bounds b: (minX: Int, minY: Int, maxX: Int, maxY: Int, area: Int), width: Int, height: Int
  ) -> MaskRegion {
    MaskRegion(
      label: label, area: b.area,
      box: Rect2D(
        x: Double(b.minX) / Double(width), y: Double(b.minY) / Double(height),
        width: Double(b.maxX - b.minX + 1) / Double(width), height: Double(b.maxY - b.minY + 1) / Double(height)),
      touchesBorder: b.minX == 0 || b.minY == 0 || b.maxX == width - 1 || b.maxY == height - 1,
      areaFraction: Double(b.area) / Double(width * height))
  }
}

/// Which regions inside one compartment crop are plausibly whole doses.
public struct PillRegionFilter: Codable, Hashable, Sendable {
  public var minimumAreaFraction: Double
  public var maximumAreaFraction: Double
  /// Regions touching the crop edge are usually blister walls or neighbouring pockets.
  public var rejectBorderRegions: Bool

  public init(minimumAreaFraction: Double, maximumAreaFraction: Double, rejectBorderRegions: Bool) {
    self.minimumAreaFraction = minimumAreaFraction
    self.maximumAreaFraction = maximumAreaFraction
    self.rejectBorderRegions = rejectBorderRegions
  }

  public static let standard = PillRegionFilter(
    minimumAreaFraction: 0.004, maximumAreaFraction: 0.45, rejectBorderRegions: true)

  public func pills(_ regions: [MaskRegion]) -> [MaskRegion] {
    regions.filter {
      $0.areaFraction >= minimumAreaFraction && $0.areaFraction <= maximumAreaFraction
        && !(rejectBorderRegions && $0.touchesBorder)
    }
  }
}

/// Classical, training-free pill finder for one compartment crop: pixels that differ clearly from
/// the compartment's background (estimated from its outer ring) are grouped into blobs.
///
/// It is an independent second opinion for the learned segmenter; a white pill on a white card can
/// be invisible to it, which is why disagreement sends the compartment to review instead of guessing.
public struct BlobSegmenter: Sendable {
  public var analysisMaxDimension: Int
  /// Minimum difference from the background, in grey levels.
  public var minimumContrast: Int
  /// Difference threshold in multiples of the background's median absolute deviation.
  public var deviationMultiplier: Double
  public var filter: PillRegionFilter

  public init(
    analysisMaxDimension: Int = 160, minimumContrast: Int = 24, deviationMultiplier: Double = 5,
    filter: PillRegionFilter = .standard
  ) {
    self.analysisMaxDimension = analysisMaxDimension
    self.minimumContrast = minimumContrast
    self.deviationMultiplier = deviationMultiplier
    self.filter = filter
  }

  public func pills(in image: GrayImage) -> [MaskRegion] {
    let sample = image.downsampled(maxDimension: analysisMaxDimension)
    let w = sample.width
    let h = sample.height
    guard w >= 8, h >= 8 else { return [] }
    let ringWidth = max(1, min(w, h) / 12)
    var ring: [Int] = []
    for y in 0..<h {
      for x in 0..<w where x < ringWidth || y < ringWidth || x >= w - ringWidth || y >= h - ringWidth {
        ring.append(Int(sample[x, y]))
      }
    }
    let background = Self.median(ring)
    let deviation = Self.median(ring.map { abs($0 - background) })
    let threshold = max(minimumContrast, Int((Double(deviation) * deviationMultiplier).rounded(.up)))
    let mask = sample.pixels.map { abs(Int($0) - background) > threshold }
    return filter.pills(MaskAnalysis.components(mask: mask, width: w, height: h))
  }

  private static func median(_ values: [Int]) -> Int {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
  }
}

/// A pill candidate found in one compartment crop.
public struct PocketDetection: Hashable, Sendable {
  /// Normalized box within the crop (origin top-left).
  public var box: Rect2D
  public var confidence: Double
}

/// Combines a learned segmenter with the classical one. Matching counts give confident pills;
/// any disagreement gives low-confidence pills, which the decision engine sends to review.
public struct PocketSegmentation: Sendable {
  public var filter: PillRegionFilter
  public var agreedConfidence: Double
  public var disagreedConfidence: Double

  public init(filter: PillRegionFilter = .standard, agreedConfidence: Double = 0.9, disagreedConfidence: Double = 0.5) {
    self.filter = filter
    self.agreedConfidence = agreedConfidence
    self.disagreedConfidence = disagreedConfidence
  }

  /// - Parameter learned: Regions from the learned segmenter, or `nil` if it could not run.
  public func combine(learned: [MaskRegion]?, classical: [MaskRegion]) -> [PocketDetection] {
    let classicalPills = filter.pills(classical)
    guard let learned else {
      return classicalPills.map { PocketDetection(box: $0.box, confidence: disagreedConfidence) }
    }
    let learnedPills = filter.pills(learned)
    if learnedPills.count == classicalPills.count {
      return learnedPills.map { PocketDetection(box: $0.box, confidence: agreedConfidence) }
    }
    let larger = learnedPills.count >= classicalPills.count ? learnedPills : classicalPills
    return larger.map { PocketDetection(box: $0.box, confidence: disagreedConfidence) }
  }
}
