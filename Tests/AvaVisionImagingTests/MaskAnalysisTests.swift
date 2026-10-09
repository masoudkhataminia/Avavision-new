import AvaVisionCore
import XCTest

@testable import AvaVisionImaging

final class MaskAnalysisTests: XCTestCase {
  func testLabelMaskRegions() {
    // 6×4 mask: instance 1 top-left 2×2, instance 2 touching the right edge.
    let labels: [UInt8] = [
      1, 1, 0, 0, 0, 0,
      1, 1, 0, 0, 2, 2,
      0, 0, 0, 0, 2, 2,
      0, 0, 0, 0, 0, 0,
    ]
    let regions = MaskAnalysis.regions(labels: labels, width: 6, height: 4)
    XCTAssertEqual(regions.map(\.label), [1, 2])
    XCTAssertEqual(regions[0].area, 4)
    XCTAssertEqual(regions[0].box, Rect2D(x: 0, y: 0, width: 2.0 / 6.0, height: 0.5))
    XCTAssertTrue(regions[0].touchesBorder)
    XCTAssertEqual(regions[1].box, Rect2D(x: 4.0 / 6.0, y: 0.25, width: 2.0 / 6.0, height: 0.5))
    XCTAssertEqual(regions[1].areaFraction, 4.0 / 24.0, accuracy: 1e-12)
    XCTAssertEqual(MaskAnalysis.regions(labels: [1], width: 2, height: 2), [])
  }

  func testBinaryComponentsAreFourConnected() {
    let mask: [Bool] = [
      false, false, false, false, false,
      false, true, false, false, false,
      false, false, true, true, false,
      false, false, true, false, false,
      false, false, false, false, false,
    ]
    let components = MaskAnalysis.components(mask: mask, width: 5, height: 5)
    XCTAssertEqual(components.map(\.area).sorted(), [1, 3])
    XCTAssertTrue(components.allSatisfy { !$0.touchesBorder })
  }

  func testFilterDropsSpecksWallsAndHugeRegions() {
    func region(_ fraction: Double, border: Bool = false) -> MaskRegion {
      MaskRegion(label: 1, area: 1, box: .unit, touchesBorder: border, areaFraction: fraction)
    }
    let kept = PillRegionFilter.standard.pills([
      region(0.001), region(0.05), region(0.05, border: true), region(0.8),
    ])
    XCTAssertEqual(kept.map(\.areaFraction), [0.05])
  }

  /// Compartment crop: light card, a darker blister ring at the edge, and `pills` round dark pills.
  private func compartment(pills: [(Int, Int)], pillValue: UInt8 = 90, size: Int = 120) -> GrayImage {
    var pixels = [UInt8](repeating: 225, count: size * size)
    for y in 0..<size {
      for x in 0..<size {
        if x < 3 || y < 3 || x >= size - 3 || y >= size - 3 { pixels[y * size + x] = 200 }
        for (cx, cy) in pills where (x - cx) * (x - cx) + (y - cy) * (y - cy) <= 100 {
          pixels[y * size + x] = pillValue
        }
      }
    }
    return GrayImage(width: size, height: size, pixels: pixels)!
  }

  func testBlobSegmenterCountsSeparatedPills() {
    let segmenter = BlobSegmenter()
    XCTAssertEqual(segmenter.pills(in: compartment(pills: [])).count, 0)
    XCTAssertEqual(segmenter.pills(in: compartment(pills: [(40, 60)])).count, 1)
    let three = segmenter.pills(in: compartment(pills: [(30, 30), (80, 40), (60, 90)]))
    XCTAssertEqual(three.count, 3)
    for region in three {
      XCTAssertEqual(region.box.width, 21.0 / 120.0, accuracy: 0.03)
    }
  }

  func testLowContrastPillIsInvisibleToTheClassicalSegmenter() {
    XCTAssertEqual(BlobSegmenter().pills(in: compartment(pills: [(60, 60)], pillValue: 235)).count, 0)
  }
}

final class PocketSegmentationTests: XCTestCase {
  private func region(_ x: Double) -> MaskRegion {
    MaskRegion(
      label: 1, area: 50, box: Rect2D(x: x, y: 0.4, width: 0.2, height: 0.2), touchesBorder: false, areaFraction: 0.05)
  }

  func testAgreementGivesConfidentPillsAndDisagreementGoesToReview() {
    let segmentation = PocketSegmentation()
    let agreed = segmentation.combine(learned: [region(0.1), region(0.5)], classical: [region(0.12), region(0.52)])
    XCTAssertEqual(agreed.map(\.confidence), [0.9, 0.9])
    XCTAssertEqual(agreed.first?.box.x, 0.1)

    let disagreed = segmentation.combine(learned: [region(0.1)], classical: [region(0.1), region(0.5)])
    XCTAssertEqual(disagreed.count, 2)
    XCTAssertTrue(disagreed.allSatisfy { $0.confidence < DecisionPolicy.standard.minimumDetectionConfidence })

    let fallback = segmentation.combine(learned: nil, classical: [region(0.3)])
    XCTAssertEqual(fallback.map(\.confidence), [0.5])
    XCTAssertGreaterThanOrEqual(segmentation.agreedConfidence, DecisionPolicy.standard.minimumDetectionConfidence)
  }
}
