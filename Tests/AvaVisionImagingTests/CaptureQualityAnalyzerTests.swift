import AvaVisionCore
import XCTest

@testable import AvaVisionImaging

final class CaptureQualityAnalyzerTests: XCTestCase {
  private let analyzer = CaptureQualityAnalyzer()

  /// High-contrast checkerboard: sharp and well exposed.
  private func checkerboard(width: Int = 800, height: Int = 600, cell: Int = 16, dark: UInt8 = 50, light: UInt8 = 190)
    -> GrayImage
  {
    var pixels = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
      for x in 0..<width {
        pixels[y * width + x] = ((x / cell) + (y / cell)) % 2 == 0 ? dark : light
      }
    }
    return GrayImage(width: width, height: height, pixels: pixels)!
  }

  /// Repeated separable box blur.
  private func blurred(_ image: GrayImage, radius: Int, passes: Int) -> GrayImage {
    var pixels = image.pixels.map(Int.init)
    let w = image.width
    let h = image.height
    for _ in 0..<passes {
      var horizontal = pixels
      for y in 0..<h {
        for x in 0..<w {
          var sum = 0
          var n = 0
          for dx in -radius...radius where (0..<w).contains(x + dx) {
            sum += pixels[y * w + x + dx]
            n += 1
          }
          horizontal[y * w + x] = sum / n
        }
      }
      for y in 0..<h {
        for x in 0..<w {
          var sum = 0
          var n = 0
          for dy in -radius...radius where (0..<h).contains(y + dy) {
            sum += horizontal[(y + dy) * w + x]
            n += 1
          }
          pixels[y * w + x] = sum / n
        }
      }
    }
    return GrayImage(width: w, height: h, pixels: pixels.map { UInt8($0) })!
  }

  func testSharpWellExposedImageIsAccepted() {
    let assessment = analyzer.assess(checkerboard())
    XCTAssertEqual(assessment.issues, [])
    XCTAssertTrue(assessment.isAcceptable)
  }

  func testBlurredImageIsRejected() throws {
    let sharp = try XCTUnwrap(analyzer.assess(checkerboard()).metrics)
    let soft = analyzer.assess(blurred(checkerboard(), radius: 6, passes: 3))
    XCTAssertEqual(soft.issues, [.blurry])
    XCTAssertLessThan(try XCTUnwrap(soft.metrics).sharpness, sharp.sharpness / 10)
  }

  func testExposureProblems() {
    XCTAssertTrue(analyzer.assess(checkerboard(dark: 0, light: 30)).issues.contains(.tooDark))
    XCTAssertTrue(analyzer.assess(checkerboard(dark: 220, light: 245)).issues.contains(.tooBright))
  }

  func testSpecularGlareIsRejected() {
    var pixels = checkerboard().pixels
    for y in 200..<300 {
      for x in 300..<500 { pixels[y * 800 + x] = 255 }
    }
    let assessment = analyzer.assess(GrayImage(width: 800, height: 600, pixels: pixels)!)
    XCTAssertTrue(assessment.issues.contains(.glare))
  }

  func testRegionRestrictsAnalysis() {
    var pixels = checkerboard().pixels
    for y in 0..<600 {
      for x in 0..<400 { pixels[y * 800 + x] = 255 }
    }
    let image = GrayImage(width: 800, height: 600, pixels: pixels)!
    XCTAssertTrue(analyzer.assess(image).issues.contains(.glare))
    XCTAssertEqual(analyzer.assess(image, region: Rect2D(x: 0.55, y: 0, width: 0.45, height: 1)).issues, [])
  }

  func testTinyImageCannotBeAssessed() {
    let assessment = analyzer.assess(GrayImage(width: 2, height: 2, fill: 128))
    XCTAssertEqual(assessment.issues, [.blurry])
    XCTAssertNil(assessment.metrics)
  }

  func testFourChannelConversionAndCropping() throws {
    let rgba: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]
    let image = try XCTUnwrap(GrayImage(fourChannel: rgba, width: 2, height: 2, bytesPerRow: 8, order: .rgba))
    XCTAssertEqual(image.pixels, [76, 150, 29, 255])
    let bgra = try XCTUnwrap(GrayImage(fourChannel: rgba, width: 2, height: 2, bytesPerRow: 8, order: .bgra))
    XCTAssertEqual(bgra.pixels, [29, 150, 76, 255])
    XCTAssertEqual(image.cropped(to: Rect2D(x: 0.5, y: 0.5, width: 0.5, height: 0.5))?.pixels, [255])
    XCTAssertNil(image.cropped(to: Rect2D(x: 2, y: 2, width: 1, height: 1)))
    XCTAssertNil(GrayImage(fourChannel: [1, 2, 3], width: 1, height: 1, bytesPerRow: 4, order: .rgba))
  }

  func testDownsamplingAverages() {
    let image = GrayImage(width: 4, height: 2, pixels: [0, 100, 200, 200, 100, 200, 200, 200])!
    XCTAssertEqual(image.downsampled(maxDimension: 2).pixels, [100, 200])
    XCTAssertEqual(image.downsampled(maxDimension: 10), image)
  }
}
