import AvaVisionImaging
import CoreGraphics
import CoreImage
import Foundation

enum ImageConversion {
  private static let context = CIContext(options: [.cacheIntermediates: false])

  /// Renders `image` as a CGImage whose longer side is at most `maxDimension` pixels.
  static func render(_ image: CIImage, maxDimension: CGFloat) -> CGImage? {
    var image = image
    let longest = max(image.extent.width, image.extent.height)
    guard longest > 0 else { return nil }
    if longest > maxDimension {
      let scale = maxDimension / longest
      image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
    image = image.transformed(
      by: CGAffineTransform(translationX: -image.extent.origin.x, y: -image.extent.origin.y))
    return context.createCGImage(image, from: image.extent.integral)
  }

  /// Decodes photo data, applying its EXIF orientation so the result is upright.
  static func uprightImage(fromPhotoData data: Data, maxDimension: CGFloat = 1920) -> CGImage? {
    guard let image = CIImage(data: data, options: [.applyOrientationProperty: true]) else { return nil }
    return render(image, maxDimension: maxDimension)
  }

  /// Luminance copy of `image`, at most `maxDimension` pixels on its longer side.
  static func grayImage(from image: CGImage, maxDimension: Int = 1024) -> GrayImage? {
    let scale = min(1, Double(maxDimension) / Double(max(image.width, image.height)))
    let width = max(1, Int(Double(image.width) * scale))
    let height = max(1, Int(Double(image.height) * scale))
    var pixels = [UInt8](repeating: 0, count: width * height)
    let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
      guard
        let context = CGContext(
          data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
          space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
      else { return false }
      context.interpolationQuality = .medium
      context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
      return true
    }
    return drawn ? GrayImage(width: width, height: height, pixels: pixels) : nil
  }
}
