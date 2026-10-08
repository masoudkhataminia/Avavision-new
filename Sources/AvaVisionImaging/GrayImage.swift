import AvaVisionCore

/// An 8-bit single-channel image (luminance), row-major, origin top-left.
public struct GrayImage: Sendable, Equatable {
  public let width: Int
  public let height: Int
  public let pixels: [UInt8]

  public init?(width: Int, height: Int, pixels: [UInt8]) {
    guard width > 0, height > 0, pixels.count == width * height else { return nil }
    self.width = width
    self.height = height
    self.pixels = pixels
  }

  public init(width: Int, height: Int, fill: UInt8) {
    self.width = max(1, width)
    self.height = max(1, height)
    self.pixels = Array(repeating: fill, count: self.width * self.height)
  }

  public enum ChannelOrder: Sendable {
    case rgba
    case bgra
  }

  /// Converts interleaved 8-bit 4-channel pixels using Rec. 601 luma weights.
  public init?(fourChannel bytes: [UInt8], width: Int, height: Int, bytesPerRow: Int, order: ChannelOrder) {
    guard width > 0, height > 0, bytesPerRow >= width * 4, bytes.count >= bytesPerRow * (height - 1) + width * 4
    else { return nil }
    let (r, b) = order == .rgba ? (0, 2) : (2, 0)
    var output = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
      let row = y * bytesPerRow
      for x in 0..<width {
        let p = row + x * 4
        let luma = 299 * Int(bytes[p + r]) + 587 * Int(bytes[p + 1]) + 114 * Int(bytes[p + b])
        output[y * width + x] = UInt8((luma + 500) / 1000)
      }
    }
    self.init(width: width, height: height, pixels: output)
  }

  public subscript(x: Int, y: Int) -> UInt8 {
    pixels[y * width + x]
  }

  /// Crops to a rectangle in normalized coordinates, clamped to the image.
  public func cropped(to rect: Rect2D) -> GrayImage? {
    let x0 = max(0, min(width, Int((rect.minX * Double(width)).rounded(.down))))
    let y0 = max(0, min(height, Int((rect.minY * Double(height)).rounded(.down))))
    let x1 = max(0, min(width, Int((rect.maxX * Double(width)).rounded(.up))))
    let y1 = max(0, min(height, Int((rect.maxY * Double(height)).rounded(.up))))
    guard x1 > x0, y1 > y0 else { return nil }
    var output = [UInt8]()
    output.reserveCapacity((x1 - x0) * (y1 - y0))
    for y in y0..<y1 {
      output.append(contentsOf: pixels[(y * width + x0)..<(y * width + x1)])
    }
    return GrayImage(width: x1 - x0, height: y1 - y0, pixels: output)
  }

  /// Box-averages so the longer side is at most `maxDimension`, making metrics resolution-independent.
  public func downsampled(maxDimension: Int) -> GrayImage {
    let factor = Int((Double(max(width, height)) / Double(max(1, maxDimension))).rounded(.up))
    guard factor > 1 else { return self }
    let outWidth = max(1, width / factor)
    let outHeight = max(1, height / factor)
    var output = [UInt8](repeating: 0, count: outWidth * outHeight)
    let area = factor * factor
    for oy in 0..<outHeight {
      for ox in 0..<outWidth {
        var sum = 0
        for y in (oy * factor)..<(oy * factor + factor) {
          let row = y * width
          for x in (ox * factor)..<(ox * factor + factor) {
            sum += Int(pixels[row + x])
          }
        }
        output[oy * outWidth + ox] = UInt8(sum / area)
      }
    }
    return GrayImage(width: outWidth, height: outHeight, pixels: output) ?? self
  }
}
