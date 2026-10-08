import Foundation

/// A point in a normalized coordinate space (origin top-left, x to the right, y downwards).
public struct Point2D: Codable, Hashable, Sendable {
  public var x: Double
  public var y: Double

  public init(x: Double, y: Double) {
    self.x = x
    self.y = y
  }

  public static func - (lhs: Point2D, rhs: Point2D) -> Point2D {
    Point2D(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
  }

  public func distance(to other: Point2D) -> Double {
    let d = self - other
    return (d.x * d.x + d.y * d.y).squareRoot()
  }

  /// Scales x and y independently, e.g. to convert normalized coordinates to pixels.
  public func scaled(x sx: Double, y sy: Double) -> Point2D {
    Point2D(x: x * sx, y: y * sy)
  }
}

/// An axis-aligned rectangle (origin top-left).
public struct Rect2D: Codable, Hashable, Sendable {
  public var x: Double
  public var y: Double
  public var width: Double
  public var height: Double

  public init(x: Double, y: Double, width: Double, height: Double) {
    self.x = x
    self.y = y
    self.width = width
    self.height = height
  }

  public static let unit = Rect2D(x: 0, y: 0, width: 1, height: 1)

  public var minX: Double { x }
  public var minY: Double { y }
  public var maxX: Double { x + width }
  public var maxY: Double { y + height }
  public var center: Point2D { Point2D(x: x + width / 2, y: y + height / 2) }

  public func contains(_ point: Point2D) -> Bool {
    point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY
  }

  /// Returns the intersection with `other`, or `nil` if they do not overlap.
  public func intersection(_ other: Rect2D) -> Rect2D? {
    let x0 = max(minX, other.minX)
    let y0 = max(minY, other.minY)
    let x1 = min(maxX, other.maxX)
    let y1 = min(maxY, other.maxY)
    guard x1 > x0, y1 > y0 else { return nil }
    return Rect2D(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
  }
}

/// A quadrilateral with named corners, as returned by a rectangle detector.
public struct Quad: Codable, Hashable, Sendable {
  public var topLeft: Point2D
  public var topRight: Point2D
  public var bottomRight: Point2D
  public var bottomLeft: Point2D

  public init(topLeft: Point2D, topRight: Point2D, bottomRight: Point2D, bottomLeft: Point2D) {
    self.topLeft = topLeft
    self.topRight = topRight
    self.bottomRight = bottomRight
    self.bottomLeft = bottomLeft
  }

  /// Corners in clockwise order starting at the top-left (in a y-down space).
  public var corners: [Point2D] { [topLeft, topRight, bottomRight, bottomLeft] }

  public func scaled(x sx: Double, y sy: Double) -> Quad {
    Quad(
      topLeft: topLeft.scaled(x: sx, y: sy),
      topRight: topRight.scaled(x: sx, y: sy),
      bottomRight: bottomRight.scaled(x: sx, y: sy),
      bottomLeft: bottomLeft.scaled(x: sx, y: sy)
    )
  }

  /// Unsigned area (shoelace formula).
  public var area: Double {
    let c = corners
    var sum = 0.0
    for i in 0..<4 {
      let a = c[i]
      let b = c[(i + 1) % 4]
      sum += a.x * b.y - b.x * a.y
    }
    return abs(sum) / 2
  }

  /// True when all corners turn the same way, i.e. the quad is convex and not self-intersecting.
  public var isConvex: Bool {
    let c = corners
    var sign = 0.0
    for i in 0..<4 {
      let a = c[i]
      let b = c[(i + 1) % 4]
      let d = c[(i + 2) % 4]
      let cross = (b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x)
      if abs(cross) < 1e-12 { return false }
      if sign == 0 {
        sign = cross
      } else if (cross > 0) != (sign > 0) {
        return false
      }
    }
    return true
  }

  /// Interior angle at each corner in degrees, in `corners` order.
  public var interiorAngles: [Double] {
    let c = corners
    return (0..<4).map { i in
      let previous = c[(i + 3) % 4]
      let current = c[i]
      let next = c[(i + 1) % 4]
      let u = previous - current
      let v = next - current
      let lengths = (u.x * u.x + u.y * u.y).squareRoot() * (v.x * v.x + v.y * v.y).squareRoot()
      guard lengths > 0 else { return 0 }
      let cosine = max(-1, min(1, (u.x * v.x + u.y * v.y) / lengths))
      return acos(cosine) * 180 / .pi
    }
  }

  /// Relabels the corners for a pack lying rotated in the image. One quarter turn means the pack's
  /// top edge runs along the image's left edge (pack top-left = image bottom-left).
  public func rotated(quarterTurns: Int) -> Quad {
    let turns = ((quarterTurns % 4) + 4) % 4
    var c = corners
    for _ in 0..<turns { c = [c[3], c[0], c[1], c[2]] }
    return Quad(topLeft: c[0], topRight: c[1], bottomRight: c[2], bottomLeft: c[3])
  }

  /// Axis-aligned bounding box.
  public var boundingBox: Rect2D {
    let xs = corners.map(\.x)
    let ys = corners.map(\.y)
    let minX = xs.min() ?? 0
    let minY = ys.min() ?? 0
    return Rect2D(x: minX, y: minY, width: (xs.max() ?? 0) - minX, height: (ys.max() ?? 0) - minY)
  }
}
