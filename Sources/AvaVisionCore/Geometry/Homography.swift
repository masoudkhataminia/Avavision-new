import Foundation

/// A projective transform between two planes, used to map camera-image coordinates onto the
/// flat pack card regardless of camera perspective.
public struct Homography: Codable, Hashable, Sendable {
  /// Row-major 3×3 matrix, normalized so that the last element is 1 when possible.
  public let matrix: [Double]

  public init?(matrix: [Double]) {
    guard matrix.count == 9, matrix.allSatisfy(\.isFinite) else { return nil }
    let scale = abs(matrix[8]) > 1e-12 ? matrix[8] : 1
    self.matrix = matrix.map { $0 / scale }
  }

  /// Solves the transform that maps each `source` point onto the matching `destination` point.
  /// Requires exactly four correspondences with no three source points collinear.
  public init?(mapping source: [Point2D], to destination: [Point2D]) {
    guard source.count == 4, destination.count == 4 else { return nil }
    var system = [[Double]]()
    for (s, d) in zip(source, destination) {
      system.append([s.x, s.y, 1, 0, 0, 0, -s.x * d.x, -s.y * d.x, d.x])
      system.append([0, 0, 0, s.x, s.y, 1, -s.x * d.y, -s.y * d.y, d.y])
    }
    guard let h = Self.solve(system) else { return nil }
    self.init(matrix: h + [1])
  }

  /// Maps a point; returns `nil` for points on the transform's line at infinity.
  public func apply(_ point: Point2D) -> Point2D? {
    let m = matrix
    let w = m[6] * point.x + m[7] * point.y + m[8]
    guard abs(w) > 1e-12 else { return nil }
    return Point2D(
      x: (m[0] * point.x + m[1] * point.y + m[2]) / w,
      y: (m[3] * point.x + m[4] * point.y + m[5]) / w
    )
  }

  public var inverse: Homography? {
    let m = matrix
    let a = m[4] * m[8] - m[5] * m[7]
    let b = m[5] * m[6] - m[3] * m[8]
    let c = m[3] * m[7] - m[4] * m[6]
    let determinant = m[0] * a + m[1] * b + m[2] * c
    guard abs(determinant) > 1e-12 else { return nil }
    let adjugate = [
      a, m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
      b, m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
      c, m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3],
    ]
    return Homography(matrix: adjugate.map { $0 / determinant })
  }

  /// Gaussian elimination with partial pivoting on an n×(n+1) augmented matrix.
  private static func solve(_ input: [[Double]]) -> [Double]? {
    var a = input
    let n = a.count
    for column in 0..<n {
      guard
        let pivot = (column..<n).max(by: { abs(a[$0][column]) < abs(a[$1][column]) }),
        abs(a[pivot][column]) > 1e-12
      else { return nil }
      a.swapAt(column, pivot)
      for row in 0..<n where row != column {
        let factor = a[row][column] / a[column][column]
        guard factor != 0 else { continue }
        for k in column...n {
          a[row][k] -= factor * a[column][k]
        }
      }
    }
    let solution = (0..<n).map { a[$0][n] / a[$0][$0] }
    return solution.allSatisfy(\.isFinite) ? solution : nil
  }
}
