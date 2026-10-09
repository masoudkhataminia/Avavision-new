import Foundation

/// Identifies the model that produced an embedding. Embeddings from different embedders live in
/// different spaces and are never compared with each other.
public struct EmbedderID: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral,
  CustomStringConvertible
{
  public let rawValue: String

  public init(rawValue: String) { self.rawValue = rawValue }
  public init(_ rawValue: String) { self.rawValue = rawValue }
  public init(stringLiteral value: String) { self.rawValue = value }

  public var description: String { rawValue }
}

/// An L2-normalized appearance vector of one pill image.
public struct Embedding: Hashable, Sendable {
  public let embedderID: EmbedderID
  public let vector: [Float]

  /// Normalizes `raw`; returns `nil` for empty, zero or non-finite vectors.
  public init?(embedderID: EmbedderID, raw: [Float]) {
    guard !raw.isEmpty, raw.allSatisfy(\.isFinite) else { return nil }
    let norm = raw.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
    guard norm > 0, norm.isFinite else { return nil }
    self.embedderID = embedderID
    self.vector = raw.map { $0 / norm }
  }

  public var dimension: Int { vector.count }

  /// Cosine similarity in −1…1, or `nil` if the embeddings are not comparable.
  public func similarity(to other: Embedding) -> Float? {
    guard embedderID == other.embedderID, dimension == other.dimension else { return nil }
    return VectorMath.dot(vector, other.vector)
  }
}

extension Embedding: Codable {
  private enum CodingKeys: String, CodingKey {
    case embedderID
    case dimension
    case vector
  }

  /// The vector is stored as little-endian Float32 bytes (base64 in JSON) to keep files compact.
  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let id = try container.decode(EmbedderID.self, forKey: .embedderID)
    let dimension = try container.decode(Int.self, forKey: .dimension)
    let data = try container.decode(Data.self, forKey: .vector)
    guard dimension > 0, data.count == dimension * 4 else {
      throw DecodingError.dataCorruptedError(
        forKey: .vector, in: container, debugDescription: "Vector size does not match dimension")
    }
    let floats = (0..<dimension).map { index -> Float in
      let base = data.startIndex + index * 4
      let bits =
        UInt32(data[base]) | UInt32(data[base + 1]) << 8 | UInt32(data[base + 2]) << 16 | UInt32(data[base + 3]) << 24
      return Float(bitPattern: bits)
    }
    guard let embedding = Embedding(embedderID: id, raw: floats) else {
      throw DecodingError.dataCorruptedError(
        forKey: .vector, in: container, debugDescription: "Vector is zero or not finite")
    }
    self = embedding
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(embedderID, forKey: .embedderID)
    try container.encode(dimension, forKey: .dimension)
    var data = Data(capacity: dimension * 4)
    for value in vector {
      let bits = value.bitPattern
      data.append(contentsOf: [
        UInt8(truncatingIfNeeded: bits), UInt8(truncatingIfNeeded: bits >> 8),
        UInt8(truncatingIfNeeded: bits >> 16), UInt8(truncatingIfNeeded: bits >> 24),
      ])
    }
    try container.encode(data, forKey: .vector)
  }
}

enum VectorMath {
  @inline(__always)
  static func dot(_ a: [Float], _ b: [Float]) -> Float {
    a.withUnsafeBufferPointer { x in
      b.withUnsafeBufferPointer { y in
        var sum: Float = 0
        for i in 0..<min(x.count, y.count) { sum += x[i] * y[i] }
        return sum
      }
    }
  }
}
