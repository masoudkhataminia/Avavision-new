import Foundation

/// SHA-256 (FIPS 180-4), implemented in-house so the core has no third-party dependencies.
public struct SHA256Hasher: Sendable {
  private static let roundConstants: [UInt32] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
  ]

  private var state: [UInt32] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
  ]
  private var buffer: [UInt8] = []
  private var totalBytes: UInt64 = 0

  public init() {
    buffer.reserveCapacity(64)
  }

  public mutating func update(_ data: Data) {
    data.withUnsafeBytes { update(bytes: $0) }
  }

  public mutating func update(bytes: UnsafeRawBufferPointer) {
    guard bytes.count > 0 else { return }
    totalBytes &+= UInt64(bytes.count)
    var offset = 0
    if !buffer.isEmpty {
      let take = min(64 - buffer.count, bytes.count)
      buffer.append(contentsOf: bytes[0..<take])
      offset = take
      if buffer.count == 64 {
        buffer.withUnsafeBytes { compress(block: $0) }
        buffer.removeAll(keepingCapacity: true)
      }
    }
    while bytes.count - offset >= 64 {
      compress(block: UnsafeRawBufferPointer(rebasing: bytes[offset..<(offset + 64)]))
      offset += 64
    }
    if offset < bytes.count {
      buffer.append(contentsOf: bytes[offset...])
    }
  }

  /// Returns the 32-byte digest. The hasher should not be used afterwards.
  public mutating func finalize() -> [UInt8] {
    let bitLength = totalBytes &* 8
    var padding: [UInt8] = [0x80]
    let used = (buffer.count + 1) % 64
    padding.append(contentsOf: repeatElement(0, count: used <= 56 ? 56 - used : 120 - used))
    for shift in stride(from: 56, through: 0, by: -8) {
      padding.append(UInt8(truncatingIfNeeded: bitLength >> UInt64(shift)))
    }
    let length = totalBytes
    padding.withUnsafeBytes { update(bytes: $0) }
    totalBytes = length
    precondition(buffer.isEmpty, "SHA-256 padding must end on a block boundary")

    var digest: [UInt8] = []
    digest.reserveCapacity(32)
    for word in state {
      digest.append(UInt8(truncatingIfNeeded: word >> 24))
      digest.append(UInt8(truncatingIfNeeded: word >> 16))
      digest.append(UInt8(truncatingIfNeeded: word >> 8))
      digest.append(UInt8(truncatingIfNeeded: word))
    }
    return digest
  }

  public static func hash(_ data: Data) -> [UInt8] {
    var hasher = SHA256Hasher()
    hasher.update(data)
    return hasher.finalize()
  }

  @inline(__always)
  private static func rotateRight(_ value: UInt32, _ count: UInt32) -> UInt32 {
    (value >> count) | (value << (32 - count))
  }

  private mutating func compress(block: UnsafeRawBufferPointer) {
    var w = [UInt32](repeating: 0, count: 64)
    for i in 0..<16 {
      let base = i * 4
      w[i] =
        UInt32(block[base]) << 24 | UInt32(block[base + 1]) << 16 | UInt32(block[base + 2]) << 8
        | UInt32(block[base + 3])
    }
    for i in 16..<64 {
      let s0 = Self.rotateRight(w[i - 15], 7) ^ Self.rotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3)
      let s1 = Self.rotateRight(w[i - 2], 17) ^ Self.rotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10)
      w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
    }

    var a = state[0]
    var b = state[1]
    var c = state[2]
    var d = state[3]
    var e = state[4]
    var f = state[5]
    var g = state[6]
    var h = state[7]
    for i in 0..<64 {
      let s1 = Self.rotateRight(e, 6) ^ Self.rotateRight(e, 11) ^ Self.rotateRight(e, 25)
      let choice = (e & f) ^ (~e & g)
      let temp1 = h &+ s1 &+ choice &+ Self.roundConstants[i] &+ w[i]
      let s0 = Self.rotateRight(a, 2) ^ Self.rotateRight(a, 13) ^ Self.rotateRight(a, 22)
      let majority = (a & b) ^ (a & c) ^ (b & c)
      let temp2 = s0 &+ majority
      h = g
      g = f
      f = e
      e = d &+ temp1
      d = c
      c = b
      b = a
      a = temp1 &+ temp2
    }
    state[0] &+= a
    state[1] &+= b
    state[2] &+= c
    state[3] &+= d
    state[4] &+= e
    state[5] &+= f
    state[6] &+= g
    state[7] &+= h
  }
}
