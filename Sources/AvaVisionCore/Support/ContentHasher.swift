import Crypto
import Foundation

/// SHA-256 helpers used for model integrity, image provenance and the audit chain.
public enum ContentHasher {
  public static func sha256Hex(_ data: Data) -> String {
    hex(SHA256.hash(data: data))
  }

  public static func sha256Hex(_ string: String) -> String {
    sha256Hex(Data(string.utf8))
  }

  /// Hashes a file, or a directory (such as a compiled `.mlmodelc`) as the hash of its sorted
  /// `relative-path NUL file-hash LF` lines. `.DS_Store` files are ignored.
  public static func sha256Hex(contentsOf url: URL) throws -> String {
    let base = url.standardizedFileURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: base.path, isDirectory: &isDirectory) else {
      throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: base.path])
    }
    guard isDirectory.boolValue else {
      return sha256Hex(try Data(contentsOf: base, options: .mappedIfSafe))
    }

    guard
      let enumerator = FileManager.default.enumerator(
        at: base, includingPropertiesForKeys: [.isRegularFileKey], options: [])
    else {
      throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: base.path])
    }
    let prefix = base.path.hasSuffix("/") ? base.path : base.path + "/"
    var lines: [String] = []
    for case let file as URL in enumerator {
      let resolved = file.standardizedFileURL.resolvingSymlinksInPath()
      guard try resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true,
        resolved.lastPathComponent != ".DS_Store"
      else { continue }
      let relative = resolved.path.hasPrefix(prefix) ? String(resolved.path.dropFirst(prefix.count)) : resolved.path
      let fileHash = sha256Hex(try Data(contentsOf: resolved, options: .mappedIfSafe))
      lines.append("\(relative)\u{0}\(fileHash)\n")
    }
    return sha256Hex(lines.sorted().joined())
  }

  private static func hex<D: Sequence>(_ bytes: D) -> String where D.Element == UInt8 {
    let digits = Array("0123456789abcdef".utf8)
    var output = [UInt8]()
    for byte in bytes {
      output.append(digits[Int(byte >> 4)])
      output.append(digits[Int(byte & 0x0f)])
    }
    return String(decoding: output, as: UTF8.self)
  }
}
