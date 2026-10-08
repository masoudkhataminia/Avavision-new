import Foundation

/// Stores one Codable value as a JSON file, written atomically.
public actor JSONFileStore<Value: Codable & Sendable> {
  public nonisolated let fileURL: URL

  public init(fileURL: URL) {
    self.fileURL = fileURL
  }

  /// Returns `nil` when nothing has been saved yet.
  public func load() throws -> Value? {
    guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
    return try AvaVisionJSON.decoder().decode(Value.self, from: Data(contentsOf: fileURL))
  }

  public func save(_ value: Value) throws {
    try FileManager.default.createDirectory(
      at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try AvaVisionJSON.encoder(pretty: true).encode(value).write(to: fileURL, options: .atomic)
  }
}
