import Foundation

/// Model information captured in the audit trail.
public struct ModelSummary: Codable, Hashable, Sendable {
  public var modelID: String
  public var version: String
  public var stage: ModelStage
  public var modelSHA256: String
  public var capability: ModelCapability

  public init(_ model: ActiveModel) {
    modelID = model.manifest.modelID
    version = model.manifest.version
    stage = model.manifest.stage
    modelSHA256 = model.manifest.modelSHA256
    capability = model.capability
  }
}

/// Everything needed to reconstruct one completed pack check.
public struct CheckRecord: Codable, Hashable, Sendable, Identifiable {
  public var id: UUID
  public var createdAt: Date
  public var appVersion: String
  public var deviceIdentifier: String
  public var layout: PackLayout
  public var profile: PackProfile
  public var model: ModelSummary?
  public var result: PackVerificationResult
  public var signOff: PharmacistSignOff
  public var frameImageSHA256s: [String]

  public init(
    id: UUID = UUID(), createdAt: Date = Date(), appVersion: String, deviceIdentifier: String,
    layout: PackLayout, profile: PackProfile, model: ModelSummary?, result: PackVerificationResult,
    signOff: PharmacistSignOff, frameImageSHA256s: [String]
  ) {
    self.id = id
    self.createdAt = createdAt
    self.appVersion = appVersion
    self.deviceIdentifier = deviceIdentifier
    self.layout = layout
    self.profile = profile
    self.model = model
    self.result = result
    self.signOff = signOff
    self.frameImageSHA256s = frameImageSHA256s
  }
}

/// One link of the tamper-evident audit chain. `payload` is the exact JSON that was hashed.
public struct AuditEntry: Codable, Hashable, Sendable {
  public let sequence: Int
  public let previousHash: String
  public let payload: String
  public let hash: String

  static func hash(sequence: Int, previousHash: String, payload: String) -> String {
    ContentHasher.sha256Hex("\(sequence)\n\(previousHash)\n\(payload)")
  }
}

public enum AuditVerification: Hashable, Sendable {
  public enum Defect: String, Hashable, Sendable {
    case sequenceGap
    case previousHashMismatch
    case hashMismatch
  }

  case intact(entries: Int)
  case broken(sequence: Int, defect: Defect)
}

/// Append-only, hash-chained list of check records. Editing or removing any entry breaks the
/// chain from that point on, which `verify()` detects.
public struct AuditChain: Sendable {
  public static let genesisHash = String(repeating: "0", count: 64)

  public private(set) var entries: [AuditEntry]

  public init(entries: [AuditEntry] = []) {
    self.entries = entries
  }

  @discardableResult
  public mutating func append(_ record: CheckRecord) throws -> AuditEntry {
    let data = try AvaVisionJSON.encoder().encode(record)
    let payload = String(decoding: data, as: UTF8.self)
    let sequence = entries.count
    let previous = entries.last?.hash ?? Self.genesisHash
    let entry = AuditEntry(
      sequence: sequence, previousHash: previous, payload: payload,
      hash: AuditEntry.hash(sequence: sequence, previousHash: previous, payload: payload))
    entries.append(entry)
    return entry
  }

  public func verify() -> AuditVerification {
    var previous = Self.genesisHash
    for (position, entry) in entries.enumerated() {
      guard entry.sequence == position else { return .broken(sequence: position, defect: .sequenceGap) }
      guard entry.previousHash == previous else {
        return .broken(sequence: position, defect: .previousHashMismatch)
      }
      let expected = AuditEntry.hash(sequence: entry.sequence, previousHash: entry.previousHash, payload: entry.payload)
      guard entry.hash == expected else { return .broken(sequence: position, defect: .hashMismatch) }
      previous = entry.hash
    }
    return .intact(entries: entries.count)
  }

  public func records() throws -> [CheckRecord] {
    let decoder = AvaVisionJSON.decoder()
    return try entries.map { try decoder.decode(CheckRecord.self, from: Data($0.payload.utf8)) }
  }
}

/// Persists the audit chain as JSON Lines (one `AuditEntry` per line), appending only.
public actor AuditLogStore {
  public nonisolated let fileURL: URL
  private var cached: AuditChain?

  public init(fileURL: URL) {
    self.fileURL = fileURL
  }

  public func chain() throws -> AuditChain {
    if let cached { return cached }
    let loaded: AuditChain
    if FileManager.default.fileExists(atPath: fileURL.path) {
      let decoder = AvaVisionJSON.decoder()
      let text = String(decoding: try Data(contentsOf: fileURL), as: UTF8.self)
      let entries = try text.split(separator: "\n", omittingEmptySubsequences: true).map {
        try decoder.decode(AuditEntry.self, from: Data($0.utf8))
      }
      loaded = AuditChain(entries: entries)
    } else {
      loaded = AuditChain()
    }
    cached = loaded
    return loaded
  }

  /// Refuses to extend a chain that no longer verifies.
  @discardableResult
  public func append(_ record: CheckRecord) throws -> AuditEntry {
    var current = try chain()
    if case .broken(let sequence, let defect) = current.verify() {
      throw AuditStoreError.chainBroken(sequence: sequence, defect: defect)
    }
    let entry = try current.append(record)
    var line = try AvaVisionJSON.encoder().encode(entry)
    line.append(contentsOf: Data("\n".utf8))

    let directory = fileURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if !FileManager.default.fileExists(atPath: fileURL.path) {
      guard FileManager.default.createFile(atPath: fileURL.path, contents: nil) else {
        throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: fileURL.path])
      }
    }
    let handle = try FileHandle(forWritingTo: fileURL)
    defer { try? handle.close() }
    try handle.seekToEnd()
    try handle.write(contentsOf: line)
    cached = current
    return entry
  }

  public func records() throws -> [CheckRecord] {
    try chain().records()
  }

  public func verify() throws -> AuditVerification {
    try chain().verify()
  }
}

public enum AuditStoreError: Error, Hashable, Sendable {
  case chainBroken(sequence: Int, defect: AuditVerification.Defect)
}
