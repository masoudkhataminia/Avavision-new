import AvaVisionCore
import Foundation

/// Locations of everything the app persists. No pack photos and no patient data are stored; the
/// brain keeps single-pill crops only.
struct Storage: Sendable {
  let root: URL
  let catalog: JSONFileStore<MedicationCatalog>
  let profiles: JSONFileStore<[PackProfile]>
  let layouts: JSONFileStore<[PackLayout]>
  let audit: AuditLogStore
  let brain: BrainStore

  init(root: URL) {
    self.root = root
    catalog = JSONFileStore(fileURL: root.appendingPathComponent("catalog.json"))
    profiles = JSONFileStore(fileURL: root.appendingPathComponent("profiles.json"))
    layouts = JSONFileStore(fileURL: root.appendingPathComponent("layouts.json"))
    audit = AuditLogStore(fileURL: root.appendingPathComponent("audit/audit-log.jsonl"))
    brain = BrainStore(root: root.appendingPathComponent("brain", isDirectory: true))
  }

  var auditFileURL: URL { audit.fileURL }

  static func applicationSupport() throws -> Storage {
    let base = try FileManager.default.url(
      for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    let root = base.appendingPathComponent("AvaVision", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return Storage(root: root)
  }
}
