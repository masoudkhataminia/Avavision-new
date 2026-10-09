import AvaVisionCore
import CoreGraphics
import Foundation

/// Persists the brain and its pill crops under `Application Support/AvaVision/brain`.
/// Crops are protected with complete file protection and excluded from iCloud backup.
actor BrainStore {
  nonisolated let root: URL
  nonisolated let cropsDirectory: URL
  private let state: JSONFileStore<BrainState>

  init(root: URL) {
    self.root = root
    cropsDirectory = root.appendingPathComponent("crops", isDirectory: true)
    state = JSONFileStore(fileURL: root.appendingPathComponent("brain.json"))
  }

  func load() async throws -> BrainState? {
    try await state.load()
  }

  func save(_ brain: BrainState) async throws {
    try await state.save(brain)
  }

  /// Stores a crop and returns its file name.
  func saveCrop(_ image: CGImage, id: UUID = UUID()) throws -> String {
    try FileManager.default.createDirectory(at: cropsDirectory, withIntermediateDirectories: true)
    var directory = cropsDirectory
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try? directory.setResourceValues(values)
    guard let data = PillCrops.jpegData(image) else { throw CocoaError(.fileWriteUnknown) }
    let name = "\(id.uuidString).jpg"
    #if os(iOS)
      try data.write(to: cropsDirectory.appendingPathComponent(name), options: [.atomic, .completeFileProtection])
    #else
      try data.write(to: cropsDirectory.appendingPathComponent(name), options: .atomic)
    #endif
    return name
  }

  nonisolated func cropURL(_ name: String) -> URL {
    cropsDirectory.appendingPathComponent(name)
  }

  /// Removes crops that no exemplar or labelling task refers to any more.
  func pruneCrops(keeping names: Set<String>) {
    guard let files = try? FileManager.default.contentsOfDirectory(atPath: cropsDirectory.path) else { return }
    for file in files where !names.contains(file) {
      try? FileManager.default.removeItem(at: cropsDirectory.appendingPathComponent(file))
    }
  }

  /// Writes every remembered crop and its label to `destination` for offline training:
  /// `crops/<file>.jpg`, `labels.jsonl` with one exemplar per line, and `brain.json` for
  /// `avavision brain-report`.
  func exportTrainingData(_ brain: BrainState, to destination: URL) throws -> Int {
    let fileManager = FileManager.default
    try? fileManager.removeItem(at: destination)
    let crops = destination.appendingPathComponent("crops", isDirectory: true)
    try fileManager.createDirectory(at: crops, withIntermediateDirectories: true)
    struct Line: Encodable {
      let crop: String
      let medicationID: MedicationID
      let source: ExemplarSource
      let groupID: UUID
      let createdAt: Date
    }
    var lines = Data()
    var count = 0
    let encoder = AvaVisionJSON.encoder()
    for exemplar in brain.knowledge.exemplars {
      guard let name = exemplar.cropFile, fileManager.fileExists(atPath: cropURL(name).path) else { continue }
      try fileManager.copyItem(at: cropURL(name), to: crops.appendingPathComponent(name))
      lines.append(
        try encoder.encode(
          Line(
            crop: "crops/\(name)", medicationID: exemplar.medicationID, source: exemplar.source,
            groupID: exemplar.groupID, createdAt: exemplar.createdAt)))
      lines.append(Data("\n".utf8))
      count += 1
    }
    try lines.write(to: destination.appendingPathComponent("labels.jsonl"), options: .atomic)
    try AvaVisionJSON.encoder(pretty: true).encode(brain).write(
      to: destination.appendingPathComponent("brain.json"), options: .atomic)
    return count
  }
}
