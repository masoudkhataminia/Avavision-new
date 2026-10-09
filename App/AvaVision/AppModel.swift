import AvaVisionCore
import Foundation
import Observation
import UIKit
import Vision

/// Application state shared by all screens: catalog, profiles, layouts, model and audit trail.
@MainActor
@Observable
final class AppModel {
  private(set) var catalog = MedicationCatalog()
  private(set) var profiles: [PackProfile] = []
  private(set) var layouts: [PackLayout] = [.weekly7x4]
  private(set) var modelState: ModelState = .loading
  private(set) var isLoaded = false
  /// The learning brain; `nil` until loaded.
  private(set) var brain: BrainState?
  private(set) var embedderTitle = ""
  private(set) var brainWarning: String?
  /// Set while the brain is busy re-learning (for example after an embedder upgrade).
  private(set) var brainActivity: String?
  var lastError: String?

  @ObservationIgnored let storage: Storage?
  @ObservationIgnored private(set) var embedder: PillEmbedder?
  /// Brain writes run one after another, like the other stores.
  @ObservationIgnored private var lastBrainWrite: Task<Void, Never>?
  /// Writes run one after another so a later snapshot can never be overwritten by an earlier one.
  @ObservationIgnored private var lastWrite: Task<Void, Never>?

  init(storage: Storage? = try? Storage.applicationSupport()) {
    self.storage = storage
  }

  func load() async {
    guard !isLoaded else { return }
    guard let storage else {
      lastError = "The app's data folder could not be opened."
      return
    }
    do {
      catalog = try await storage.catalog.load() ?? MedicationCatalog()
      profiles = try await storage.profiles.load() ?? []
      layouts = Self.merged(try await storage.layouts.load() ?? [])
    } catch {
      lastError = "Saved data could not be read: \(error.localizedDescription)"
    }
    isLoaded = true
    modelState = await Task.detached(priority: .userInitiated) { ModelLoader.load() }.value
    await loadBrain()
  }

  private func loadBrain() async {
    let (embedder, warning) = await Task.detached(priority: .userInitiated) { EmbedderLoader.load() }.value
    self.embedder = embedder
    embedderTitle = embedder.title
    brainWarning = warning
    guard let store = storage?.brain else {
      brain = BrainState(embedderID: embedder.id)
      return
    }
    do {
      let saved = try await store.load()
      if let saved, saved.embedderID != embedder.id {
        brainActivity = "Re-learning \(saved.knowledge.exemplars.count) remembered pills with the new embedder…"
        let migrated = await Task.detached(priority: .utility) {
          Self.reembed(saved, with: embedder, store: store)
        }.value
        brain = migrated
        brainActivity = nil
        persistBrain()
      } else {
        brain = saved ?? BrainState(embedderID: embedder.id)
      }
    } catch {
      brain = BrainState(embedderID: embedder.id)
      lastError = "The brain could not be read: \(error.localizedDescription)"
    }
  }

  nonisolated private static func reembed(_ brain: BrainState, with embedder: PillEmbedder, store: BrainStore)
    -> BrainState
  {
    var embeddings: [UUID: Embedding] = [:]
    for exemplar in brain.knowledge.exemplars {
      guard let name = exemplar.cropFile, let crop = PillCrops.image(contentsOf: store.cropURL(name)),
        let embedding = try? embedder.embed(crop)
      else { continue }
      embeddings[exemplar.id] = embedding
    }
    return brain.migrated(to: embedder.id, embeddings: embeddings)
  }

  /// Built-in layouts, replaced by any saved (calibrated) version with the same identifier.
  private static func merged(_ saved: [PackLayout]) -> [PackLayout] {
    let builtIn = [PackLayout.weekly7x4]
    return builtIn.map { layout in saved.first { $0.id == layout.id } ?? layout }
  }

  // MARK: - Lookups

  func layout(id: String) -> PackLayout? {
    layouts.first { $0.id == id }
  }

  func profile(id: UUID) -> PackProfile? {
    profiles.first { $0.id == id }
  }

  func issues(for profile: PackProfile) -> [PackProfile.Issue] {
    guard let layout = layout(id: profile.layoutID) else {
      return [.layoutMismatch(expected: "a known layout", found: profile.layoutID)]
    }
    return profile.issues(layout: layout, catalog: catalog)
  }

  // MARK: - Editing

  func upsertMedication(_ medication: Medication) {
    catalog.upsert(medication)
    persistCatalog()
  }

  func deleteMedication(_ id: MedicationID) {
    catalog.remove(id)
    persistCatalog()
  }

  /// True when any saved profile still expects this medication.
  func isMedicationInUse(_ id: MedicationID) -> Bool {
    profiles.contains { profile in
      profile.compartments.contains { $0.items.contains { $0.medicationID == id } }
    }
  }

  func saveProfile(_ profile: PackProfile) {
    if let index = profiles.firstIndex(where: { $0.id == profile.id }) {
      profiles[index] = profile
    } else {
      profiles.append(profile)
    }
    profiles.sort { $0.reference.localizedStandardCompare($1.reference) == .orderedAscending }
    persistProfiles()
  }

  func deleteProfile(_ id: UUID) {
    profiles.removeAll { $0.id == id }
    persistProfiles()
  }

  func saveLayout(_ layout: PackLayout) {
    guard let index = layouts.firstIndex(where: { $0.id == layout.id }) else { return }
    layouts[index] = layout
    let snapshot = layouts
    persist { try await $0.layouts.save(snapshot) }
  }

  private func persistCatalog() {
    let snapshot = catalog
    persist { try await $0.catalog.save(snapshot) }
  }

  private func persistProfiles() {
    let snapshot = profiles
    persist { try await $0.profiles.save(snapshot) }
  }

  private func persist(_ write: @escaping @Sendable (Storage) async throws -> Void) {
    guard let storage else { return }
    let previous = lastWrite
    lastWrite = Task {
      await previous?.value
      do {
        try await write(storage)
      } catch {
        lastError = "Saving failed: \(error.localizedDescription)"
      }
    }
  }

  /// Waits until every pending write has reached the disk.
  func flushWrites() async {
    await lastWrite?.value
    await lastBrainWrite?.value
  }

  // MARK: - Checking

  func engine(for layout: PackLayout) -> VerificationEngine {
    VerificationEngine(layout: layout, model: modelState.activeModel, brain: brain.map(BrainContext.init))
  }

  func analyzer(for layout: PackLayout, orientation: PackOrientation) -> FrameAnalyzer {
    FrameAnalyzer(layout: layout, orientation: orientation, detector: modelState.detector)
  }

  /// Asks the brain about pills; `nil` until the brain and embedder are ready.
  func pillIdentifier() -> PillIdentifier? {
    guard let brain, let embedder, brain.embedderID == embedder.id else { return nil }
    return PillIdentifier(embedder: embedder, classifier: brain.classifier())
  }

  // MARK: - Brain

  /// Learns from a signed-off check: only pills in compartments the pharmacist confirmed correct.
  func learn(from record: CheckRecord, pills: [IdentifiedPill]) async {
    guard var brain, let store = storage?.brain else { return }
    let confirmed = Set(record.signOff.reviews.filter { $0.outcome == .confirmedCorrect }.map(\.compartment))
    var sightings: [PillSighting] = []
    for pill in pills where confirmed.contains(pill.compartment) {
      let id = UUID()
      let file = try? await store.saveCrop(pill.crop, id: id)
      sightings.append(
        PillSighting(
          id: id, compartment: pill.compartment, embedding: pill.embedding, identity: pill.identity, cropFile: file))
    }
    let plan = LearningPlanner.plan(
      checkID: record.id, signOff: record.signOff, profile: record.profile, layout: record.layout,
      sightings: sightings)
    brain.learn(plan)
    self.brain = brain
    persistBrain(pruneCrops: true)
  }

  /// Teaches a medication from crops of pills the pharmacist knows to be that medication.
  func teach(_ medication: MedicationID, crops: [CGImage]) async -> Int {
    guard var brain, let embedder, let store = storage?.brain else { return 0 }
    var embeddings: [(embedding: Embedding, cropFile: String?)] = []
    for crop in crops {
      guard let embedding = try? embedder.embed(crop) else { continue }
      embeddings.append((embedding, try? await store.saveCrop(crop)))
    }
    let summary = brain.teach(medication, embeddings: embeddings)
    self.brain = brain
    persistBrain(pruneCrops: true)
    return summary.exemplarsAdded
  }

  func resolveLabellingTask(_ id: UUID, labels: [UUID: MedicationID]) throws {
    guard var brain else { return }
    try brain.resolveTask(id, labels: labels)
    self.brain = brain
    persistBrain(pruneCrops: true)
  }

  func discardLabellingTask(_ id: UUID) {
    brain?.discardTask(id)
    persistBrain(pruneCrops: true)
  }

  func recalibrateBrain() {
    brain?.recalibrate()
    persistBrain()
  }

  func forgetInBrain(_ id: MedicationID) {
    brain?.forget(medication: id)
    persistBrain(pruneCrops: true)
  }

  func cropURL(_ name: String) -> URL? {
    storage?.brain.cropURL(name)
  }

  /// Exports remembered crops and labels to the Documents folder (visible in the Files app).
  func exportTrainingData() async throws -> Int {
    guard let brain, let store = storage?.brain else { return 0 }
    let documents = try FileManager.default.url(
      for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    return try await store.exportTrainingData(brain, to: documents.appendingPathComponent("AvaVision-training-data"))
  }

  private func persistBrain(pruneCrops: Bool = false) {
    guard let brain, let store = storage?.brain else { return }
    let previous = lastBrainWrite
    lastBrainWrite = Task {
      await previous?.value
      do {
        try await store.save(brain)
        if pruneCrops {
          let referenced = Set(
            brain.knowledge.exemplars.compactMap(\.cropFile)
              + brain.labellingQueue.flatMap { $0.sightings.compactMap(\.cropFile) })
          await store.pruneCrops(keeping: referenced)
        }
      } catch {
        lastError = "Saving the brain failed: \(error.localizedDescription)"
      }
    }
  }

  func appendToAudit(_ record: CheckRecord) async throws {
    guard let storage else { throw CocoaError(.fileWriteUnknown) }
    try await storage.audit.append(record)
  }

  var appVersion: String {
    let info = Bundle.main.infoDictionary
    let version = info?["CFBundleShortVersionString"] as? String ?? "0"
    let build = info?["CFBundleVersion"] as? String ?? "0"
    return "\(version) (\(build))"
  }

  var deviceIdentifier: String {
    UIDevice.current.identifierForVendor?.uuidString ?? "unknown-device"
  }
}
