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
  var lastError: String?

  @ObservationIgnored let storage: Storage?
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
  }

  // MARK: - Checking

  func engine(for layout: PackLayout) -> VerificationEngine {
    VerificationEngine(layout: layout, model: modelState.activeModel)
  }

  func analyzer(for layout: PackLayout, orientation: PackOrientation) -> FrameAnalyzer {
    FrameAnalyzer(layout: layout, orientation: orientation, visionModel: modelState.visionModel)
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
