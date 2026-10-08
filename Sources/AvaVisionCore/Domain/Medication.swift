import Foundation

/// Visual description a pharmacist records for a medication. Used for display and future
/// multi-evidence checks (colour, shape, imprint); never used to infer identity on its own.
public struct MedicationAppearance: Codable, Hashable, Sendable {
  public var colour: String?
  public var shape: String?
  public var imprint: String?

  public init(colour: String? = nil, shape: String? = nil, imprint: String? = nil) {
    self.colour = colour
    self.shape = shape
    self.imprint = imprint
  }
}

/// A product the pharmacy packs. Contains no patient information.
public struct Medication: Codable, Hashable, Sendable, Identifiable {
  public var id: MedicationID
  public var name: String
  public var strength: String
  public var appearance: MedicationAppearance

  public init(
    id: MedicationID, name: String, strength: String = "",
    appearance: MedicationAppearance = MedicationAppearance()
  ) {
    self.id = id
    self.name = name
    self.strength = strength
    self.appearance = appearance
  }

  public var displayName: String {
    strength.isEmpty ? name : "\(name) \(strength)"
  }
}

/// The set of medications known to this pharmacy installation.
public struct MedicationCatalog: Codable, Hashable, Sendable {
  public private(set) var medications: [Medication]

  public init(medications: [Medication] = []) {
    self.medications = []
    for medication in medications { upsert(medication) }
  }

  public func medication(_ id: MedicationID) -> Medication? {
    medications.first { $0.id == id }
  }

  public func contains(_ id: MedicationID) -> Bool {
    medication(id) != nil
  }

  /// Inserts the medication or replaces the one with the same identifier, keeping the list sorted.
  public mutating func upsert(_ medication: Medication) {
    medications.removeAll { $0.id == medication.id }
    medications.append(medication)
    medications.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }

  public mutating func remove(_ id: MedicationID) {
    medications.removeAll { $0.id == id }
  }
}
