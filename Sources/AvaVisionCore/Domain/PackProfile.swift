import Foundation

/// One medication expected in a compartment.
public struct ExpectedItem: Codable, Hashable, Sendable {
  public var medicationID: MedicationID
  public var quantity: Int

  public init(medicationID: MedicationID, quantity: Int) {
    self.medicationID = medicationID
    self.quantity = quantity
  }
}

/// Everything a compartment is supposed to contain. An empty `items` list means the
/// compartment must be empty, which is different from the compartment being unspecified.
public struct CompartmentExpectation: Codable, Hashable, Sendable {
  public var compartment: CompartmentIndex
  public var items: [ExpectedItem]

  public init(compartment: CompartmentIndex, items: [ExpectedItem]) {
    self.compartment = compartment
    self.items = items
  }

  public var totalQuantity: Int { items.reduce(0) { $0 + $1.quantity } }

  /// Quantity per medication, merging duplicate lines.
  public var quantities: [MedicationID: Int] {
    items.reduce(into: [:]) { $0[$1.medicationID, default: 0] += $1.quantity }
  }
}

/// The expected content of one physical pack.
///
/// `reference` is an opaque pharmacy reference (for example a pack barcode). Patient names,
/// dates of birth, addresses or script numbers must never be stored here.
public struct PackProfile: Codable, Hashable, Sendable, Identifiable {
  public var id: UUID
  public var reference: String
  public var layoutID: String
  public var compartments: [CompartmentExpectation]
  public var createdAt: Date

  public init(
    id: UUID = UUID(), reference: String, layoutID: String,
    compartments: [CompartmentExpectation], createdAt: Date = Date()
  ) {
    self.id = id
    self.reference = reference
    self.layoutID = layoutID
    self.compartments = compartments
    self.createdAt = createdAt
  }

  /// A profile where every compartment of `layout` is explicitly empty.
  public static func empty(reference: String, layout: PackLayout) -> PackProfile {
    PackProfile(
      reference: reference,
      layoutID: layout.id,
      compartments: layout.allCompartments.map { CompartmentExpectation(compartment: $0, items: []) }
    )
  }

  public func expectation(for index: CompartmentIndex) -> CompartmentExpectation? {
    compartments.first { $0.compartment == index }
  }

  public mutating func setItems(_ items: [ExpectedItem], for index: CompartmentIndex) {
    compartments.removeAll { $0.compartment == index }
    compartments.append(CompartmentExpectation(compartment: index, items: items))
    compartments.sort { $0.compartment < $1.compartment }
  }

  public enum Issue: Hashable, Sendable {
    case emptyReference
    case layoutMismatch(expected: String, found: String)
    case compartmentOutsideLayout(CompartmentIndex)
    case duplicateCompartment(CompartmentIndex)
    case unspecifiedCompartment(CompartmentIndex)
    case unknownMedication(MedicationID)
    case invalidQuantity(CompartmentIndex, MedicationID)
  }

  /// Returns every reason the profile cannot be trusted as the expected state of the pack.
  public func issues(layout: PackLayout, catalog: MedicationCatalog) -> [Issue] {
    var issues: [Issue] = []
    if reference.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      issues.append(.emptyReference)
    }
    if layoutID != layout.id {
      issues.append(.layoutMismatch(expected: layout.id, found: layoutID))
    }
    var seen = Set<CompartmentIndex>()
    var unknown = Set<MedicationID>()
    for expectation in compartments {
      let index = expectation.compartment
      if !layout.contains(index) { issues.append(.compartmentOutsideLayout(index)) }
      if !seen.insert(index).inserted { issues.append(.duplicateCompartment(index)) }
      for item in expectation.items {
        if item.quantity < 1 { issues.append(.invalidQuantity(index, item.medicationID)) }
        if !catalog.contains(item.medicationID), unknown.insert(item.medicationID).inserted {
          issues.append(.unknownMedication(item.medicationID))
        }
      }
    }
    for index in layout.allCompartments where !seen.contains(index) {
      issues.append(.unspecifiedCompartment(index))
    }
    return issues
  }
}
