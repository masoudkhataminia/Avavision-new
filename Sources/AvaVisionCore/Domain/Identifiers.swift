import Foundation

/// Stable identifier of a medication in the pharmacy catalog.
///
/// It identifies a product (for example `"metformin-500-tab"`), never a patient.
public struct MedicationID: RawRepresentable, Hashable, Comparable, Codable, CodingKeyRepresentable,
  Sendable, ExpressibleByStringLiteral, CustomStringConvertible
{
  public let rawValue: String

  public init(rawValue: String) { self.rawValue = rawValue }
  public init(_ rawValue: String) { self.rawValue = rawValue }
  public init(stringLiteral value: String) { self.rawValue = value }

  public var description: String { rawValue }

  public static func < (lhs: MedicationID, rhs: MedicationID) -> Bool {
    lhs.rawValue < rhs.rawValue
  }
}

/// Position of one compartment (blister cell) in a pack, zero-based.
public struct CompartmentIndex: Hashable, Comparable, Codable, Sendable, CustomStringConvertible {
  public let row: Int
  public let column: Int

  public init(row: Int, column: Int) {
    self.row = row
    self.column = column
  }

  public var description: String { "r\(row)c\(column)" }

  public static func < (lhs: CompartmentIndex, rhs: CompartmentIndex) -> Bool {
    (lhs.row, lhs.column) < (rhs.row, rhs.column)
  }
}
