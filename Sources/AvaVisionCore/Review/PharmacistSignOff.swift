import Foundation

public enum CompartmentReviewOutcome: String, Codable, Hashable, Sendable, CaseIterable {
  /// The pharmacist inspected the compartment and its content is correct.
  case confirmedCorrect
  /// The content was wrong and the pharmacist corrected the pack.
  case corrected
  /// The content is wrong or uncertain and was not corrected.
  case unresolved
}

public struct CompartmentReview: Codable, Hashable, Sendable {
  public var compartment: CompartmentIndex
  public var outcome: CompartmentReviewOutcome
  public var note: String?

  public init(compartment: CompartmentIndex, outcome: CompartmentReviewOutcome, note: String? = nil) {
    self.compartment = compartment
    self.outcome = outcome
    self.note = note
  }
}

public enum SignOffDecision: String, Codable, Hashable, Sendable {
  /// The pack may leave the pharmacy.
  case released
  /// The pack must not leave the pharmacy.
  case withheld
}

/// The pharmacist's final, accountable decision on a pack.
public struct PharmacistSignOff: Codable, Hashable, Sendable {
  /// Pharmacist initials or staff code; never a patient identifier.
  public var pharmacistIdentifier: String
  public var decision: SignOffDecision
  public var reviews: [CompartmentReview]
  /// The pharmacist has looked at pack-level findings (for example a loose object on the card).
  public var acknowledgedPackFindings: Bool
  public var note: String?
  public var signedAt: Date

  public init(
    pharmacistIdentifier: String, decision: SignOffDecision, reviews: [CompartmentReview],
    acknowledgedPackFindings: Bool, note: String? = nil, signedAt: Date = Date()
  ) {
    self.pharmacistIdentifier = pharmacistIdentifier
    self.decision = decision
    self.reviews = reviews
    self.acknowledgedPackFindings = acknowledgedPackFindings
    self.note = note
    self.signedAt = signedAt
  }

  public func review(for index: CompartmentIndex) -> CompartmentReview? {
    reviews.last { $0.compartment == index }
  }
}

public enum SignOffError: Error, Hashable, Sendable {
  case missingPharmacistIdentifier
  case reviewForUnknownCompartment(CompartmentIndex)
  case unreviewedCompartments([CompartmentIndex])
  case unresolvedCompartments([CompartmentIndex])
  case packFindingsNotAcknowledged
}

public enum SignOffValidator {
  /// A pack can be released only when every compartment that is not `verified` has been inspected
  /// and is correct or corrected, and every pack-level finding has been acknowledged.
  /// Withholding a pack is always allowed.
  public static func validate(_ signOff: PharmacistSignOff, for result: PackVerificationResult) throws {
    guard !signOff.pharmacistIdentifier.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw SignOffError.missingPharmacistIdentifier
    }
    let known = Set(result.compartments.map(\.compartment))
    if let unknown = signOff.reviews.first(where: { !known.contains($0.compartment) }) {
      throw SignOffError.reviewForUnknownCompartment(unknown.compartment)
    }
    guard signOff.decision == .released else { return }

    let unreviewed = result.compartmentsRequiringReview.filter { signOff.review(for: $0) == nil }
    guard unreviewed.isEmpty else { throw SignOffError.unreviewedCompartments(unreviewed) }

    let unresolved = result.compartments.map(\.compartment).filter {
      signOff.review(for: $0)?.outcome == .unresolved
    }
    guard unresolved.isEmpty else { throw SignOffError.unresolvedCompartments(unresolved) }

    if !result.packFindings.isEmpty, !signOff.acknowledgedPackFindings {
      throw SignOffError.packFindingsNotAcknowledged
    }
  }
}
