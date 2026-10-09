import AvaVisionCore
import SwiftUI

// Human-readable text, colours and symbols for core values. Wording is for Australian pharmacy staff.

extension CompartmentStatus {
  var title: String {
    switch self {
    case .verified: "Verified"
    case .countMatched: "Count OK"
    case .needsReview: "Review"
    case .mismatch: "Mismatch"
    }
  }

  var color: Color {
    switch self {
    case .verified: .green
    case .countMatched: .blue
    case .needsReview: .orange
    case .mismatch: .red
    }
  }

  var symbol: String {
    switch self {
    case .verified: "checkmark.seal.fill"
    case .countMatched: "number.circle.fill"
    case .needsReview: "questionmark.diamond.fill"
    case .mismatch: "xmark.octagon.fill"
    }
  }
}

extension PackStatus {
  var title: String {
    switch self {
    case .verified: "Verified — pharmacist sign-off required"
    case .countMatched: "Counts match — identity not checked"
    case .needsReview: "Pharmacist review required"
    case .mismatch: "Mismatch found"
    case .retakeRequired: "Retake photos"
    }
  }

  var color: Color {
    switch self {
    case .verified: .green
    case .countMatched: .blue
    case .needsReview: .orange
    case .mismatch: .red
    case .retakeRequired: .gray
    }
  }

  var symbol: String {
    switch self {
    case .verified: "checkmark.seal.fill"
    case .countMatched: "number.circle.fill"
    case .needsReview: "questionmark.diamond.fill"
    case .mismatch: "xmark.octagon.fill"
    case .retakeRequired: "camera.badge.ellipsis"
    }
  }
}

extension Finding {
  func text(catalog: MedicationCatalog) -> String {
    func name(_ id: MedicationID) -> String { catalog.medication(id)?.displayName ?? id.rawValue }
    switch self {
    case .missing(let expected, let observed):
      return "Missing dose: expected \(expected), counted \(observed)."
    case .extra(let expected, let observed):
      return "Extra dose: expected \(expected), counted \(observed)."
    case .wrongQuantity(let id, let expected, let observed):
      return "\(name(id)): expected \(expected), identified \(observed)."
    case .unexpectedMedication(let id, let observed):
      return "\(name(id)) identified (\(observed)) but not expected here."
    case .brokenDose(let count):
      return "Broken or damaged dose detected (\(count))."
    case .foreignObject(let count):
      return "Foreign object detected (\(count))."
    case .lowConfidenceObject(let count):
      return "\(count) object(s) detected with low confidence."
    case .objectOnBorder(let count):
      return "\(count) object(s) on the border with a neighbouring compartment."
    case .unstableAcrossFrames:
      return "Photos disagreed about this compartment."
    case .identityNotVerified:
      return "Count matches. Medication identity was not checked automatically."
    case .suspectedMedication(let id):
      return "AvaVision's brain thinks a pill here looks like \(name(id)), which is not expected."
    case .unrecognisedPill(let count):
      return "\(count) pill(s) look unlike every medication expected here."
    case .conflictingIdentity(let count):
      return "\(count) pill(s) were identified differently by the model and the brain."
    case .noExpectation:
      return "The pack profile does not define this compartment."
    case .layoutUncalibrated:
      return "Pack layout is not calibrated, so results cannot be accepted automatically."
    case .notEvaluated:
      return "Not evaluated automatically."
    }
  }
}

extension PackFinding {
  var text: String {
    switch self {
    case .insufficientUsableFrames(let usable, let required):
      return "Only \(usable) of \(required) required photos were usable."
    case .captureIssues(let issues):
      return "Photo problems: " + issues.map(\.text).joined(separator: ", ") + "."
    case .registrationIssues(let issues):
      return "Pack position problems: " + issues.map(\.text).joined(separator: ", ") + "."
    case .objectsOutsideCompartments(let count):
      return "\(count) object(s) found on the card outside the compartments."
    case .modelUnavailable:
      return "No usable detection model is installed, so nothing was checked automatically."
    case .profileLayoutMismatch:
      return "The pack profile belongs to a different pack layout."
    }
  }
}

extension CaptureIssue {
  var text: String {
    switch self {
    case .blurry: "blurry"
    case .tooDark: "too dark"
    case .tooBright: "too bright"
    case .glare: "glare on the pack"
    case .analysisFailed: "image could not be analysed"
    }
  }
}

extension RegistrationIssue {
  var text: String {
    switch self {
    case .packNotFound: "pack not found"
    case .lowDetectorConfidence: "pack outline unclear"
    case .packNotConvex: "pack outline invalid"
    case .packTooSmall: "move closer"
    case .packTouchesFrameEdge: "whole pack must be in view"
    case .perspectiveTooSteep: "hold the camera straight above the pack"
    case .aspectRatioMismatch: "pack shape does not match the layout"
    }
  }
}

extension ModelCapability {
  var title: String {
    switch self {
    case .unavailable: "Automatic checking off"
    case .countOnly: "Count only — identity locked"
    case .identity(let ids): "Count + identity (\(ids.count) medications)"
    }
  }

  var color: Color {
    switch self {
    case .unavailable: .gray
    case .countOnly: .blue
    case .identity: .green
    }
  }
}

extension GateBlocker {
  var text: String {
    func percent(_ value: Double) -> String { value.formatted(.percent.precision(.fractionLength(1))) }
    switch self {
    case .modelFileMissing: return "Model file is missing."
    case .modelHashMismatch: return "Model file does not match its manifest (SHA-256)."
    case .unsupportedManifestSchema(let version): return "Unsupported manifest version \(version)."
    case .modelRevoked: return "Model has been revoked."
    case .noPillLabels: return "Model has no pill labels."
    case .stageNotReleased(let stage): return "Model stage is \(stage.rawValue), not released."
    case .evaluationMissing: return "No holdout evaluation attached."
    case .insufficientSamples(let have, let need): return "Holdout samples: \(have) of \(need)."
    case .insufficientErrorSamples(let have, let need): return "Error samples: \(have) of \(need)."
    case .countAccuracyTooLow(let have, let need): return "Count accuracy \(percent(have)) < \(percent(need))."
    case .identityPrecisionTooLow(let have, let need):
      return "Identity precision \(percent(have)) < \(percent(need))."
    case .identityRecallTooLow(let have, let need): return "Identity recall \(percent(have)) < \(percent(need))."
    case .falseAcceptanceTooHigh(let have, let limit):
      return "False acceptance \(percent(have)) > \(percent(limit))."
    case .noMedicationLabels: return "Model does not identify any medication."
    case .medicationUndersampled(let id, let have, let need): return "\(id): \(have) of \(need) samples."
    case .medicationBelowThreshold(let id): return "\(id): precision or recall below threshold."
    }
  }
}

extension PackProfile.Issue {
  func text(catalog: MedicationCatalog, layout: PackLayout?) -> String {
    func label(_ index: CompartmentIndex) -> String { layout?.label(for: index) ?? index.description }
    switch self {
    case .emptyReference: return "Enter a pack reference."
    case .layoutMismatch(_, let found): return "Unknown pack layout '\(found)'."
    case .compartmentOutsideLayout(let index): return "Compartment \(index) is outside the layout."
    case .duplicateCompartment(let index): return "\(label(index)) is defined twice."
    case .unspecifiedCompartment(let index): return "\(label(index)) is not defined."
    case .unknownMedication(let id): return "Medication '\(id)' is not in the catalog."
    case .invalidQuantity(let index, let id):
      return "\(label(index)): quantity of \(catalog.medication(id)?.displayName ?? id.rawValue) must be at least 1."
    }
  }
}

extension SignOffError {
  func text(layout: PackLayout) -> String {
    func labels(_ indices: [CompartmentIndex]) -> String {
      indices.map(layout.label(for:)).joined(separator: ", ")
    }
    switch self {
    case .missingPharmacistIdentifier: return "Enter the pharmacist's initials."
    case .reviewForUnknownCompartment(let index): return "Review refers to unknown compartment \(index)."
    case .unreviewedCompartments(let indices): return "Inspect every flagged compartment first: \(labels(indices))."
    case .unresolvedCompartments(let indices):
      return "A pack with unresolved compartments cannot be released: \(labels(indices))."
    case .packFindingsNotAcknowledged: return "Confirm that you have checked the pack-level findings."
    }
  }
}

extension CompartmentReviewOutcome {
  var title: String {
    switch self {
    case .confirmedCorrect: "Correct"
    case .corrected: "Corrected"
    case .unresolved: "Unresolved"
    }
  }
}

extension PackProfile {
  var totalDoses: Int {
    compartments.reduce(0) { $0 + $1.totalQuantity }
  }
}

extension TrustStatus {
  var title: String {
    switch self {
    case .trusted: "Trusted"
    case .suspended: "Suspended after an error"
    case .learning(let progress): "Learning · \(Int((progress * 100).rounded(.down)))%"
    }
  }

  var color: Color {
    switch self {
    case .trusted: .green
    case .suspended: .red
    case .learning: .orange
    }
  }
}

extension CalibrationReport.Outcome {
  var text: String {
    switch self {
    case .calibrated(let accept, let margin):
      "Calibrated (accept ≥ \(accept.formatted(.number.precision(.fractionLength(3)))), "
        + "margin ≥ \(margin.formatted(.number.precision(.fractionLength(3)))))"
    case .insufficientData(let medications, let queries):
      "Needs more teaching: \(medications) medication(s) and \(queries) pills are well covered "
        + "(at least 2 medications and 30 pills, each from 2+ photos)."
    case .targetUnreachable:
      "The remembered medications look too alike to separate safely yet. Teach more varied photos."
    }
  }
}
