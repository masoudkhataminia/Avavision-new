import Foundation

/// What a model output label means to the verification engine.
///
/// Encoded as a string: `"pill"`, `"broken"`, `"foreign"`, `"ignore"` or `"medication:<id>"`.
public enum LabelMeaning: Hashable, Sendable, Codable, CustomStringConvertible {
  /// A whole solid dose whose identity the model does not claim.
  case pill
  /// A whole solid dose the model identifies as a specific medication.
  case medication(MedicationID)
  /// A fragment or damaged dose.
  case broken
  /// An object that is not a dose (packaging debris, hair, desiccant…).
  case foreign
  /// A label the engine must not use (for example a model's background class).
  case ignore

  private static let medicationPrefix = "medication:"

  public init?(string: String) {
    switch string {
    case "pill": self = .pill
    case "broken": self = .broken
    case "foreign": self = .foreign
    case "ignore": self = .ignore
    default:
      guard string.hasPrefix(Self.medicationPrefix) else { return nil }
      let id = String(string.dropFirst(Self.medicationPrefix.count))
      guard !id.isEmpty else { return nil }
      self = .medication(MedicationID(id))
    }
  }

  public var description: String {
    switch self {
    case .pill: "pill"
    case .medication(let id): Self.medicationPrefix + id.rawValue
    case .broken: "broken"
    case .foreign: "foreign"
    case .ignore: "ignore"
    }
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let string = try container.decode(String.self)
    guard let meaning = LabelMeaning(string: string) else {
      throw DecodingError.dataCorruptedError(
        in: container, debugDescription: "Unknown label meaning '\(string)'")
    }
    self = meaning
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(description)
  }
}

/// One object reported by the detection model.
public struct Detection: Codable, Hashable, Sendable {
  public var label: String
  public var confidence: Double
  /// Bounding box in normalized image coordinates (origin top-left).
  public var boundingBox: Rect2D
  /// The brain's opinion of what this object is, when it was asked.
  public var identity: IdentityEvidence?

  public init(label: String, confidence: Double, boundingBox: Rect2D, identity: IdentityEvidence? = nil) {
    self.label = label
    self.confidence = confidence
    self.boundingBox = boundingBox
    self.identity = identity
  }
}

/// Everything measured on one camera frame.
public struct FrameObservation: Codable, Hashable, Sendable {
  public var capturedAt: Date
  public var quality: CaptureQualityAssessment
  public var registration: RegistrationOutcome
  public var detections: [Detection]
  /// Hash of the encoded image, so the audit trail can prove which image was analysed.
  public var imageSHA256: String?

  public init(
    capturedAt: Date, quality: CaptureQualityAssessment, registration: RegistrationOutcome,
    detections: [Detection], imageSHA256: String? = nil
  ) {
    self.capturedAt = capturedAt
    self.quality = quality
    self.registration = registration
    self.detections = detections
    self.imageSHA256 = imageSHA256
  }

  /// A frame is evidence only if the image is good and the pack was located.
  public var isUsable: Bool { quality.isAcceptable && registration.registration != nil }
}
