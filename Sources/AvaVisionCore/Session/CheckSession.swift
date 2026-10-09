import Foundation

/// The lifecycle of checking one pack: capture → analysis → pharmacist sign-off → audit record.
public struct CheckSession: Sendable, Identifiable {
  public enum Phase: String, Hashable, Sendable {
    case capturing
    case analyzed
    case completed
  }

  public enum SessionError: Error, Hashable, Sendable {
    case invalidProfile([PackProfile.Issue])
    case invalidLayout([PackLayout.ValidationError])
    case wrongPhase(Phase)
    case signOffRejected(SignOffError)
  }

  /// Frames kept in memory; only the most recent usable ones are ever evaluated.
  public static let maximumRetainedFrames = 12

  public let id: UUID
  public let startedAt: Date
  public let layout: PackLayout
  public let profile: PackProfile
  public private(set) var phase: Phase = .capturing
  public private(set) var frames: [FrameObservation] = []
  public private(set) var result: PackVerificationResult?
  public private(set) var signOff: PharmacistSignOff?

  /// Starts a session only for a valid layout and a complete, valid profile.
  public init(
    id: UUID = UUID(), layout: PackLayout, profile: PackProfile, catalog: MedicationCatalog,
    startedAt: Date = Date()
  ) throws {
    let layoutErrors = layout.validationErrors()
    guard layoutErrors.isEmpty else { throw SessionError.invalidLayout(layoutErrors) }
    let issues = profile.issues(layout: layout, catalog: catalog)
    guard issues.isEmpty else { throw SessionError.invalidProfile(issues) }
    self.id = id
    self.layout = layout
    self.profile = profile
    self.startedAt = startedAt
  }

  public var usableFrameCount: Int { frames.filter(\.isUsable).count }

  public func isReadyToAnalyze(policy: DecisionPolicy = .standard) -> Bool {
    usableFrameCount >= max(1, policy.requiredConsistentFrames)
  }

  public mutating func record(_ frame: FrameObservation) throws {
    guard phase == .capturing else { throw SessionError.wrongPhase(phase) }
    frames.append(frame)
    if frames.count > Self.maximumRetainedFrames {
      frames.removeFirst(frames.count - Self.maximumRetainedFrames)
    }
  }

  @discardableResult
  public mutating func analyze(with engine: VerificationEngine, at date: Date = Date()) throws
    -> PackVerificationResult
  {
    guard phase == .capturing else { throw SessionError.wrongPhase(phase) }
    let result = engine.evaluate(profile: profile, frames: frames, evaluatedAt: date)
    self.result = result
    phase = .analyzed
    return result
  }

  /// Discards the analysis and captured frames to capture the pack again.
  public mutating func retake() throws {
    guard phase == .analyzed else { throw SessionError.wrongPhase(phase) }
    frames.removeAll()
    result = nil
    phase = .capturing
  }

  /// Validates the pharmacist's decision and produces the record for the audit trail.
  public mutating func complete(
    with signOff: PharmacistSignOff, appVersion: String, deviceIdentifier: String, model: ModelSummary?,
    brain: BrainSummary? = nil, at date: Date = Date()
  ) throws -> CheckRecord {
    guard phase == .analyzed, let result else { throw SessionError.wrongPhase(phase) }
    do {
      try SignOffValidator.validate(signOff, for: result)
    } catch let error as SignOffError {
      throw SessionError.signOffRejected(error)
    }
    self.signOff = signOff
    phase = .completed
    return CheckRecord(
      createdAt: date,
      appVersion: appVersion,
      deviceIdentifier: deviceIdentifier,
      layout: layout,
      profile: profile,
      model: model,
      brain: brain,
      result: result,
      signOff: signOff,
      frameImageSHA256s: frames.compactMap(\.imageSHA256)
    )
  }
}
