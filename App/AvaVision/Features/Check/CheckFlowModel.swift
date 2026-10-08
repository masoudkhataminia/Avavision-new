import AvaVisionCore
import CoreGraphics
import Foundation
import Observation

/// Drives one pack check: live camera → capture → analysis → sign-off → audit.
@MainActor
@Observable
final class CheckFlowModel {
  enum Stage: Equatable {
    case live
    case collecting
    case analyzing
    case result
    case signOff
    case completed
  }

  /// Upper bound on frames per capture attempt before analysing anyway (which then asks for a retake).
  static let maximumFramesPerAttempt = 12

  let layout: PackLayout
  let model: ActiveModel?
  private(set) var session: CheckSession
  private(set) var stage: Stage = .live
  private(set) var cameraState: CameraController.State = .idle
  /// Most recent analysed frame, for live guidance.
  private(set) var latestFrame: AnalyzedFrame?
  /// Most recent usable frame of the analysed attempt, shown as evidence.
  private(set) var evidenceFrame: AnalyzedFrame?
  private(set) var savedRecord: CheckRecord?
  var errorMessage: String?

  @ObservationIgnored let camera = CameraController()
  @ObservationIgnored private let analyzer: FrameAnalyzer
  @ObservationIgnored private let engine: VerificationEngine
  @ObservationIgnored private var framesThisAttempt = 0
  @ObservationIgnored private var pendingRecord: CheckRecord?

  init(session: CheckSession, analyzer: FrameAnalyzer, engine: VerificationEngine, model: ActiveModel?) {
    self.session = session
    self.layout = session.layout
    self.analyzer = analyzer
    self.engine = engine
    self.model = model
  }

  var result: PackVerificationResult? { session.result }
  var requiredFrames: Int { max(1, engine.policy.requiredConsistentFrames) }
  var usableFrames: Int { session.usableFrameCount }

  // MARK: Camera

  func startCamera() {
    camera.setAnalyzer(analyzer)
    camera.onFrame = { [weak self] frame in self?.receive(frame) }
    camera.onStateChange = { [weak self] state in self?.cameraState = state }
    camera.start()
  }

  func stopCamera() {
    camera.stop()
  }

  func startCapture() {
    guard stage == .live else { return }
    framesThisAttempt = 0
    stage = .collecting
  }

  private func receive(_ frame: AnalyzedFrame) {
    latestFrame = frame
    guard stage == .collecting else { return }
    record(frame)
    framesThisAttempt += 1
    if session.isReadyToAnalyze(policy: engine.policy) || framesThisAttempt >= Self.maximumFramesPerAttempt {
      analyze()
    }
  }

  // MARK: Photos

  /// Analyses still photos of the pack (for example taken on the station or in the Simulator).
  /// Each photo is one frame; the same agreement rules apply, so take at least three.
  func analyzePhotos(_ images: [CGImage]) async {
    guard stage == .live, !images.isEmpty else { return }
    stage = .analyzing
    let analyzer = self.analyzer
    let frames = await Task.detached(priority: .userInitiated) {
      images.map { analyzer.analyze($0) }
    }.value
    for frame in frames { record(frame) }
    latestFrame = frames.last
    analyze()
  }

  // MARK: Analysis

  private func record(_ frame: AnalyzedFrame) {
    do {
      try session.record(frame.observation)
      if frame.observation.isUsable { evidenceFrame = frame }
    } catch {
      errorMessage = "Frame could not be recorded."
    }
  }

  private func analyze() {
    stage = .analyzing
    do {
      try session.analyze(with: engine)
      stage = .result
      camera.stop()
    } catch {
      errorMessage = "Analysis failed. Please try again."
      stage = .live
    }
  }

  func retake() {
    do {
      try session.retake()
      evidenceFrame = nil
      latestFrame = nil
      stage = .live
      camera.start()
    } catch {
      errorMessage = "Cannot retake at this point."
    }
  }

  func showSignOff() {
    guard stage == .result else { return }
    stage = .signOff
  }

  func showResult() {
    guard stage == .signOff else { return }
    stage = .result
  }

  // MARK: Sign-off

  /// Validates the sign-off, then appends the record to the audit trail. If saving fails the
  /// same record can be saved again; the decision itself cannot be changed once accepted.
  func complete(with signOff: PharmacistSignOff, app: AppModel) async {
    do {
      if pendingRecord == nil {
        pendingRecord = try session.complete(
          with: signOff, appVersion: app.appVersion, deviceIdentifier: app.deviceIdentifier,
          model: model.map(ModelSummary.init))
      }
      guard let record = pendingRecord else { return }
      try await app.appendToAudit(record)
      savedRecord = record
      stage = .completed
    } catch CheckSession.SessionError.signOffRejected(let error) {
      errorMessage = error.text(layout: layout)
    } catch {
      errorMessage =
        "The audit record could not be saved: \(error.localizedDescription). Tap the button again to retry."
    }
  }

  var isAwaitingSaveRetry: Bool { pendingRecord != nil && savedRecord == nil }
}
