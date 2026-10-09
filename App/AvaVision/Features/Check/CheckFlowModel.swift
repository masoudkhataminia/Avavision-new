import AvaVisionCore
import CoreGraphics
import Foundation
import Observation

/// Drives one pack check: live camera → capture → detection and brain → result → sign-off → audit → learning.
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
  /// Most recent located frame, for live guidance.
  private(set) var latestFrame: AnalyzedFrame?
  /// Last evaluated frame, shown as evidence.
  private(set) var evidenceFrame: AnalyzedFrame?
  /// Usable frames collected in the current attempt.
  private(set) var usableFrames = 0
  private(set) var savedRecord: CheckRecord?
  private(set) var learningNote: String?
  var errorMessage: String?

  @ObservationIgnored let camera = CameraController()
  @ObservationIgnored private let analyzer: FrameAnalyzer
  @ObservationIgnored private let engine: VerificationEngine
  @ObservationIgnored private let identifier: PillIdentifier?
  @ObservationIgnored private let brain: BrainSummary?
  @ObservationIgnored private var captured: [AnalyzedFrame] = []
  /// Pills of the evidence frame with their crops; the brain learns from them after sign-off.
  @ObservationIgnored private var evidencePills: [IdentifiedPill] = []
  @ObservationIgnored private var pendingRecord: CheckRecord?
  @ObservationIgnored private var isEvaluating = false

  init(
    session: CheckSession, analyzer: FrameAnalyzer, engine: VerificationEngine, model: ActiveModel?,
    identifier: PillIdentifier?, brain: BrainSummary?
  ) {
    self.session = session
    self.layout = session.layout
    self.analyzer = analyzer
    self.engine = engine
    self.model = model
    self.identifier = identifier
    self.brain = brain
  }

  var result: PackVerificationResult? { session.result }
  var requiredFrames: Int { max(1, engine.policy.requiredConsistentFrames) }

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
    captured.removeAll()
    usableFrames = 0
    stage = .collecting
  }

  private func receive(_ frame: AnalyzedFrame) {
    latestFrame = frame
    guard stage == .collecting else { return }
    captured.append(frame)
    usableFrames = captured.filter(\.observation.isUsable).count
    if usableFrames >= requiredFrames || captured.count >= Self.maximumFramesPerAttempt {
      stage = .analyzing
      camera.stop()
      Task { await evaluateCaptured() }
    }
  }

  // MARK: Photos

  /// Analyses still photos of the pack (for example taken on the station or in the Simulator).
  /// Each photo is one frame; the same agreement rules apply, so take at least three.
  func analyzePhotos(_ images: [CGImage]) async {
    guard stage == .live, !images.isEmpty else { return }
    stage = .analyzing
    let analyzer = self.analyzer
    captured = await Task.detached(priority: .userInitiated) {
      images.map { analyzer.locate($0) }
    }.value
    latestFrame = captured.last
    await evaluateCaptured()
  }

  // MARK: Analysis

  /// Detects objects and asks the brain on the frames that will be evaluated, then runs the engine.
  private func evaluateCaptured() async {
    guard stage == .analyzing, !isEvaluating else { return }
    isEvaluating = true
    defer { isEvaluating = false }
    let frames = captured
    let windowIndices = Set(frames.indices.filter { frames[$0].observation.isUsable }.suffix(requiredFrames))
    let analyzer = self.analyzer
    let identifier = self.identifier
    let layout = self.layout
    let model = self.model
    let minimumConfidence = engine.policy.minimumDetectionConfidence

    let (processed, pills) = await Task.detached(priority: .userInitiated) {
      var processed: [AnalyzedFrame] = []
      var pills: [IdentifiedPill] = []
      for (index, frame) in frames.enumerated() {
        guard windowIndices.contains(index) else {
          processed.append(frame)
          continue
        }
        var detected = analyzer.detectObjects(in: frame)
        if let identifier, let model {
          let identified = identifier.identify(
            detected.observation, image: detected.image, layout: layout, model: model,
            minimumConfidence: minimumConfidence)
          detected.observation = identified.observation
          pills = identified.pills
        }
        processed.append(detected)
      }
      return (processed, pills)
    }.value

    do {
      for frame in processed { try session.record(frame.observation) }
      evidenceFrame = windowIndices.max().map { processed[$0] }
      evidencePills = pills
      try session.analyze(with: engine)
      stage = .result
    } catch {
      errorMessage = "Analysis failed. Please try again."
      resetCapture()
    }
  }

  private func resetCapture() {
    captured.removeAll()
    evidencePills.removeAll()
    evidenceFrame = nil
    latestFrame = nil
    usableFrames = 0
    stage = .live
    camera.start()
  }

  func retake() {
    do {
      try session.retake()
      resetCapture()
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

  /// Validates the sign-off, appends the record to the audit trail, then lets the brain learn from the
  /// compartments the pharmacist confirmed. If saving fails the same record can be saved again; the
  /// decision itself cannot be changed once accepted.
  func complete(with signOff: PharmacistSignOff, app: AppModel) async {
    do {
      if pendingRecord == nil {
        pendingRecord = try session.complete(
          with: signOff, appVersion: app.appVersion, deviceIdentifier: app.deviceIdentifier,
          model: model.map(ModelSummary.init), brain: brain)
      }
      guard let record = pendingRecord else { return }
      try await app.appendToAudit(record)
      savedRecord = record
      stage = .completed
      let before = app.brain?.knowledge.exemplars.count ?? 0
      await app.learn(from: record, pills: evidencePills)
      let learned = (app.brain?.knowledge.exemplars.count ?? 0) - before
      if learned > 0 { learningNote = "The brain learned \(learned) new pill images from this check." }
    } catch CheckSession.SessionError.signOffRejected(let error) {
      errorMessage = error.text(layout: layout)
    } catch {
      errorMessage =
        "The audit record could not be saved: \(error.localizedDescription). Tap the button again to retry."
    }
  }

  var isAwaitingSaveRetry: Bool { pendingRecord != nil && savedRecord == nil }
}
