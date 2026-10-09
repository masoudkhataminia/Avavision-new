import AVFoundation
import CoreImage
import Foundation

/// Runs the back camera and analyses a few frames per second on a background queue.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  enum State: Equatable {
    case idle
    case running
    case unauthorized
    case unavailable
    case failed(String)
  }

  let session = AVCaptureSession()
  /// Called on the main queue.
  var onFrame: ((AnalyzedFrame) -> Void)?
  /// Called on the main queue.
  var onStateChange: ((State) -> Void)?

  private let sessionQueue = DispatchQueue(label: "avavision.camera.session")
  private let videoQueue = DispatchQueue(label: "avavision.camera.video")
  private let lock = NSLock()
  private var analyzer: FrameAnalyzer?
  private var isConfigured = false
  private var lastAnalysis = Date.distantPast
  private let analysisInterval: TimeInterval = 0.3

  func setAnalyzer(_ analyzer: FrameAnalyzer) {
    lock.lock()
    self.analyzer = analyzer
    lock.unlock()
  }

  private func currentAnalyzer() -> FrameAnalyzer? {
    lock.lock()
    defer { lock.unlock() }
    return analyzer
  }

  func start() {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      sessionQueue.async { self.configureAndRun() }
    case .notDetermined:
      AVCaptureDevice.requestAccess(for: .video) { granted in
        if granted {
          self.sessionQueue.async { self.configureAndRun() }
        } else {
          self.report(.unauthorized)
        }
      }
    default:
      report(.unauthorized)
    }
  }

  func stop() {
    sessionQueue.async {
      if self.session.isRunning { self.session.stopRunning() }
    }
  }

  private func report(_ state: State) {
    DispatchQueue.main.async { self.onStateChange?(state) }
  }

  private func configureAndRun() {
    if !isConfigured {
      guard
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
        let input = try? AVCaptureDeviceInput(device: device)
      else {
        report(.unavailable)
        return
      }
      session.beginConfiguration()
      if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
      let output = AVCaptureVideoDataOutput()
      output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
      output.alwaysDiscardsLateVideoFrames = true
      output.setSampleBufferDelegate(self, queue: videoQueue)
      guard session.canAddInput(input), session.canAddOutput(output) else {
        session.commitConfiguration()
        report(.failed("The camera could not be configured."))
        return
      }
      session.addInput(input)
      session.addOutput(output)
      if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
        connection.videoRotationAngle = 90
      }
      session.commitConfiguration()

      if (try? device.lockForConfiguration()) != nil {
        if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
        if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
        device.unlockForConfiguration()
      }
      isConfigured = true
    }
    if !session.isRunning { session.startRunning() }
    report(.running)
  }

  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection
  ) {
    let now = Date()
    guard now.timeIntervalSince(lastAnalysis) >= analysisInterval,
      let analyzer = currentAnalyzer(),
      let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
      let image = ImageConversion.render(CIImage(cvPixelBuffer: pixelBuffer), maxDimension: 1920)
    else { return }
    lastAnalysis = now
    let frame = analyzer.locate(image, capturedAt: now)
    DispatchQueue.main.async { self.onFrame?(frame) }
  }
}
