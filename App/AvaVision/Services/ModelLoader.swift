import AvaVisionCore
import CoreML
import Foundation
import Vision

/// The object detector as far as the app is concerned.
enum ModelState: @unchecked Sendable {
  case loading
  /// The bundled model's manifest could not be read or the model could not be loaded.
  case failed(String)
  /// The release gate has decided the capability; `detector` is set only if it may run.
  case ready(ActiveModel, ObjectDetector?)

  var activeModel: ActiveModel? {
    if case .ready(let model, _) = self { return model }
    return nil
  }

  var detector: ObjectDetector? {
    if case .ready(_, let detector) = self { return detector }
    return nil
  }

  var capability: ModelCapability {
    activeModel?.capability ?? .unavailable
  }
}

/// Loads `Models/AvaVisionDetector.mlmodelc` and its manifest `Models/AvaVisionDetector.json` from the
/// app bundle and runs them through the release gate. Without a bundled detector the app uses the
/// training-free pocket segmenter, so it can count from the first day.
enum ModelLoader {
  static let modelName = "AvaVisionDetector"

  static func load(bundle: Bundle = .main) -> ModelState {
    guard let manifestURL = bundle.url(forResource: modelName, withExtension: "json", subdirectory: "Models")
    else { return .ready(.builtInPocketSegmenter, PocketSegmenter()) }
    do {
      let manifest = try AvaVisionJSON.decoder().decode(ModelManifest.self, from: Data(contentsOf: manifestURL))
      let modelURL = bundle.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Models")
      let hash = modelURL.flatMap { try? ContentHasher.sha256Hex(contentsOf: $0) }
      let decision = ReleaseGate().evaluate(manifest, computedModelSHA256: hash)
      let active = ActiveModel(manifest: manifest, decision: decision)
      // A bundled model that fails its integrity check is never silently replaced by another detector.
      guard decision.capability.canCount, let modelURL else { return .ready(active, nil) }

      let configuration = MLModelConfiguration()
      configuration.computeUnits = .all
      let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: configuration))
      return .ready(active, CoreMLObjectDetector(model: model))
    } catch {
      return .failed(error.localizedDescription)
    }
  }
}
