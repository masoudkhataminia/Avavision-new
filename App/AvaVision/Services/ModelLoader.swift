import AvaVisionCore
import CoreML
import Foundation
import Vision

/// The detection model as far as the app is concerned.
/// `VNCoreMLModel` is immutable after loading and safe to share between threads.
enum ModelState: @unchecked Sendable {
  case loading
  /// No model is bundled: automatic checking is disabled, manual checking still works.
  case notInstalled
  /// The manifest could not be read or the model could not be loaded.
  case failed(String)
  /// The release gate has decided the capability; `visionModel` is set only if it may run.
  case ready(ActiveModel, VNCoreMLModel?)

  var activeModel: ActiveModel? {
    if case .ready(let model, _) = self { return model }
    return nil
  }

  var visionModel: VNCoreMLModel? {
    if case .ready(_, let model) = self { return model }
    return nil
  }

  var capability: ModelCapability {
    activeModel?.capability ?? .unavailable
  }
}

/// Loads `Models/AvaVisionDetector.mlmodelc` and its manifest `Models/AvaVisionDetector.json`
/// from the app bundle and runs them through the release gate.
enum ModelLoader {
  static let modelName = "AvaVisionDetector"

  static func load(bundle: Bundle = .main) -> ModelState {
    guard let manifestURL = bundle.url(forResource: modelName, withExtension: "json", subdirectory: "Models")
    else { return .notInstalled }
    do {
      let manifest = try AvaVisionJSON.decoder().decode(ModelManifest.self, from: Data(contentsOf: manifestURL))
      let modelURL = bundle.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Models")
      let hash = modelURL.flatMap { try? ContentHasher.sha256Hex(contentsOf: $0) }
      let decision = ReleaseGate().evaluate(manifest, computedModelSHA256: hash)
      let active = ActiveModel(manifest: manifest, decision: decision)
      guard decision.capability.canCount, let modelURL else { return .ready(active, nil) }

      let configuration = MLModelConfiguration()
      configuration.computeUnits = .all
      let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: configuration))
      return .ready(active, model)
    } catch {
      return .failed(error.localizedDescription)
    }
  }
}
