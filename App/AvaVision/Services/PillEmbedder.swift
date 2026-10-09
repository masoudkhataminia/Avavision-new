import AvaVisionCore
import CoreGraphics
import CoreML
import Foundation
import Vision

/// Turns a pill crop into an appearance embedding for the brain.
protocol PillEmbedder: Sendable {
  var id: EmbedderID { get }
  /// Human-readable description for the brain screen.
  var title: String { get }
  func embed(_ image: CGImage) throws -> Embedding
}

enum EmbedderError: Error {
  case noOutput
  case invalidVector
}

/// Apple's built-in image feature print (Vision, revision 2). Always available, needs no training:
/// the brain has a working sense of appearance from the first day.
final class FeaturePrintEmbedder: PillEmbedder, @unchecked Sendable {
  let id: EmbedderID = "apple-vision-featureprint-r2"
  let title = "Apple Vision feature print (built in)"

  func embed(_ image: CGImage) throws -> Embedding {
    let request = VNGenerateImageFeaturePrintRequest()
    request.revision = VNGenerateImageFeaturePrintRequestRevision2
    request.imageCropAndScaleOption = .scaleFit
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    guard let observation = request.results?.first else { throw EmbedderError.noOutput }
    let floats: [Float]
    switch observation.elementType {
    case .float:
      floats = observation.data.withUnsafeBytes {
        Array($0.bindMemory(to: Float.self).prefix(observation.elementCount))
      }
    case .double:
      floats = observation.data.withUnsafeBytes {
        $0.bindMemory(to: Double.self).prefix(observation.elementCount).map(Float.init)
      }
    default:
      throw EmbedderError.invalidVector
    }
    guard let embedding = Embedding(embedderID: id, raw: floats) else { throw EmbedderError.invalidVector }
    return embedding
  }
}

/// A bundled Core ML embedder (for example DINOv2 fine-tuned on AvaVision's confirmed pills).
final class CoreMLEmbedder: PillEmbedder, @unchecked Sendable {
  let id: EmbedderID
  let title: String
  private let model: VNCoreMLModel
  private let outputName: String

  init(manifest: EmbedderManifest, model: VNCoreMLModel) {
    id = manifest.embedderID
    title = "\(manifest.embedderID) \(manifest.version)"
    self.model = model
    outputName = manifest.outputName
  }

  func embed(_ image: CGImage) throws -> Embedding {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFit
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    let observations = request.results?.compactMap { $0 as? VNCoreMLFeatureValueObservation } ?? []
    guard
      let array = (observations.first { $0.featureName == outputName } ?? observations.first)?.featureValue
        .multiArrayValue
    else { throw EmbedderError.noOutput }
    let floats = (0..<array.count).map { Float(truncating: array[$0]) }
    guard let embedding = Embedding(embedderID: id, raw: floats) else { throw EmbedderError.invalidVector }
    return embedding
  }
}

enum EmbedderLoader {
  static let modelName = "AvaVisionEmbedder"

  /// Uses the bundled Core ML embedder when present and intact, otherwise Apple's feature print.
  static func load(bundle: Bundle = .main) -> (embedder: PillEmbedder, warning: String?) {
    guard let manifestURL = bundle.url(forResource: modelName, withExtension: "json", subdirectory: "Models")
    else { return (FeaturePrintEmbedder(), nil) }
    do {
      let manifest = try AvaVisionJSON.decoder().decode(EmbedderManifest.self, from: Data(contentsOf: manifestURL))
      guard let modelURL = bundle.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Models"),
        manifest.isIntact(computedModelSHA256: try? ContentHasher.sha256Hex(contentsOf: modelURL))
      else {
        return (FeaturePrintEmbedder(), "The bundled embedder does not match its manifest and was not used.")
      }
      let configuration = MLModelConfiguration()
      configuration.computeUnits = .all
      let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: configuration))
      return (CoreMLEmbedder(manifest: manifest, model: model), nil)
    } catch {
      return (FeaturePrintEmbedder(), "The bundled embedder could not be loaded: \(error.localizedDescription)")
    }
  }
}
