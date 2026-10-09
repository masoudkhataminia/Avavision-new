import Foundation

/// Describes a bundled appearance-embedding model (for example a fine-tuned DINOv2 exported to Core ML).
///
/// The embedder only produces vectors; it never decides anything on its own, so it needs an
/// integrity check rather than the release gate. Trust is earned per medication by the brain.
public struct EmbedderManifest: Codable, Hashable, Sendable {
  public static let currentSchemaVersion = 1

  public var schemaVersion: Int
  public var embedderID: EmbedderID
  public var version: String
  /// SHA-256 of the compiled model (see `ContentHasher`).
  public var modelSHA256: String
  /// Square input size in pixels.
  public var inputSize: Int
  /// Name of the model output holding the embedding.
  public var outputName: String
  /// Licence of the base weights, recorded for due diligence (must be commercially usable).
  public var baseModelLicense: String
  public var notes: String?

  public init(
    schemaVersion: Int = EmbedderManifest.currentSchemaVersion, embedderID: EmbedderID, version: String,
    modelSHA256: String, inputSize: Int, outputName: String, baseModelLicense: String, notes: String? = nil
  ) {
    self.schemaVersion = schemaVersion
    self.embedderID = embedderID
    self.version = version
    self.modelSHA256 = modelSHA256
    self.inputSize = inputSize
    self.outputName = outputName
    self.baseModelLicense = baseModelLicense
    self.notes = notes
  }

  /// The model may be used only if the file on disk is exactly the one described.
  public func isIntact(computedModelSHA256: String?) -> Bool {
    guard schemaVersion == Self.currentSchemaVersion, inputSize > 0, let computedModelSHA256 else { return false }
    return computedModelSHA256.lowercased() == modelSHA256.lowercased()
  }
}

extension ActiveModel {
  /// The training-free pocket segmenter built into the app: counts only, never identifies.
  public static let builtInPocketSegmenter: ActiveModel = {
    let manifest = ModelManifest(
      modelID: "avavision-pocket-segmenter", version: "1", stage: .development, modelSHA256: "built-in",
      labels: ["pill": .pill], notes: "Apple Vision instance masks cross-checked with classical segmentation")
    return ActiveModel(manifest: manifest, decision: ReleaseGate().evaluate(manifest, computedModelSHA256: "built-in"))
  }()
}
