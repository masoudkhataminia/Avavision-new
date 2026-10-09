import AvaVisionCore
import Foundation

/// Command-line tools for the model and audit pipeline (runs on macOS and Linux).
let usage = """
  Usage:
    avavision hash <file-or-directory>
        Prints the SHA-256 used in model manifests.
    avavision evaluate <samples.json> <dataset-id>
        Builds an evaluation report from holdout samples (JSON array of EvaluationSample).
    avavision gate <manifest.json> <model-path>
        Shows what the release gate allows for this model.
    avavision verify-audit <audit.jsonl>
        Verifies an exported audit chain.
    avavision embedder-manifest <model.mlmodelc> <embedder-id> <version> <base-licence>
        Prints the manifest that lets the app load a Core ML embedder.
    avavision brain-report <brain.json>
        Recalibrates an exported brain and prints what it knows and trusts.
  """

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data((message + "\n").utf8))
  exit(1)
}

func printJSON<T: Encodable>(_ value: T) throws {
  let data = try AvaVisionJSON.encoder(pretty: true).encode(value)
  print(String(decoding: data, as: UTF8.self))
}

let arguments = Array(CommandLine.arguments.dropFirst())

do {
  switch (arguments.first, arguments.count) {
  case ("hash", 2):
    print(try ContentHasher.sha256Hex(contentsOf: URL(fileURLWithPath: arguments[1])))

  case ("evaluate", 3):
    let data = try Data(contentsOf: URL(fileURLWithPath: arguments[1]))
    let samples = try AvaVisionJSON.decoder().decode([EvaluationSample].self, from: data)
    try printJSON(Evaluator.evaluate(samples, datasetID: arguments[2]))

  case ("gate", 3):
    let data = try Data(contentsOf: URL(fileURLWithPath: arguments[1]))
    let manifest = try AvaVisionJSON.decoder().decode(ModelManifest.self, from: data)
    let hash = try? ContentHasher.sha256Hex(contentsOf: URL(fileURLWithPath: arguments[2]))
    try printJSON(ReleaseGate().evaluate(manifest, computedModelSHA256: hash))

  case ("verify-audit", 2):
    let store = AuditLogStore(fileURL: URL(fileURLWithPath: arguments[1]))
    switch try await store.verify() {
    case .intact(let entries):
      print("Audit chain intact: \(entries) entries")
    case .broken(let sequence, let defect):
      fail("Audit chain BROKEN at entry \(sequence): \(defect.rawValue)")
    }

  case ("embedder-manifest", 5):
    let manifest = EmbedderManifest(
      embedderID: EmbedderID(arguments[2]), version: arguments[3],
      modelSHA256: try ContentHasher.sha256Hex(contentsOf: URL(fileURLWithPath: arguments[1])), inputSize: 224,
      outputName: "embedding", baseModelLicense: arguments[4])
    try printJSON(manifest)

  case ("brain-report", 2):
    let data = try Data(contentsOf: URL(fileURLWithPath: arguments[1]))
    var brain = try AvaVisionJSON.decoder().decode(BrainState.self, from: data)
    let report = brain.recalibrate()
    print("Embedder: \(brain.embedderID)")
    print("Pills remembered: \(brain.knowledge.exemplars.count), labelling queue: \(brain.labellingQueue.count)")
    print("Calibration: \(report.outcome)")
    print(
      "  self-test trials \(report.trials), named \(report.identified), wrong \(report.falseIdentifications), "
        + "precision lower bound \(report.precisionLowerBound), coverage \(report.coverage)")
    print("Medication\tpills\tphotos\tcorrect\twrong\tunsure\tstatus")
    for id in brain.knowledge.medications.union(brain.ledger.records.keys).sorted() {
      let record = brain.ledger.record(for: id)
      print(
        "\(id)\t\(brain.knowledge.count(for: id))\t\(brain.knowledge.groupCount(for: id))\t\(record.correct)\t"
          + "\(record.falseIdentifications)\t\(record.abstentions)\t\(brain.trustStatus(of: id))")
    }

  default:
    fail(usage)
  }
} catch {
  fail("Error: \(error)")
}
