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

  default:
    fail(usage)
  }
} catch {
  fail("Error: \(error)")
}
