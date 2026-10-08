import Foundation

/// One compartment of the holdout set: what the profile expected, what the pharmacist confirmed
/// was really there, and what the system said.
public struct EvaluationSample: Codable, Hashable, Sendable {
  public var sampleID: String
  public var expected: [MedicationID: Int]
  /// Pharmacist-confirmed true content.
  public var truth: [MedicationID: Int]
  /// Pharmacist confirmed a broken dose or foreign object.
  public var truthHasBrokenOrForeign: Bool
  public var predictedStatus: CompartmentStatus
  /// The system's consensus count, `nil` when it did not settle on one.
  public var predictedCount: Int?
  public var predictedMedications: [MedicationID: Int]

  public init(
    sampleID: String, expected: [MedicationID: Int], truth: [MedicationID: Int],
    truthHasBrokenOrForeign: Bool = false, predictedStatus: CompartmentStatus, predictedCount: Int?,
    predictedMedications: [MedicationID: Int] = [:]
  ) {
    self.sampleID = sampleID
    self.expected = expected
    self.truth = truth
    self.truthHasBrokenOrForeign = truthHasBrokenOrForeign
    self.predictedStatus = predictedStatus
    self.predictedCount = predictedCount
    self.predictedMedications = predictedMedications
  }

  var trueCount: Int { truth.values.reduce(0, +) }
  var expectedCount: Int { expected.values.reduce(0, +) }

  /// The compartment content differs from what the profile requires.
  public var isErroneous: Bool {
    truthHasBrokenOrForeign || truth.filter { $0.value > 0 } != expected.filter { $0.value > 0 }
  }

  var hasCountError: Bool { truthHasBrokenOrForeign || trueCount != expectedCount }
}

public struct MedicationMetrics: Codable, Hashable, Sendable {
  /// Samples whose true content includes the medication.
  public var samples: Int
  public var precision: Double
  public var recall: Double

  public init(samples: Int, precision: Double, recall: Double) {
    self.samples = samples
    self.precision = precision
    self.recall = recall
  }
}

/// Holdout results embedded in a model manifest and checked by the release gate.
public struct EvaluationReport: Codable, Hashable, Sendable {
  public var datasetID: String
  public var totalSamples: Int
  /// Samples whose true content is wrong in some way.
  public var errorSamples: Int
  /// Among samples with a consensus count, the fraction where it equals the true count.
  public var countAccuracy: Double
  public var identityPrecision: Double
  public var identityRecall: Double
  /// Erroneous samples the system marked `verified`, over all erroneous samples.
  public var falseAcceptanceRate: Double
  /// Samples with a wrong count marked `verified` or `countMatched`, over samples with a wrong count.
  public var countFalseAcceptanceRate: Double
  /// Fraction of samples sent to pharmacist review.
  public var reviewRate: Double
  public var perMedication: [MedicationID: MedicationMetrics]

  public init(
    datasetID: String, totalSamples: Int, errorSamples: Int, countAccuracy: Double,
    identityPrecision: Double, identityRecall: Double, falseAcceptanceRate: Double,
    countFalseAcceptanceRate: Double, reviewRate: Double, perMedication: [MedicationID: MedicationMetrics]
  ) {
    self.datasetID = datasetID
    self.totalSamples = totalSamples
    self.errorSamples = errorSamples
    self.countAccuracy = countAccuracy
    self.identityPrecision = identityPrecision
    self.identityRecall = identityRecall
    self.falseAcceptanceRate = falseAcceptanceRate
    self.countFalseAcceptanceRate = countFalseAcceptanceRate
    self.reviewRate = reviewRate
    self.perMedication = perMedication
  }
}

public enum Evaluator {
  public static func evaluate(_ samples: [EvaluationSample], datasetID: String) -> EvaluationReport {
    func ratio(_ numerator: Int, _ denominator: Int) -> Double {
      denominator == 0 ? 0 : Double(numerator) / Double(denominator)
    }
    func ratio(_ numerator: Int, _ denominator: Int, emptyValue: Double) -> Double {
      denominator == 0 ? emptyValue : Double(numerator) / Double(denominator)
    }

    let counted = samples.filter { $0.predictedCount != nil }
    let correctCounts = counted.filter { $0.predictedCount == $0.trueCount }.count

    let erroneous = samples.filter(\.isErroneous)
    let falselyVerified = erroneous.filter { $0.predictedStatus == .verified }.count

    let countErrors = samples.filter(\.hasCountError)
    let countAccepted = countErrors.filter { $0.predictedStatus == .verified || $0.predictedStatus == .countMatched }
      .count

    var truePositives = 0
    var falsePositives = 0
    var falseNegatives = 0
    var perMedicationTallies: [MedicationID: (samples: Int, tp: Int, fp: Int, fn: Int)] = [:]
    for sample in samples {
      let ids = Set(sample.truth.keys).union(sample.predictedMedications.keys)
      for id in ids {
        let truth = sample.truth[id] ?? 0
        let predicted = sample.predictedMedications[id] ?? 0
        let tp = min(truth, predicted)
        let fp = max(0, predicted - truth)
        let fn = max(0, truth - predicted)
        truePositives += tp
        falsePositives += fp
        falseNegatives += fn
        var tally = perMedicationTallies[id] ?? (0, 0, 0, 0)
        if truth > 0 { tally.samples += 1 }
        tally.tp += tp
        tally.fp += fp
        tally.fn += fn
        perMedicationTallies[id] = tally
      }
    }

    let perMedication = perMedicationTallies.mapValues {
      MedicationMetrics(
        samples: $0.samples,
        precision: ratio($0.tp, $0.tp + $0.fp),
        recall: ratio($0.tp, $0.tp + $0.fn)
      )
    }

    return EvaluationReport(
      datasetID: datasetID,
      totalSamples: samples.count,
      errorSamples: erroneous.count,
      countAccuracy: ratio(correctCounts, counted.count),
      identityPrecision: ratio(truePositives, truePositives + falsePositives),
      identityRecall: ratio(truePositives, truePositives + falseNegatives),
      falseAcceptanceRate: ratio(falselyVerified, erroneous.count, emptyValue: 1),
      countFalseAcceptanceRate: ratio(countAccepted, countErrors.count, emptyValue: 1),
      reviewRate: ratio(samples.filter { $0.predictedStatus == .needsReview }.count, samples.count),
      perMedication: perMedication
    )
  }
}
