import Foundation

public struct CalibrationReport: Codable, Hashable, Sendable {
  public enum Outcome: Codable, Hashable, Sendable {
    case calibrated(acceptSimilarity: Float, minimumMargin: Float)
    /// Too little knowledge to estimate safe thresholds; the policy is left unchanged.
    case insufficientData(medications: Int, queries: Int)
    /// No threshold reached the required precision; identification stays off.
    case targetUnreachable
  }

  public var outcome: Outcome
  public var calibratedAt: Date
  /// Self-tests run: each remembered pill against the rest, and against memory without its medication.
  public var trials: Int
  /// Trials where a medication was named.
  public var identified: Int
  /// Named trials that were wrong (including naming a medication for a pill whose medication was hidden).
  public var falseIdentifications: Int
  /// Wilson 95% lower bound on precision at the chosen thresholds.
  public var precisionLowerBound: Double
  /// Fraction of known-medication trials that were named.
  public var coverage: Double

  public init(
    outcome: Outcome, calibratedAt: Date, trials: Int, identified: Int, falseIdentifications: Int,
    precisionLowerBound: Double, coverage: Double
  ) {
    self.outcome = outcome
    self.calibratedAt = calibratedAt
    self.trials = trials
    self.identified = identified
    self.falseIdentifications = falseIdentifications
    self.precisionLowerBound = precisionLowerBound
    self.coverage = coverage
  }
}

public enum Statistics {
  /// Lower bound of the Wilson score interval for a proportion (z = 1.96 → 95%).
  public static func wilsonLowerBound(successes: Int, trials: Int, z: Double = 1.96) -> Double {
    guard trials > 0 else { return 0 }
    let n = Double(trials)
    let p = Double(successes) / n
    let z2 = z * z
    let centre = p + z2 / (2 * n)
    let spread = z * ((p * (1 - p) + z2 / (4 * n)) / n).squareRoot()
    return max(0, (centre - spread) / (1 + z2 / n))
  }
}

/// Learns identification thresholds from the brain's own memory by self-testing:
/// every remembered pill is classified against the other photos (closed-set trial) and against memory
/// with its whole medication hidden (open-set trial, where naming anything is an error).
///
/// Naming a medication is only a claim: acceptance additionally requires the per-medication track
/// record in `TrustLedger`. The calibration target therefore balances useful early warnings against
/// noise, and must be reachable with a few hundred remembered pills.
public struct IdentityCalibrator: Sendable {
  public var requiredPrecisionLowerBound: Double
  public var minimumMedications: Int
  public var minimumQueries: Int
  /// Upper bound on self-test queries, to keep calibration fast on a phone.
  public var maximumQueries: Int
  public var marginGrid: [Float]

  public init(
    requiredPrecisionLowerBound: Double = 0.9, minimumMedications: Int = 2, minimumQueries: Int = 30,
    maximumQueries: Int = 600, marginGrid: [Float] = [0, 0.01, 0.02, 0.04, 0.08, 0.15]
  ) {
    self.requiredPrecisionLowerBound = requiredPrecisionLowerBound
    self.minimumMedications = minimumMedications
    self.minimumQueries = minimumQueries
    self.maximumQueries = maximumQueries
    self.marginGrid = marginGrid
  }

  private struct Trial {
    let truth: MedicationID?
    let best: IdentityCandidate?
    let margin: Float
  }

  /// Returns the report and, when calibration succeeded, the new policy (version incremented).
  public func calibrate(knowledge: KnowledgeBase, base: IdentityPolicy, at date: Date = Date())
    -> (report: CalibrationReport, policy: IdentityPolicy?)
  {
    let index = KnowledgeIndex(knowledge)
    let eligible = knowledge.medications.filter {
      knowledge.count(for: $0) >= base.minimumExemplars && knowledge.groupCount(for: $0) >= base.minimumGroups
    }
    let queries = sampled(knowledge.exemplars.filter { eligible.contains($0.medicationID) })

    func failure(_ outcome: CalibrationReport.Outcome, trials: Int) -> (CalibrationReport, IdentityPolicy?) {
      (
        CalibrationReport(
          outcome: outcome, calibratedAt: date, trials: trials, identified: 0, falseIdentifications: 0,
          precisionLowerBound: 0, coverage: 0), nil
      )
    }
    guard eligible.count >= minimumMedications, queries.count >= minimumQueries else {
      return failure(.insufficientData(medications: eligible.count, queries: queries.count), trials: 0)
    }

    var trials: [Trial] = []
    for query in queries {
      for hideMedication in [false, true] {
        let candidates = index.candidates(
          for: query.embedding, neighbours: base.neighbours, excludingGroup: query.groupID,
          excludingMedication: hideMedication ? query.medicationID : nil
        ).filter { eligible.contains($0.medicationID) || $0.medicationID == query.medicationID }
        let margin = candidates.count > 1 ? candidates[0].score - candidates[1].score : .infinity
        trials.append(Trial(truth: hideMedication ? nil : query.medicationID, best: candidates.first, margin: margin))
      }
    }

    let closedTrials = trials.filter { $0.truth != nil }.count
    let thresholds = Set(trials.compactMap { $0.best?.score }).sorted()
    var best: (threshold: Float, margin: Float, named: Int, wrong: Int, bound: Double)?
    for margin in marginGrid {
      for threshold in thresholds {
        var named = 0
        var wrong = 0
        var namedClosed = 0
        for trial in trials {
          guard let candidate = trial.best, candidate.score >= threshold, trial.margin >= margin else { continue }
          named += 1
          if candidate.medicationID != trial.truth { wrong += 1 }
          if trial.truth != nil { namedClosed += 1 }
        }
        guard named > 0 else { continue }
        let bound = Statistics.wilsonLowerBound(successes: named - wrong, trials: named)
        guard bound >= requiredPrecisionLowerBound else { continue }
        if best == nil || namedClosed > best!.named || (namedClosed == best!.named && bound > best!.bound) {
          best = (threshold, margin, namedClosed, wrong, bound)
        }
      }
    }

    guard let best else { return failure(.targetUnreachable, trials: trials.count) }
    var policy = base
    policy.acceptSimilarity = best.threshold
    policy.minimumMargin = best.margin
    policy.version = base.version + 1
    let report = CalibrationReport(
      outcome: .calibrated(acceptSimilarity: best.threshold, minimumMargin: best.margin), calibratedAt: date,
      trials: trials.count, identified: best.named, falseIdentifications: best.wrong,
      precisionLowerBound: best.bound, coverage: closedTrials == 0 ? 0 : Double(best.named) / Double(closedTrials))
    return (report, policy)
  }

  /// Deterministic, evenly spread subset so large memories stay fast to calibrate.
  private func sampled(_ exemplars: [PillExemplar]) -> [PillExemplar] {
    let sorted = exemplars.sorted { $0.id.uuidString < $1.id.uuidString }
    guard sorted.count > maximumQueries, maximumQueries > 0 else { return sorted }
    let step = Double(sorted.count) / Double(maximumQueries)
    return (0..<maximumQueries).map { sorted[Int(Double($0) * step)] }
  }
}
