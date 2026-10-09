import Foundation

/// How the brain turns similarities into a decision. Thresholds depend on the embedder, so they are
/// learned by `IdentityCalibrator` from the brain's own memory; an uncalibrated policy never identifies.
public struct IdentityPolicy: Codable, Hashable, Sendable {
  /// Number of nearest exemplars averaged per medication.
  public var neighbours: Int
  /// A medication needs this many exemplars, from at least `minimumGroups` photos/checks, to be named.
  public var minimumExemplars: Int
  public var minimumGroups: Int
  /// Minimum score to name a medication; `nil` until calibrated.
  public var acceptSimilarity: Float?
  /// Minimum lead of the best medication over the runner-up.
  public var minimumMargin: Float
  /// Increments whenever thresholds change.
  public var version: Int

  public init(
    neighbours: Int, minimumExemplars: Int, minimumGroups: Int, acceptSimilarity: Float?, minimumMargin: Float,
    version: Int
  ) {
    self.neighbours = neighbours
    self.minimumExemplars = minimumExemplars
    self.minimumGroups = minimumGroups
    self.acceptSimilarity = acceptSimilarity
    self.minimumMargin = minimumMargin
    self.version = version
  }

  public static let uncalibrated = IdentityPolicy(
    neighbours: 5, minimumExemplars: 5, minimumGroups: 2, acceptSimilarity: nil, minimumMargin: 0, version: 0)

  public var isCalibrated: Bool { acceptSimilarity != nil }
}

public struct IdentityCandidate: Codable, Hashable, Sendable {
  public var medicationID: MedicationID
  /// Mean similarity of the nearest exemplars.
  public var score: Float
  /// Exemplars of this medication in memory.
  public var support: Int

  public init(medicationID: MedicationID, score: Float, support: Int) {
    self.medicationID = medicationID
    self.score = score
    self.support = support
  }
}

public enum IdentityDecision: Codable, Hashable, Sendable {
  /// Confidently looks like this medication (a claim, not yet a trusted fact).
  case identified(MedicationID)
  /// Two medications are too close to call.
  case ambiguous
  /// Unlike every medication the brain knows well: possibly a pill it has never seen.
  case unrecognised
  /// The brain does not know enough (uncalibrated or too few examples) to say anything.
  case insufficientKnowledge

  public var medicationID: MedicationID? {
    if case .identified(let id) = self { return id }
    return nil
  }
}

/// The brain's opinion about one pill, recorded with the detection for audit.
public struct IdentityEvidence: Codable, Hashable, Sendable {
  public var embedderID: EmbedderID
  public var policyVersion: Int
  public var decision: IdentityDecision
  /// Best candidates, best first (at most three).
  public var candidates: [IdentityCandidate]

  public init(
    embedderID: EmbedderID, policyVersion: Int, decision: IdentityDecision, candidates: [IdentityCandidate]
  ) {
    self.embedderID = embedderID
    self.policyVersion = policyVersion
    self.decision = decision
    self.candidates = candidates
  }
}

/// Contiguous copy of the knowledge base for fast nearest-neighbour search.
public struct KnowledgeIndex: Sendable {
  struct Entry: Sendable {
    let medicationID: MedicationID
    let vectors: [Float]
    let groups: [UUID]
    var count: Int { groups.count }
  }

  public let embedderID: EmbedderID
  public let dimension: Int
  let entries: [Entry]

  public init(_ knowledge: KnowledgeBase) {
    embedderID = knowledge.embedderID
    dimension = knowledge.exemplars.first?.embedding.dimension ?? 0
    var grouped: [MedicationID: (vectors: [Float], groups: [UUID])] = [:]
    for exemplar in knowledge.exemplars {
      grouped[exemplar.medicationID, default: ([], [])].vectors.append(contentsOf: exemplar.embedding.vector)
      grouped[exemplar.medicationID, default: ([], [])].groups.append(exemplar.groupID)
    }
    entries = grouped.keys.sorted().map {
      Entry(medicationID: $0, vectors: grouped[$0]!.vectors, groups: grouped[$0]!.groups)
    }
  }

  /// Every medication's score for `query`, best first.
  ///
  /// - Parameters:
  ///   - excludingGroup: Exemplars from this group are ignored (used for honest self-evaluation).
  ///   - excludingMedication: This medication is ignored (simulates a pill the brain has never seen).
  func candidates(
    for query: Embedding, neighbours: Int, excludingGroup: UUID? = nil, excludingMedication: MedicationID? = nil
  ) -> [IdentityCandidate] {
    guard query.embedderID == embedderID, query.dimension == dimension, dimension > 0 else { return [] }
    let k = max(1, neighbours)
    var result: [IdentityCandidate] = []
    query.vector.withUnsafeBufferPointer { q in
      for entry in entries where entry.medicationID != excludingMedication {
        var top: [Float] = []
        var used = 0
        entry.vectors.withUnsafeBufferPointer { v in
          for row in 0..<entry.count where entry.groups[row] != excludingGroup {
            used += 1
            var sum: Float = 0
            let base = row * dimension
            for i in 0..<dimension { sum += q[i] * v[base + i] }
            if top.count < k {
              top.append(sum)
              top.sort(by: >)
            } else if sum > top[k - 1] {
              top[k - 1] = sum
              top.sort(by: >)
            }
          }
        }
        guard used > 0 else { continue }
        result.append(
          IdentityCandidate(
            medicationID: entry.medicationID, score: top.reduce(0, +) / Float(top.count), support: used))
      }
    }
    return result.sorted { ($0.score, $1.medicationID) > ($1.score, $0.medicationID) }
  }

  func groupCount(of id: MedicationID, excludingGroup: UUID?) -> Int {
    guard let entry = entries.first(where: { $0.medicationID == id }) else { return 0 }
    return Set(entry.groups.filter { $0 != excludingGroup }).count
  }
}

/// Open-set nearest-neighbour identification against the knowledge base.
public struct IdentityClassifier: Sendable {
  public let index: KnowledgeIndex
  public let policy: IdentityPolicy

  public init(knowledge: KnowledgeBase, policy: IdentityPolicy) {
    self.index = KnowledgeIndex(knowledge)
    self.policy = policy
  }

  init(index: KnowledgeIndex, policy: IdentityPolicy) {
    self.index = index
    self.policy = policy
  }

  public func classify(_ query: Embedding) -> IdentityEvidence {
    evidence(for: query, excludingGroup: nil, excludingMedication: nil)
  }

  func evidence(for query: Embedding, excludingGroup: UUID?, excludingMedication: MedicationID?) -> IdentityEvidence {
    let candidates = index.candidates(
      for: query, neighbours: policy.neighbours, excludingGroup: excludingGroup,
      excludingMedication: excludingMedication)
    let decision = decide(candidates, excludingGroup: excludingGroup)
    return IdentityEvidence(
      embedderID: index.embedderID, policyVersion: policy.version, decision: decision,
      candidates: Array(candidates.prefix(3)))
  }

  private func decide(_ candidates: [IdentityCandidate], excludingGroup: UUID?) -> IdentityDecision {
    guard let threshold = policy.acceptSimilarity else { return .insufficientKnowledge }
    let known = candidates.filter {
      $0.support >= policy.minimumExemplars
        && index.groupCount(of: $0.medicationID, excludingGroup: excludingGroup) >= policy.minimumGroups
    }
    guard let best = candidates.first else { return .insufficientKnowledge }
    guard let bestKnown = known.first else { return .insufficientKnowledge }
    // A poorly known medication scoring highest must not be overruled by a well known one.
    guard best.medicationID == bestKnown.medicationID else {
      return best.score >= threshold ? .insufficientKnowledge : .unrecognised
    }
    guard best.score >= threshold else { return .unrecognised }
    if candidates.count > 1, best.score - candidates[1].score < policy.minimumMargin { return .ambiguous }
    return .identified(best.medicationID)
  }
}
