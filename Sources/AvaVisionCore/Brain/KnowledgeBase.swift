import Foundation

/// Where a remembered pill image came from. Only pharmacist-confirmed sources exist.
public enum ExemplarSource: String, Codable, Hashable, Sendable, CaseIterable {
  /// A pharmacist photographed known pills to teach the medication.
  case teaching
  /// A compartment the pharmacist confirmed correct during a check.
  case confirmedCheck
  /// A pharmacist labelled the pill in the labelling queue.
  case labelled
}

/// One remembered, pharmacist-confirmed example of what a medication looks like.
public struct PillExemplar: Codable, Hashable, Sendable, Identifiable {
  public var id: UUID
  public var medicationID: MedicationID
  public var embedding: Embedding
  public var source: ExemplarSource
  /// Exemplars from the same photo or check share a group; calibration never compares a pill with
  /// its own group, because near-identical views would make accuracy look better than it is.
  public var groupID: UUID
  public var createdAt: Date
  /// File name of the stored pill crop, kept so the pill can be re-embedded by a newer model.
  public var cropFile: String?

  public init(
    id: UUID = UUID(), medicationID: MedicationID, embedding: Embedding, source: ExemplarSource, groupID: UUID,
    createdAt: Date = Date(), cropFile: String? = nil
  ) {
    self.id = id
    self.medicationID = medicationID
    self.embedding = embedding
    self.source = source
    self.groupID = groupID
    self.createdAt = createdAt
    self.cropFile = cropFile
  }
}

public struct KnowledgePolicy: Codable, Hashable, Sendable {
  /// Memory per medication; beyond this the most redundant exemplar is forgotten.
  public var maximumExemplarsPerMedication: Int
  /// A new exemplar this similar to an existing one of the same medication adds nothing.
  public var duplicateSimilarity: Float

  public init(maximumExemplarsPerMedication: Int, duplicateSimilarity: Float) {
    self.maximumExemplarsPerMedication = maximumExemplarsPerMedication
    self.duplicateSimilarity = duplicateSimilarity
  }

  public static let standard = KnowledgePolicy(maximumExemplarsPerMedication: 400, duplicateSimilarity: 0.995)
}

/// The brain's memory: pharmacist-confirmed pill appearances for one embedder.
public struct KnowledgeBase: Codable, Hashable, Sendable {
  public enum AddOutcome: Hashable, Sendable {
    case added
    /// Added, and the listed most-redundant exemplar was forgotten to stay within capacity.
    case addedReplacing(UUID)
    case duplicate
    case incompatibleEmbedder
  }

  public let embedderID: EmbedderID
  public var policy: KnowledgePolicy
  public private(set) var exemplars: [PillExemplar]

  public init(embedderID: EmbedderID, policy: KnowledgePolicy = .standard, exemplars: [PillExemplar] = []) {
    self.embedderID = embedderID
    self.policy = policy
    self.exemplars = []
    for exemplar in exemplars { add(exemplar) }
  }

  public var medications: Set<MedicationID> { Set(exemplars.map(\.medicationID)) }

  public func exemplars(for id: MedicationID) -> [PillExemplar] {
    exemplars.filter { $0.medicationID == id }
  }

  public func count(for id: MedicationID) -> Int {
    exemplars.reduce(0) { $0 + ($1.medicationID == id ? 1 : 0) }
  }

  /// Number of distinct photos/checks the medication was learned from.
  public func groupCount(for id: MedicationID) -> Int {
    Set(exemplars(for: id).map(\.groupID)).count
  }

  @discardableResult
  public mutating func add(_ exemplar: PillExemplar) -> AddOutcome {
    guard exemplar.embedding.embedderID == embedderID,
      exemplars.first.map({ $0.embedding.dimension == exemplar.embedding.dimension }) ?? true
    else { return .incompatibleEmbedder }

    let same = exemplars.indices.filter { exemplars[$0].medicationID == exemplar.medicationID }
    let isDuplicate = same.contains {
      (exemplars[$0].embedding.similarity(to: exemplar.embedding) ?? -1) >= policy.duplicateSimilarity
    }
    guard !isDuplicate else { return .duplicate }

    exemplars.append(exemplar)
    guard same.count + 1 > max(1, policy.maximumExemplarsPerMedication) else { return .added }
    let forgotten = mostRedundant(among: same + [exemplars.count - 1])
    let removedID = exemplars[forgotten].id
    exemplars.remove(at: forgotten)
    return .addedReplacing(removedID)
  }

  public mutating func remove(_ id: UUID) {
    exemplars.removeAll { $0.id == id }
  }

  public mutating func forget(medication id: MedicationID) {
    exemplars.removeAll { $0.medicationID == id }
  }

  /// The exemplar whose nearest same-medication neighbour is closest: forgetting it loses the least
  /// variety. Ties keep the newer exemplar.
  private func mostRedundant(among indices: [Int]) -> Int {
    var best = indices[0]
    var bestScore = -Float.infinity
    for i in indices {
      var nearest = -Float.infinity
      for j in indices where j != i {
        nearest = max(nearest, exemplars[i].embedding.similarity(to: exemplars[j].embedding) ?? -1)
      }
      if nearest > bestScore || (nearest == bestScore && exemplars[i].createdAt < exemplars[best].createdAt) {
        bestScore = nearest
        best = i
      }
    }
    return best
  }
}
