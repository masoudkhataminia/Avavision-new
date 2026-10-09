import Foundation

/// A pill seen during a check, with what the brain thought of it.
public struct PillSighting: Codable, Hashable, Sendable, Identifiable {
  public var id: UUID
  public var compartment: CompartmentIndex
  public var embedding: Embedding?
  public var identity: IdentityEvidence?
  /// File name of the stored crop, if one was kept.
  public var cropFile: String?

  public init(
    id: UUID = UUID(), compartment: CompartmentIndex, embedding: Embedding?, identity: IdentityEvidence?,
    cropFile: String? = nil
  ) {
    self.id = id
    self.compartment = compartment
    self.embedding = embedding
    self.identity = identity
    self.cropFile = cropFile
  }
}

/// Pills in a confirmed compartment holding several medications: the pharmacist says which is which.
public struct LabellingTask: Codable, Hashable, Sendable, Identifiable {
  public var id: UUID
  public var checkID: UUID
  public var createdAt: Date
  public var compartmentLabel: String
  public var expected: [MedicationID: Int]
  public var sightings: [PillSighting]
  /// Higher first: tasks where the brain was least sure, or that teach rarely seen medications.
  public var priority: Double

  public init(
    id: UUID = UUID(), checkID: UUID, createdAt: Date, compartmentLabel: String, expected: [MedicationID: Int],
    sightings: [PillSighting], priority: Double
  ) {
    self.id = id
    self.checkID = checkID
    self.createdAt = createdAt
    self.compartmentLabel = compartmentLabel
    self.expected = expected
    self.sightings = sightings
    self.priority = priority
  }

  public enum LabellingError: Error, Hashable, Sendable {
    case unlabelledPills([UUID])
    case countsDoNotMatchExpected
  }

  /// Turns the pharmacist's labels into exemplars and track-record observations.
  /// Labels must cover every pill and add up to the expected quantities.
  public func resolve(labels: [UUID: MedicationID], at date: Date = Date()) throws -> LearningPlan {
    let missing = sightings.map(\.id).filter { labels[$0] == nil }
    guard missing.isEmpty else { throw LabellingError.unlabelledPills(missing) }
    let counts = sightings.reduce(into: [MedicationID: Int]()) { $0[labels[$1.id]!, default: 0] += 1 }
    guard counts == expected.filter({ $0.value > 0 }) else { throw LabellingError.countsDoNotMatchExpected }
    var plan = LearningPlan()
    for sighting in sightings {
      plan.learn(sighting, as: labels[sighting.id]!, source: .labelled, groupID: checkID, at: date)
    }
    return plan
  }
}

/// What the brain should learn from one event.
public struct LearningPlan: Hashable, Sendable {
  public struct Observation: Hashable, Sendable {
    public var decision: IdentityDecision
    public var truth: MedicationID
  }

  public var exemplars: [PillExemplar] = []
  public var observations: [Observation] = []
  public var labellingTasks: [LabellingTask] = []

  public init() {}

  mutating func learn(
    _ sighting: PillSighting, as medication: MedicationID, source: ExemplarSource, groupID: UUID, at date: Date
  ) {
    if let embedding = sighting.embedding {
      exemplars.append(
        PillExemplar(
          medicationID: medication, embedding: embedding, source: source, groupID: groupID, createdAt: date,
          cropFile: sighting.cropFile))
    }
    if let identity = sighting.identity {
      observations.append(Observation(decision: identity.decision, truth: medication))
    }
  }
}

public enum LearningPlanner {
  /// Learns only from compartments a pharmacist personally inspected and confirmed correct.
  ///
  /// Compartments accepted automatically are never fed back (the brain must not confirm itself),
  /// and corrected or unresolved compartments are skipped because their true content is unknown.
  /// - Parameter sightings: Pills of one frame (the evidence frame), clearly inside a compartment.
  public static func plan(
    checkID: UUID, signOff: PharmacistSignOff, profile: PackProfile, layout: PackLayout,
    sightings: [PillSighting], at date: Date = Date()
  ) -> LearningPlan {
    var plan = LearningPlan()
    let byCompartment = Dictionary(grouping: sightings, by: \.compartment)
    for review in signOff.reviews where review.outcome == .confirmedCorrect {
      guard let expectation = profile.expectation(for: review.compartment) else { continue }
      let quantities = expectation.quantities.filter { $0.value > 0 }
      let seen = byCompartment[review.compartment] ?? []
      guard !seen.isEmpty, seen.count == expectation.totalQuantity else { continue }

      if quantities.count == 1, let medication = quantities.keys.first {
        for sighting in seen {
          plan.learn(sighting, as: medication, source: .confirmedCheck, groupID: checkID, at: date)
        }
      } else if seen.allSatisfy({ $0.embedding != nil }) {
        let unsure = seen.filter { $0.identity?.decision.medicationID == nil }.count
        plan.labellingTasks.append(
          LabellingTask(
            checkID: checkID, createdAt: date, compartmentLabel: layout.label(for: review.compartment),
            expected: quantities, sightings: seen, priority: Double(unsure) / Double(seen.count)))
      }
    }
    return plan
  }
}
