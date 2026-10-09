import AvaVisionCore
import PhotosUI
import SwiftUI
import UIKit

/// The brain's dashboard: what it knows, how far each medication is from earned trust, and its to-do list.
struct BrainView: View {
  @Environment(AppModel.self) private var app
  @State private var isTeaching = false
  @State private var exportMessage: String?
  @State private var confirmingForget: MedicationID?

  var body: some View {
    List {
      if let brain = app.brain {
        Section {
          LabeledContent("Pills remembered", value: "\(brain.knowledge.exemplars.count)")
          LabeledContent("Medications known", value: "\(brain.knowledge.medications.count)")
          LabeledContent("Trusted medications", value: "\(brain.trustedMedications.count)")
          LabeledContent("Appearance model", value: app.embedderTitle)
            .font(.caption)
          if let activity = app.brainActivity {
            Label(activity, systemImage: "hourglass").foregroundStyle(.secondary)
          }
          if let warning = app.brainWarning {
            Label(warning, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
          }
        } header: {
          Text("Brain")
        } footer: {
          Text(
            "The brain learns only from pills a pharmacist confirmed. It can raise concerns from day one, but it "
              + "accepts a medication automatically only after a long, error-free track record, and loses that "
              + "trust at the first mistake.")
        }

        Section {
          Button {
            isTeaching = true
          } label: {
            Label("Teach a medication", systemImage: "camera.macro")
          }
          .disabled(app.catalog.medications.isEmpty)
          NavigationLink {
            LabellingQueueView()
          } label: {
            LabeledContent {
              Text("\(brain.labellingQueue.count)")
            } label: {
              Label("Pills waiting for labels", systemImage: "tag")
            }
          }
        }

        Section("Calibration") {
          if let report = brain.lastCalibration {
            Text(report.outcome.text).font(.callout)
            if case .calibrated = report.outcome {
              LabeledContent(
                "Self-test precision (lower bound)",
                value: report.precisionLowerBound.formatted(.percent.precision(.fractionLength(1))))
              LabeledContent(
                "Self-test coverage", value: report.coverage.formatted(.percent.precision(.fractionLength(0))))
            }
          } else {
            Text("Not calibrated yet. Teach at least two medications, each from two or more photos.")
              .font(.callout)
          }
          Button("Recalibrate now") { app.recalibrateBrain() }
            .disabled(brain.knowledge.exemplars.isEmpty)
        }

        Section("Medications") {
          if brain.knowledge.medications.isEmpty && brain.ledger.records.isEmpty {
            Text("Nothing learned yet.").foregroundStyle(.secondary)
          }
          ForEach(medications(brain), id: \.self) { id in
            MedicationBrainRow(id: id)
              .swipeActions {
                Button("Forget", role: .destructive) { confirmingForget = id }
              }
          }
        }

        Section {
          Button {
            Task {
              do {
                let count = try await app.exportTrainingData()
                exportMessage = "Exported \(count) labelled pill images to Files › On My iPhone › AvaVision."
              } catch {
                exportMessage = "Export failed: \(error.localizedDescription)"
              }
            }
          } label: {
            Label("Export training data", systemImage: "square.and.arrow.up.on.square")
          }
          .disabled(brain.knowledge.exemplars.isEmpty)
        } footer: {
          Text("Used to train AvaVision's own detector and appearance model offline (see the training guide).")
        }
      } else {
        ProgressView("Loading brain…")
      }
    }
    .navigationTitle("Brain")
    .sheet(isPresented: $isTeaching) { TeachMedicationView() }
    .alert("Export", isPresented: Binding(get: { exportMessage != nil }, set: { if !$0 { exportMessage = nil } })) {
      Button("OK", role: .cancel) {}
    } message: {
      Text(exportMessage ?? "")
    }
    .confirmationDialog(
      "Forget everything the brain learned about this medication?",
      isPresented: Binding(get: { confirmingForget != nil }, set: { if !$0 { confirmingForget = nil } }),
      titleVisibility: .visible
    ) {
      Button("Forget", role: .destructive) {
        if let id = confirmingForget { app.forgetInBrain(id) }
      }
    }
  }

  private func medications(_ brain: BrainState) -> [MedicationID] {
    Array(brain.knowledge.medications.union(brain.ledger.records.keys)).sorted()
  }
}

struct MedicationBrainRow: View {
  let id: MedicationID
  @Environment(AppModel.self) private var app

  var body: some View {
    if let brain = app.brain {
      let status = brain.trustStatus(of: id)
      let record = brain.ledger.record(for: id)
      VStack(alignment: .leading, spacing: 4) {
        HStack {
          Text(app.catalog.medication(id)?.displayName ?? id.rawValue).font(.body.weight(.semibold))
          Spacer()
          Text(status.title).font(.caption.weight(.semibold)).foregroundStyle(status.color)
        }
        if case .learning(let progress) = status {
          ProgressView(value: progress).tint(.orange)
        }
        Text(
          "\(brain.knowledge.count(for: id)) images from \(brain.knowledge.groupCount(for: id)) photos · "
            + "\(record.correct) correct, \(record.falseIdentifications) wrong, \(record.abstentions) unsure"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
      }
      .padding(.vertical, 2)
    }
  }
}

/// Teach the brain a medication: photograph several pills of it on a plain, dark surface.
struct TeachMedicationView: View {
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var medication: MedicationID?
  @State private var photoItems: [PhotosPickerItem] = []
  @State private var isCameraPresented = false
  @State private var crops: [CGImage] = []
  @State private var isWorking = false
  @State private var message: String?

  var body: some View {
    NavigationStack {
      Form {
        Section {
          Picker("Medication", selection: $medication) {
            Text("Choose…").tag(MedicationID?.none)
            ForEach(app.catalog.medications) { medication in
              Text(medication.displayName).tag(MedicationID?.some(medication.id))
            }
          }
        } footer: {
          Text(
            "Spread 5–20 pills of this one medication, not touching, on a plain dark surface under the station "
              + "light. Take two or more photos, turning the pills over between photos.")
        }

        Section {
          Button {
            isCameraPresented = true
          } label: {
            Label("Take photo", systemImage: "camera")
          }
          .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
          PhotosPicker(selection: $photoItems, maxSelectionCount: 10, matching: .images) {
            Label("Choose photos", systemImage: "photo.on.rectangle")
          }
        }

        if isWorking {
          Section { ProgressView("Finding pills…") }
        }

        if !crops.isEmpty {
          Section("Pills found (\(crops.count))") {
            ScrollView(.horizontal) {
              HStack {
                ForEach(crops.indices, id: \.self) { index in
                  Image(decorative: crops[index], scale: 1)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
              }
            }
            Button("Clear") { crops.removeAll() }
          }
        }

        if let message {
          Section { Text(message) }
        }
      }
      .navigationTitle("Teach medication")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button("Close") { dismiss() }
        }
        ToolbarItem(placement: .confirmationAction) {
          Button("Teach") { Task { await teach() } }
            .disabled(medication == nil || crops.isEmpty || isWorking)
        }
      }
      .onChange(of: photoItems) { _, items in
        guard !items.isEmpty else { return }
        Task { await load(items) }
      }
      .fullScreenCover(isPresented: $isCameraPresented) {
        CameraPicker { image in
          isCameraPresented = false
          if let image { Task { await findPills(in: [image]) } }
        }
        .ignoresSafeArea()
      }
    }
  }

  private func load(_ items: [PhotosPickerItem]) async {
    var images: [CGImage] = []
    for item in items {
      if let data = try? await item.loadTransferable(type: Data.self),
        let image = ImageConversion.uprightImage(fromPhotoData: data, maxDimension: 2400)
      {
        images.append(image)
      }
    }
    photoItems = []
    await findPills(in: images)
  }

  private func findPills(in images: [CGImage]) async {
    isWorking = true
    let found = await Task.detached(priority: .userInitiated) {
      images.flatMap { LoosePillFinder.pills(in: $0) }
    }.value
    crops.append(contentsOf: found)
    isWorking = false
    message = found.isEmpty ? "No pills were found. Use a plain dark surface and keep pills apart." : nil
  }

  private func teach() async {
    guard let medication else { return }
    isWorking = true
    let added = await app.teach(medication, crops: crops)
    isWorking = false
    crops.removeAll()
    let name = app.catalog.medication(medication)?.displayName ?? medication.rawValue
    message = "The brain learned \(added) new images of \(name). Take more photos from other angles to strengthen it."
  }
}

/// Pills from confirmed compartments that held several medications: the pharmacist says which is which.
struct LabellingQueueView: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    List {
      let tasks = app.brain?.labellingQueue ?? []
      if tasks.isEmpty {
        ContentUnavailableView(
          "Nothing to label", systemImage: "tag",
          description: Text("Pills from confirmed compartments with several medications appear here."))
      }
      ForEach(tasks) { task in
        NavigationLink {
          LabellingTaskView(task: task)
        } label: {
          VStack(alignment: .leading, spacing: 2) {
            Text(task.compartmentLabel).font(.body.weight(.semibold))
            Text("\(task.sightings.count) pills · \(task.createdAt.formatted(date: .abbreviated, time: .shortened))")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
      }
    }
    .navigationTitle("Label pills")
  }
}

struct LabellingTaskView: View {
  let task: LabellingTask
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var labels: [UUID: MedicationID] = [:]
  @State private var message: String?

  var body: some View {
    let medications = task.expected.keys.sorted()
    Form {
      Section {
        Text("This compartment was confirmed correct. Tap the medication for each pill.")
          .font(.callout)
        ForEach(medications, id: \.self) { id in
          LabeledContent(name(id), value: "× \(task.expected[id] ?? 0)")
        }
      }
      ForEach(task.sightings) { sighting in
        HStack(spacing: 12) {
          crop(sighting)
          Picker("Medication", selection: binding(sighting.id)) {
            Text("–").tag(MedicationID?.none)
            ForEach(medications, id: \.self) { id in
              Text(name(id)).tag(MedicationID?.some(id))
            }
          }
          .labelsHidden()
        }
      }
      if let message {
        Section { Text(message).foregroundStyle(.red) }
      }
      Section {
        Button("Save labels") {
          do {
            try app.resolveLabellingTask(task.id, labels: labels)
            dismiss()
          } catch {
            message = "Every pill needs a label, and the totals must match the expected quantities."
          }
        }
        Button("Discard", role: .destructive) {
          app.discardLabellingTask(task.id)
          dismiss()
        }
      }
    }
    .navigationTitle(task.compartmentLabel)
    .navigationBarTitleDisplayMode(.inline)
    .onAppear {
      for sighting in task.sightings where labels[sighting.id] == nil {
        if let id = sighting.identity?.decision.medicationID, task.expected[id] != nil { labels[sighting.id] = id }
      }
    }
  }

  private func name(_ id: MedicationID) -> String {
    app.catalog.medication(id)?.displayName ?? id.rawValue
  }

  private func binding(_ id: UUID) -> Binding<MedicationID?> {
    Binding(get: { labels[id] }, set: { labels[id] = $0 })
  }

  @ViewBuilder
  private func crop(_ sighting: PillSighting) -> some View {
    if let file = sighting.cropFile, let url = app.cropURL(file), let image = PillCrops.image(contentsOf: url) {
      Image(decorative: image, scale: 1)
        .resizable()
        .scaledToFill()
        .frame(width: 64, height: 64)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    } else {
      RoundedRectangle(cornerRadius: 8).fill(.secondary.opacity(0.2)).frame(width: 64, height: 64)
    }
  }
}

/// Minimal camera capture for teaching photos.
struct CameraPicker: UIViewControllerRepresentable {
  let onFinish: (CGImage?) -> Void

  func makeCoordinator() -> Coordinator { Coordinator(onFinish: onFinish) }

  func makeUIViewController(context: Context) -> UIImagePickerController {
    let picker = UIImagePickerController()
    picker.sourceType = .camera
    picker.delegate = context.coordinator
    return picker
  }

  func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

  final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    let onFinish: (CGImage?) -> Void

    init(onFinish: @escaping (CGImage?) -> Void) {
      self.onFinish = onFinish
    }

    func imagePickerController(
      _ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
      let image = (info[.originalImage] as? UIImage).flatMap { image -> CGImage? in
        let renderer = UIGraphicsImageRenderer(size: image.size, format: .init(for: .init(displayScale: 1)))
        return renderer.image { _ in image.draw(at: .zero) }.cgImage
      }
      onFinish(image)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
      onFinish(nil)
    }
  }
}
