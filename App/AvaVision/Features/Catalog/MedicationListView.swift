import AvaVisionCore
import SwiftUI

enum MedicationEditorTarget: Identifiable {
  case new
  case existing(Medication)

  var id: String {
    switch self {
    case .new: "new"
    case .existing(let medication): medication.id.rawValue
    }
  }
}

struct MedicationListView: View {
  @Environment(AppModel.self) private var app
  @State private var editing: MedicationEditorTarget?

  var body: some View {
    List {
      if app.catalog.medications.isEmpty {
        ContentUnavailableView(
          "No medications", systemImage: "pills",
          description: Text("Add the medications this pharmacy packs. Profiles refer to them by code."))
      }
      ForEach(app.catalog.medications) { medication in
        Button {
          editing = .existing(medication)
        } label: {
          VStack(alignment: .leading, spacing: 2) {
            Text(medication.displayName).foregroundStyle(.primary)
            Text(subtitle(medication)).font(.caption).foregroundStyle(.secondary)
          }
        }
      }
      .onDelete(perform: delete)
    }
    .navigationTitle("Medications")
    .toolbar {
      Button {
        editing = .new
      } label: {
        Image(systemName: "plus")
      }
      .accessibilityLabel("Add medication")
    }
    .sheet(item: $editing) { target in
      MedicationEditorView(target: target)
    }
  }

  private func subtitle(_ medication: Medication) -> String {
    let appearance = [medication.appearance.colour, medication.appearance.shape, medication.appearance.imprint]
      .compactMap { $0 }
      .filter { !$0.isEmpty }
    return ([medication.id.rawValue] + appearance).joined(separator: " · ")
  }

  private func delete(_ offsets: IndexSet) {
    let ids = offsets.map { app.catalog.medications[$0].id }
    for id in ids {
      if app.isMedicationInUse(id) {
        app.lastError = "\(id) is used by a pack profile. Remove it from the profiles first."
      } else {
        app.deleteMedication(id)
      }
    }
  }
}

struct MedicationEditorView: View {
  let target: MedicationEditorTarget
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var code: String
  @State private var name: String
  @State private var strength: String
  @State private var colour: String
  @State private var shape: String
  @State private var imprint: String
  @State private var message: String?

  init(target: MedicationEditorTarget) {
    self.target = target
    var medication: Medication?
    if case .existing(let existing) = target { medication = existing }
    _code = State(initialValue: medication?.id.rawValue ?? "")
    _name = State(initialValue: medication?.name ?? "")
    _strength = State(initialValue: medication?.strength ?? "")
    _colour = State(initialValue: medication?.appearance.colour ?? "")
    _shape = State(initialValue: medication?.appearance.shape ?? "")
    _imprint = State(initialValue: medication?.appearance.imprint ?? "")
  }

  private var isNew: Bool {
    if case .new = target { return true }
    return false
  }

  var body: some View {
    NavigationStack {
      Form {
        Section {
          TextField("Code (e.g. PBS item or internal code)", text: $code)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .disabled(!isNew)
          TextField("Name", text: $name)
          TextField("Strength (e.g. 500 mg)", text: $strength)
        } header: {
          Text("Medication")
        } footer: {
          Text("The code links profiles and model labels to this medication and cannot be changed later.")
        }
        Section {
          TextField("Colour", text: $colour)
          TextField("Shape", text: $shape)
          TextField("Imprint", text: $imprint)
        } header: {
          Text("Appearance")
        } footer: {
          Text("Helps staff during review. Appearance alone is never used to identify a medication.")
        }
        if let message {
          Section {
            Text(message).foregroundStyle(.red)
          }
        }
      }
      .navigationTitle(isNew ? "New medication" : "Edit medication")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button("Cancel") { dismiss() }
        }
        ToolbarItem(placement: .confirmationAction) {
          Button("Save", action: save)
        }
      }
    }
  }

  private func save() {
    let normalizedCode = code.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      .replacingOccurrences(of: " ", with: "-")
    let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalizedCode.isEmpty, !trimmedName.isEmpty else {
      message = "Code and name are required."
      return
    }
    let id = MedicationID(normalizedCode)
    if isNew, app.catalog.contains(id) {
      message = "A medication with code \(normalizedCode) already exists."
      return
    }
    func optional(_ text: String) -> String? {
      let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
      return trimmed.isEmpty ? nil : trimmed
    }
    app.upsertMedication(
      Medication(
        id: id, name: trimmedName, strength: strength.trimmingCharacters(in: .whitespacesAndNewlines),
        appearance: MedicationAppearance(colour: optional(colour), shape: optional(shape), imprint: optional(imprint))))
    dismiss()
  }
}
