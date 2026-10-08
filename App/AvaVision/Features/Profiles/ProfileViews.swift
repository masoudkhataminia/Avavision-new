import AvaVisionCore
import SwiftUI

/// Wraps a compartment index for `.sheet(item:)`.
struct SelectedCompartment: Identifiable, Hashable {
  let index: CompartmentIndex
  var id: CompartmentIndex { index }
}

struct ProfileListView: View {
  @Environment(AppModel.self) private var app
  @State private var isCreating = false

  var body: some View {
    List {
      if app.profiles.isEmpty {
        ContentUnavailableView(
          "No pack profiles", systemImage: "square.grid.3x3",
          description: Text("A profile lists exactly what each compartment of one pack must contain."))
      }
      ForEach(app.profiles) { profile in
        NavigationLink {
          ProfileEditorView(profileID: profile.id)
        } label: {
          ProfileRow(profile: profile)
        }
      }
      .onDelete { offsets in
        let ids = offsets.map { app.profiles[$0].id }
        for id in ids { app.deleteProfile(id) }
      }
    }
    .navigationTitle("Pack profiles")
    .toolbar {
      Button {
        isCreating = true
      } label: {
        Image(systemName: "plus")
      }
      .accessibilityLabel("New profile")
    }
    .sheet(isPresented: $isCreating) {
      NewProfileView()
    }
  }
}

struct ProfileRow: View {
  @Environment(AppModel.self) private var app
  let profile: PackProfile

  var body: some View {
    let issues = app.issues(for: profile)
    VStack(alignment: .leading, spacing: 2) {
      Text(profile.reference).font(.body.weight(.semibold))
      HStack(spacing: 6) {
        Text(app.layout(id: profile.layoutID)?.displayName ?? profile.layoutID)
        Text("·")
        Text("\(profile.totalDoses) doses")
        if !issues.isEmpty {
          Label("\(issues.count) to fix", systemImage: "exclamationmark.triangle.fill")
            .foregroundStyle(.orange)
        }
      }
      .font(.caption)
      .foregroundStyle(.secondary)
    }
  }
}

struct NewProfileView: View {
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var reference = ""
  @State private var layoutID = PackLayout.weekly7x4.id

  var body: some View {
    NavigationStack {
      Form {
        Section {
          TextField("Pack reference (e.g. pack barcode)", text: $reference)
            .autocorrectionDisabled()
        } footer: {
          Text("Never enter patient names, dates of birth, addresses or script numbers.")
        }
        Section("Layout") {
          Picker("Layout", selection: $layoutID) {
            ForEach(app.layouts) { layout in
              Text(layout.displayName).tag(layout.id)
            }
          }
        }
      }
      .navigationTitle("New profile")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button("Cancel") { dismiss() }
        }
        ToolbarItem(placement: .confirmationAction) {
          Button("Create", action: create)
            .disabled(reference.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
      }
    }
  }

  private func create() {
    guard let layout = app.layout(id: layoutID) else { return }
    let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
    app.saveProfile(PackProfile.empty(reference: trimmed, layout: layout))
    dismiss()
  }
}

struct ProfileEditorView: View {
  let profileID: UUID
  @Environment(AppModel.self) private var app
  @State private var editing: SelectedCompartment?

  var body: some View {
    if let profile = app.profile(id: profileID), let layout = app.layout(id: profile.layoutID) {
      let issues = app.issues(for: profile)
      List {
        Section("Reference") {
          TextField(
            "Pack reference",
            text: Binding(
              get: { profile.reference },
              set: { newValue in
                var updated = profile
                updated.reference = newValue
                app.saveProfile(updated)
              })
          )
          .autocorrectionDisabled()
        }

        if !issues.isEmpty {
          Section("To fix before checking") {
            ForEach(issues, id: \.self) { issue in
              Label(issue.text(catalog: app.catalog, layout: layout), systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
            }
          }
        }

        Section {
          ProfileGrid(layout: layout, profile: profile) { editing = SelectedCompartment(index: $0) }
        } header: {
          Text("Compartments")
        } footer: {
          Text("Tap a compartment to set its contents. Numbers show the expected doses.")
        }

        Section("Totals") {
          let totals = totalsByMedication(profile)
          if totals.isEmpty {
            Text("Every compartment is empty.").foregroundStyle(.secondary)
          }
          ForEach(totals.keys.sorted(), id: \.self) { id in
            LabeledContent(app.catalog.medication(id)?.displayName ?? id.rawValue, value: "\(totals[id] ?? 0)")
          }
        }
      }
      .navigationTitle(profile.reference)
      .sheet(item: $editing) { selection in
        CompartmentEditorView(profileID: profileID, index: selection.index)
      }
    } else {
      ContentUnavailableView("Profile not found", systemImage: "questionmark.square.dashed")
    }
  }

  private func totalsByMedication(_ profile: PackProfile) -> [MedicationID: Int] {
    var totals: [MedicationID: Int] = [:]
    for compartment in profile.compartments {
      for (id, quantity) in compartment.quantities { totals[id, default: 0] += quantity }
    }
    return totals
  }
}

/// Read-only grid of expected dose counts with row and column labels.
struct ProfileGrid: View {
  let layout: PackLayout
  let profile: PackProfile
  let onSelect: (CompartmentIndex) -> Void

  var body: some View {
    Grid(horizontalSpacing: 4, verticalSpacing: 4) {
      GridRow {
        Text("")
        ForEach(0..<layout.columns, id: \.self) { column in
          Text(layout.columnLabels.indices.contains(column) ? layout.columnLabels[column] : "\(column + 1)")
            .font(.caption2)
            .lineLimit(1)
            .minimumScaleFactor(0.6)
        }
      }
      ForEach(0..<layout.rows, id: \.self) { row in
        GridRow {
          Text(layout.rowLabels.indices.contains(row) ? layout.rowLabels[row] : "\(row + 1)")
            .font(.caption2)
            .lineLimit(1)
            .minimumScaleFactor(0.6)
            .frame(width: 52, alignment: .leading)
          ForEach(0..<layout.columns, id: \.self) { column in
            let index = CompartmentIndex(row: row, column: column)
            let expectation = profile.expectation(for: index)
            Button {
              onSelect(index)
            } label: {
              Text(expectation.map { "\($0.totalQuantity)" } ?? "?")
                .font(.callout.monospacedDigit().weight(.semibold))
                .frame(maxWidth: .infinity, minHeight: 40)
                .background(background(expectation), in: RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(layout.label(for: index)): \(expectation?.totalQuantity ?? 0) doses")
          }
        }
      }
    }
    .padding(.vertical, 4)
  }

  private func background(_ expectation: CompartmentExpectation?) -> Color {
    guard let expectation else { return .orange.opacity(0.3) }
    return expectation.totalQuantity > 0 ? Color.accentColor.opacity(0.2) : Color.secondary.opacity(0.1)
  }
}

struct CompartmentEditorView: View {
  let profileID: UUID
  let index: CompartmentIndex
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var items: [ExpectedItem] = []
  @State private var hasLoaded = false
  @State private var newMedication: MedicationID?

  var body: some View {
    NavigationStack {
      if let profile = app.profile(id: profileID), let layout = app.layout(id: profile.layoutID) {
        Form {
          Section("Contents") {
            if items.isEmpty {
              Text("Empty compartment").foregroundStyle(.secondary)
            }
            ForEach($items, id: \.medicationID) { $item in
              Stepper(value: $item.quantity, in: 1...20) {
                HStack {
                  Text(app.catalog.medication(item.medicationID)?.displayName ?? item.medicationID.rawValue)
                  Spacer()
                  Text("× \(item.quantity)").monospacedDigit()
                }
              }
            }
            .onDelete { items.remove(atOffsets: $0) }
          }

          Section("Add medication") {
            let available = app.catalog.medications.filter { medication in
              !items.contains { $0.medicationID == medication.id }
            }
            if app.catalog.medications.isEmpty {
              Text("Add medications in the Medications screen first.").foregroundStyle(.secondary)
            } else {
              Picker("Medication", selection: $newMedication) {
                Text("Choose…").tag(MedicationID?.none)
                ForEach(available) { medication in
                  Text(medication.displayName).tag(MedicationID?.some(medication.id))
                }
              }
              Button("Add") {
                guard let id = newMedication else { return }
                items.append(ExpectedItem(medicationID: id, quantity: 1))
                newMedication = nil
              }
              .disabled(newMedication == nil)
            }
          }

          Section {
            Button("Apply to whole row: \(rowLabel(layout))") {
              apply(to: (0..<layout.columns).map { CompartmentIndex(row: index.row, column: $0) }, profile: profile)
            }
            Button("Apply to whole column: \(columnLabel(layout))") {
              apply(to: (0..<layout.rows).map { CompartmentIndex(row: $0, column: index.column) }, profile: profile)
            }
            Button("Apply to every compartment") {
              apply(to: layout.allCompartments, profile: profile)
            }
          } header: {
            Text("Save and copy")
          } footer: {
            Text("Replaces the contents of those compartments with the list above.")
          }
        }
        .navigationTitle(layout.label(for: index))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .cancellationAction) {
            Button("Cancel") { dismiss() }
          }
          ToolbarItem(placement: .confirmationAction) {
            Button("Save") { apply(to: [index], profile: profile) }
          }
        }
        .onAppear {
          guard !hasLoaded else { return }
          items = profile.expectation(for: index)?.items ?? []
          hasLoaded = true
        }
      } else {
        ContentUnavailableView("Profile not found", systemImage: "questionmark.square.dashed")
      }
    }
  }

  private func rowLabel(_ layout: PackLayout) -> String {
    layout.rowLabels.indices.contains(index.row) ? layout.rowLabels[index.row] : "\(index.row + 1)"
  }

  private func columnLabel(_ layout: PackLayout) -> String {
    layout.columnLabels.indices.contains(index.column) ? layout.columnLabels[index.column] : "\(index.column + 1)"
  }

  private func apply(to indices: [CompartmentIndex], profile: PackProfile) {
    var updated = profile
    for target in indices { updated.setItems(items, for: target) }
    app.saveProfile(updated)
    dismiss()
  }
}
