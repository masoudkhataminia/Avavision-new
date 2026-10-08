import AvaVisionCore
import SwiftUI

struct ResultView: View {
  let flow: CheckFlowModel
  @Environment(AppModel.self) private var app
  @State private var selected: SelectedCompartment?

  var body: some View {
    if let result = flow.result {
      let verdicts = Dictionary(uniqueKeysWithValues: result.compartments.map { ($0.compartment, $0) })
      List {
        Section {
          Label {
            VStack(alignment: .leading, spacing: 2) {
              Text(result.status.title).font(.headline)
              Text(result.capability.title).font(.caption).foregroundStyle(.secondary)
            }
          } icon: {
            Image(systemName: result.status.symbol).foregroundStyle(result.status.color)
          }
          ForEach(result.packFindings, id: \.self) { finding in
            Label(finding.text, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
          }
        }

        if let frame = flow.evidenceFrame, result.status != .retakeRequired {
          Section("Evidence photo") {
            AnnotatedImage(
              image: frame.image, registration: frame.observation.registration.registration, layout: flow.layout,
              cellColors: verdicts.mapValues(\.status.color))
          }
        }

        Section {
          CompartmentGrid(layout: flow.layout, verdicts: verdicts) { selected = SelectedCompartment(index: $0) }
        } header: {
          Text("Compartments")
        } footer: {
          Text("Counted / expected. Tap a compartment for details.")
        }

        Section {
          Button {
            flow.showSignOff()
          } label: {
            Label(
              result.status == .retakeRequired ? "Check manually and sign off" : "Continue to pharmacist sign-off",
              systemImage: "signature")
          }
          Button {
            flow.retake()
          } label: {
            Label("Retake photos", systemImage: "arrow.counterclockwise")
          }
        }
      }
      .navigationTitle("Result")
      .navigationBarTitleDisplayMode(.inline)
      .sheet(item: $selected) { selection in
        CompartmentDetailView(
          layout: flow.layout, index: selection.index, verdict: verdicts[selection.index],
          expectation: flow.session.profile.expectation(for: selection.index))
      }
    }
  }
}

struct CompartmentGrid: View {
  let layout: PackLayout
  let verdicts: [CompartmentIndex: CompartmentVerdict]
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
            cell(CompartmentIndex(row: row, column: column))
          }
        }
      }
    }
    .padding(.vertical, 4)
  }

  private func cell(_ index: CompartmentIndex) -> some View {
    let verdict = verdicts[index]
    let color = verdict?.status.color ?? .gray
    return Button {
      onSelect(index)
    } label: {
      VStack(spacing: 2) {
        Image(systemName: verdict?.status.symbol ?? "circle.dashed")
          .font(.caption)
        Text(countText(verdict))
          .font(.caption2.monospacedDigit())
          .lineLimit(1)
          .minimumScaleFactor(0.6)
      }
      .frame(maxWidth: .infinity, minHeight: 44)
      .foregroundStyle(color)
      .background(color.opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
    }
    .buttonStyle(.plain)
    .accessibilityLabel("\(layout.label(for: index)): \(verdict?.status.title ?? "not evaluated")")
  }

  private func countText(_ verdict: CompartmentVerdict?) -> String {
    let observed = verdict?.observedCount.map(String.init) ?? "–"
    let expected = verdict?.expectedCount.map(String.init) ?? "?"
    return "\(observed)/\(expected)"
  }
}

struct CompartmentDetailView: View {
  let layout: PackLayout
  let index: CompartmentIndex
  let verdict: CompartmentVerdict?
  let expectation: CompartmentExpectation?
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      List {
        if let verdict {
          Section {
            Label(verdict.status.title, systemImage: verdict.status.symbol)
              .foregroundStyle(verdict.status.color)
              .font(.headline)
          }
        }

        Section("Expected") {
          if let expectation {
            if expectation.items.isEmpty {
              Text("Empty compartment")
            }
            ForEach(expectation.items, id: \.medicationID) { item in
              LabeledContent(name(item.medicationID), value: "× \(item.quantity)")
            }
          } else {
            Text("Not defined in the profile").foregroundStyle(.orange)
          }
        }

        Section("Observed") {
          LabeledContent("Counted", value: verdict?.observedCount.map(String.init) ?? "No agreed count")
          if let observed = verdict?.observedMedications, !observed.isEmpty {
            ForEach(observed.keys.sorted(), id: \.self) { id in
              LabeledContent(name(id), value: "× \(observed[id] ?? 0)")
            }
          }
        }

        if let verdict, !verdict.findings.isEmpty {
          Section("Findings") {
            ForEach(verdict.findings, id: \.self) { finding in
              Text(finding.text(catalog: app.catalog))
            }
          }
        }
      }
      .navigationTitle(layout.label(for: index))
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") { dismiss() }
        }
      }
    }
  }

  private func name(_ id: MedicationID) -> String {
    app.catalog.medication(id)?.displayName ?? id.rawValue
  }
}
