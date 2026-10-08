import AvaVisionCore
import SwiftUI

struct SettingsView: View {
  @Environment(AppModel.self) private var app
  @AppStorage(PackOrientation.storageKey) private var orientation: PackOrientation = .automatic

  var body: some View {
    Form {
      Section("Detection model") {
        ModelStatusBanner()
        if let model = app.modelState.activeModel {
          LabeledContent("Model", value: "\(model.manifest.modelID) \(model.manifest.version)")
          LabeledContent("Stage", value: model.manifest.stage.rawValue)
          LabeledContent("SHA-256", value: String(model.manifest.modelSHA256.prefix(16)) + "…")
            .font(.caption)
          if !model.decision.blockers.isEmpty {
            DisclosureGroup("Why some capabilities are locked (\(model.decision.blockers.count))") {
              ForEach(model.decision.blockers, id: \.self) { blocker in
                Text(blocker.text).font(.caption)
              }
            }
          }
        }
      }

      Section {
        ForEach(app.layouts) { layout in
          NavigationLink {
            LayoutEditorView(layoutID: layout.id)
          } label: {
            LabeledContent(layout.displayName, value: layout.isCalibrated ? "Calibrated" : "Not calibrated")
          }
        }
        Picker("Pack orientation in photo", selection: $orientation) {
          ForEach(PackOrientation.allCases) { option in
            Text(option.title).tag(option)
          }
        }
      } header: {
        Text("Pack layout")
      } footer: {
        Text(
          "On the capture screen the yellow label must sit on the first compartment. If it does not, change the orientation."
        )
      }

      Section("Safety rules") {
        LabeledContent("Photos that must agree", value: "\(DecisionPolicy.standard.requiredConsistentFrames)")
        LabeledContent(
          "Minimum detection confidence",
          value: DecisionPolicy.standard.minimumDetectionConfidence.formatted(.percent))
        LabeledContent("Identity gate: holdout samples", value: "≥ \(ReleaseGatePolicy.standard.minimumSamples)")
        LabeledContent(
          "Identity gate: false acceptance",
          value: "≤ "
            + ReleaseGatePolicy.standard.maximumFalseAcceptanceRate.formatted(.percent.precision(.fractionLength(1))))
      }

      Section("This device") {
        LabeledContent("App version", value: app.appVersion)
        LabeledContent("Device ID", value: app.deviceIdentifier).font(.caption)
      }

      Section {
        Text(SafetyText.disclaimer).font(.footnote).foregroundStyle(.secondary)
      }
    }
    .navigationTitle("Settings")
  }
}

struct LayoutEditorView: View {
  let layoutID: String
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss
  @State private var draft: PackLayout?

  var body: some View {
    Form {
      if let draft = Binding($draft) {
        Section("Card size (mm)") {
          number("Width", draft.widthMillimetres)
          number("Height", draft.heightMillimetres)
        }
        Section {
          number("Left edge", draft.gridRegion.x)
          number("Top edge", draft.gridRegion.y)
          number("Width", draft.gridRegion.width)
          number("Height", draft.gridRegion.height)
          number("Border band", draft.borderBand)
        } header: {
          Text("Compartment grid (fraction of card, 0–1)")
        } footer: {
          Text("Objects within the border band of a compartment edge are sent to review instead of being assigned.")
        }
        Section {
          Toggle("Calibrated", isOn: draft.isCalibrated)
        } footer: {
          Text(
            "Turn on only after measuring a real pack and confirming on the station that the grid overlay lines up "
              + "with every compartment. Until then no compartment can be accepted automatically.")
        }
        let errors = draft.wrappedValue.validationErrors()
        if !errors.isEmpty {
          Section("Problems") {
            ForEach(errors, id: \.self) { error in
              Text(String(describing: error)).foregroundStyle(.red)
            }
          }
        }
      }
    }
    .navigationTitle(draft?.displayName ?? "Layout")
    .navigationBarTitleDisplayMode(.inline)
    .toolbar {
      Button("Save") {
        if let draft {
          app.saveLayout(draft)
          dismiss()
        }
      }
      .disabled(draft.map { !$0.validationErrors().isEmpty } ?? true)
    }
    .onAppear {
      if draft == nil { draft = app.layout(id: layoutID) }
    }
  }

  private func number(_ title: String, _ value: Binding<Double>) -> some View {
    LabeledContent(title) {
      TextField(title, value: value, format: .number)
        .keyboardType(.decimalPad)
        .multilineTextAlignment(.trailing)
    }
  }
}
