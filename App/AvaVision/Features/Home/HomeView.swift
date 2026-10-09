import AvaVisionCore
import SwiftUI

struct HomeView: View {
  @Environment(AppModel.self) private var app
  @State private var isCheckPresented = false

  var body: some View {
    List {
      Section {
        ModelStatusBanner()
      }

      Section {
        Button {
          isCheckPresented = true
        } label: {
          Label("Check a pack", systemImage: "camera.viewfinder")
            .font(.headline)
        }
        .disabled(!app.isLoaded)
      }

      Section("Learning") {
        NavigationLink {
          BrainView()
        } label: {
          BrainStatusRow()
        }
      }

      Section("Set up") {
        NavigationLink {
          ProfileListView()
        } label: {
          Label("Pack profiles", systemImage: "square.grid.3x3")
        }
        NavigationLink {
          MedicationListView()
        } label: {
          Label("Medications", systemImage: "pills")
        }
      }

      Section("Records") {
        NavigationLink {
          AuditListView()
        } label: {
          Label("Audit trail", systemImage: "list.bullet.clipboard")
        }
        NavigationLink {
          SettingsView()
        } label: {
          Label("Settings", systemImage: "gearshape")
        }
      }

      Section {
        Text(SafetyText.disclaimer)
          .font(.footnote)
          .foregroundStyle(.secondary)
      }
    }
    .navigationTitle("AvaVision")
    .fullScreenCover(isPresented: $isCheckPresented) {
      CheckStartView()
    }
  }
}

enum SafetyText {
  static let disclaimer =
    "AvaVision assists the pharmacist's final check. It never replaces it: every pack must be inspected "
    + "and signed off by a pharmacist, and anything uncertain is always sent to review."
}

struct BrainStatusRow: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    let brain = app.brain
    Label {
      VStack(alignment: .leading, spacing: 2) {
        Text("Brain").font(.subheadline.bold())
        Text(
          "\(brain?.knowledge.exemplars.count ?? 0) pills remembered · "
            + "\(brain?.trustedMedications.count ?? 0) trusted medications"
            + ((brain?.labellingQueue.isEmpty ?? true) ? "" : " · \(brain!.labellingQueue.count) to label")
        )
        .font(.caption)
        .foregroundStyle(.secondary)
      }
    } icon: {
      Image(systemName: "brain").foregroundStyle(.purple)
    }
  }
}

struct ModelStatusBanner: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: symbol)
        .font(.title2)
        .foregroundStyle(color)
      VStack(alignment: .leading, spacing: 2) {
        Text(title).font(.subheadline.bold())
        Text(detail).font(.caption).foregroundStyle(.secondary)
      }
    }
    .padding(.vertical, 4)
  }

  private var title: String {
    switch app.modelState {
    case .loading: "Loading detection model…"
    case .failed: "Detection model failed to load"
    case .ready(let model, _): model.capability.title
    }
  }

  private var detail: String {
    switch app.modelState {
    case .loading:
      "Checking the model file and its manifest."
    case .failed(let message):
      message
    case .ready(let model, .none):
      "\(model.manifest.modelID) failed its integrity check, so automatic checking is off."
    case .ready(let model, _):
      "\(model.manifest.modelID) \(model.manifest.version) · \(model.manifest.stage.rawValue)"
    }
  }

  private var color: Color {
    switch app.modelState {
    case .loading: .gray
    case .failed: .red
    case .ready(let model, _): model.capability.color
    }
  }

  private var symbol: String {
    switch app.modelState {
    case .failed: "exclamationmark.triangle.fill"
    default: "cpu"
    }
  }
}
