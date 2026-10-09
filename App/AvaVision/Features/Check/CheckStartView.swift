import AvaVisionCore
import SwiftUI

/// Full-screen flow: choose a profile, then capture and check the pack.
struct CheckStartView: View {
  @Environment(AppModel.self) private var app
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      List {
        if app.profiles.isEmpty {
          ContentUnavailableView(
            "No pack profiles", systemImage: "square.grid.3x3",
            description: Text("Create a pack profile before checking a pack."))
        }
        ForEach(app.profiles) { profile in
          let issues = app.issues(for: profile)
          if issues.isEmpty {
            NavigationLink(value: profile.id) {
              ProfileRow(profile: profile)
            }
          } else {
            ProfileRow(profile: profile)
              .opacity(0.5)
              .accessibilityHint("Profile is incomplete and cannot be checked.")
          }
        }
      }
      .navigationTitle("Choose pack profile")
      .navigationBarTitleDisplayMode(.inline)
      .navigationDestination(for: UUID.self) { id in
        CheckContainer(profileID: id) { dismiss() }
      }
      .toolbar {
        ToolbarItem(placement: .cancellationAction) {
          Button("Close") { dismiss() }
        }
      }
    }
  }
}

/// Builds the check session once the profile has been chosen.
struct CheckContainer: View {
  let profileID: UUID
  let onFinish: () -> Void
  @Environment(AppModel.self) private var app
  @AppStorage(PackOrientation.storageKey) private var orientation: PackOrientation = .automatic
  @State private var flow: CheckFlowModel?
  @State private var setupError: String?

  var body: some View {
    Group {
      if let flow {
        CheckView(flow: flow, onFinish: onFinish)
      } else if let setupError {
        ContentUnavailableView(
          "Cannot start check", systemImage: "exclamationmark.triangle", description: Text(setupError))
      } else {
        ProgressView()
      }
    }
    .task { prepare() }
  }

  private func prepare() {
    guard flow == nil, setupError == nil else { return }
    guard let profile = app.profile(id: profileID), let layout = app.layout(id: profile.layoutID) else {
      setupError = "The profile or its layout no longer exists."
      return
    }
    do {
      let session = try CheckSession(layout: layout, profile: profile, catalog: app.catalog)
      flow = CheckFlowModel(
        session: session, analyzer: app.analyzer(for: layout, orientation: orientation),
        engine: app.engine(for: layout), model: app.modelState.activeModel, identifier: app.pillIdentifier(),
        brain: app.brain?.summary)
    } catch {
      setupError = "The profile or layout is incomplete. Fix it in Pack profiles or Settings."
    }
  }
}

struct CheckView: View {
  let flow: CheckFlowModel
  let onFinish: () -> Void

  var body: some View {
    Group {
      switch flow.stage {
      case .live, .collecting, .analyzing:
        CaptureView(flow: flow)
      case .result:
        ResultView(flow: flow)
      case .signOff:
        SignOffView(flow: flow)
      case .completed:
        CompletedView(flow: flow, onFinish: onFinish)
      }
    }
    .onAppear { flow.startCamera() }
    .onDisappear { flow.stopCamera() }
    .alert(
      "AvaVision",
      isPresented: Binding(get: { flow.errorMessage != nil }, set: { if !$0 { flow.errorMessage = nil } })
    ) {
      Button("OK", role: .cancel) {}
    } message: {
      Text(flow.errorMessage ?? "")
    }
  }
}
