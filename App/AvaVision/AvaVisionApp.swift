import SwiftUI

@main
struct AvaVisionApp: App {
  @State private var app = AppModel()

  var body: some Scene {
    WindowGroup {
      RootView()
        .environment(app)
        .task { await app.load() }
    }
  }
}

struct RootView: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    NavigationStack {
      HomeView()
    }
    .alert(
      "Something went wrong",
      isPresented: Binding(get: { app.lastError != nil }, set: { if !$0 { app.lastError = nil } })
    ) {
      Button("OK", role: .cancel) {}
    } message: {
      Text(app.lastError ?? "")
    }
  }
}
