import AvaVisionCore
import SwiftUI

struct AuditListView: View {
  @Environment(AppModel.self) private var app
  @State private var records: [CheckRecord] = []
  @State private var verification: AuditVerification?
  @State private var loadError: String?

  var body: some View {
    List {
      Section {
        integrityRow
      }
      Section("Checks (\(records.count))") {
        if records.isEmpty {
          Text("No checks have been signed off yet.").foregroundStyle(.secondary)
        }
        ForEach(records.reversed()) { record in
          NavigationLink {
            AuditDetailView(record: record)
          } label: {
            AuditRow(record: record)
          }
        }
      }
    }
    .navigationTitle("Audit trail")
    .toolbar {
      if let url = app.storage?.auditFileURL, !records.isEmpty {
        ShareLink(item: url) {
          Image(systemName: "square.and.arrow.up")
        }
        .accessibilityLabel("Export audit log")
      }
    }
    .task { await load() }
    .refreshable { await load() }
  }

  @ViewBuilder
  private var integrityRow: some View {
    switch verification {
    case .intact(let entries):
      Label("Audit chain intact (\(entries) entries)", systemImage: "lock.shield.fill")
        .foregroundStyle(.green)
    case .broken(let sequence, let defect):
      Label(
        "Audit chain BROKEN at entry \(sequence) (\(defect.rawValue)). Report this to the pharmacy owner.",
        systemImage: "exclamationmark.shield.fill"
      )
      .foregroundStyle(.red)
    case nil:
      if let loadError {
        Label(loadError, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
      } else {
        ProgressView()
      }
    }
  }

  private func load() async {
    guard let store = app.storage?.audit else {
      loadError = "Audit storage is unavailable."
      return
    }
    do {
      verification = try await store.verify()
      records = try await store.records()
    } catch {
      loadError = "The audit log could not be read: \(error.localizedDescription)"
    }
  }
}

struct AuditRow: View {
  let record: CheckRecord

  var body: some View {
    let released = record.signOff.decision == .released
    VStack(alignment: .leading, spacing: 2) {
      HStack {
        Text(record.profile.reference).font(.body.weight(.semibold))
        Spacer()
        Text(released ? "Released" : "Withheld")
          .font(.caption.weight(.semibold))
          .foregroundStyle(released ? .green : .orange)
      }
      HStack(spacing: 6) {
        Image(systemName: record.result.status.symbol).foregroundStyle(record.result.status.color)
        Text(record.createdAt.formatted(date: .abbreviated, time: .shortened))
        Text("·")
        Text(record.signOff.pharmacistIdentifier)
      }
      .font(.caption)
      .foregroundStyle(.secondary)
    }
  }
}

struct AuditDetailView: View {
  let record: CheckRecord
  @Environment(AppModel.self) private var app

  var body: some View {
    List {
      Section("Decision") {
        LabeledContent("Pack", value: record.profile.reference)
        LabeledContent("Decision", value: record.signOff.decision == .released ? "Released" : "Withheld")
        LabeledContent("Pharmacist", value: record.signOff.pharmacistIdentifier)
        LabeledContent("Signed", value: record.signOff.signedAt.formatted(date: .abbreviated, time: .standard))
        if let note = record.signOff.note {
          Text(note)
        }
      }

      Section("Automatic result") {
        Label(record.result.status.title, systemImage: record.result.status.symbol)
          .foregroundStyle(record.result.status.color)
        LabeledContent("Capability", value: record.result.capability.title)
        LabeledContent("Usable photos", value: "\(record.result.usableFrameCount)")
        ForEach(record.result.packFindings, id: \.self) { finding in
          Text(finding.text).font(.callout)
        }
      }

      Section("Compartments inspected (\(record.signOff.reviews.count))") {
        ForEach(record.signOff.reviews, id: \.compartment) { review in
          VStack(alignment: .leading, spacing: 2) {
            HStack {
              Text(record.layout.label(for: review.compartment))
              Spacer()
              Text(review.outcome.title).foregroundStyle(review.outcome == .unresolved ? .red : .secondary)
            }
            if let verdict = record.result.verdict(for: review.compartment) {
              ForEach(verdict.findings, id: \.self) { finding in
                Text(finding.text(catalog: app.catalog)).font(.caption).foregroundStyle(.secondary)
              }
            }
          }
        }
      }

      Section("Traceability") {
        LabeledContent(
          "Model", value: record.model.map { "\($0.modelID) \($0.version) (\($0.stage.rawValue))" } ?? "None")
        LabeledContent(
          "Layout", value: "\(record.layout.displayName)\(record.layout.isCalibrated ? "" : " · uncalibrated")")
        LabeledContent("App", value: record.appVersion)
        LabeledContent("Device", value: record.deviceIdentifier)
        LabeledContent("Photo hashes", value: "\(record.frameImageSHA256s.count)")
        LabeledContent("Record ID", value: record.id.uuidString).font(.caption)
      }
    }
    .navigationTitle(record.profile.reference)
    .navigationBarTitleDisplayMode(.inline)
  }
}
