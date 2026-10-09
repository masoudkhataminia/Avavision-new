import AvaVisionCore
import SwiftUI

struct SignOffView: View {
  let flow: CheckFlowModel
  @Environment(AppModel.self) private var app
  @State private var outcomes: [CompartmentIndex: CompartmentReviewOutcome] = [:]
  @State private var acknowledgedPackFindings = false
  @State private var pharmacist = ""
  @State private var note = ""
  @State private var validationMessage: String?
  @State private var isConfirmingBulk = false
  @State private var isSaving = false

  var body: some View {
    if let result = flow.result {
      let required = Set(result.compartmentsRequiringReview)
      let flagged = result.compartments.filter { required.contains($0.compartment) }
      let countMatched = flagged.filter { $0.status == .countMatched }.map(\.compartment)
      Form {
        Section {
          Text(
            "Inspect every listed compartment in the physical pack and record what you found. "
              + "Spot checks are verified compartments chosen at random so AvaVision's accuracy keeps being "
              + "measured. Verified compartments still form part of your final check."
          )
          .font(.callout)
        }

        Section("Compartments to inspect (\(flagged.count))") {
          if !countMatched.isEmpty {
            Button("Mark all \(countMatched.count) count-OK compartments as inspected") {
              isConfirmingBulk = true
            }
          }
          ForEach(flagged) { verdict in
            VStack(alignment: .leading, spacing: 6) {
              HStack {
                Image(systemName: verdict.status.symbol).foregroundStyle(verdict.status.color)
                Text(flow.layout.label(for: verdict.compartment)).font(.subheadline.weight(.semibold))
                Spacer()
                Text(result.spotChecks.contains(verdict.compartment) ? "Spot check" : verdict.status.title)
                  .font(.caption).foregroundStyle(verdict.status.color)
              }
              if let finding = verdict.findings.first(where: { $0 != .layoutUncalibrated }) ?? verdict.findings.first {
                Text(finding.text(catalog: app.catalog)).font(.caption).foregroundStyle(.secondary)
              }
              Picker("Outcome", selection: outcomeBinding(verdict.compartment)) {
                Text("–").tag(CompartmentReviewOutcome?.none)
                ForEach(CompartmentReviewOutcome.allCases, id: \.self) { outcome in
                  Text(outcome.title).tag(CompartmentReviewOutcome?.some(outcome))
                }
              }
              .pickerStyle(.segmented)
            }
            .padding(.vertical, 2)
          }
        }

        if !result.packFindings.isEmpty {
          Section("Pack-level findings") {
            ForEach(result.packFindings, id: \.self) { finding in
              Text(finding.text).font(.callout)
            }
            Toggle("I have checked these findings", isOn: $acknowledgedPackFindings)
          }
        }

        Section("Pharmacist") {
          TextField("Initials or staff code", text: $pharmacist)
            .textInputAutocapitalization(.characters)
            .autocorrectionDisabled()
          TextField("Note (optional)", text: $note, axis: .vertical)
        }

        Section {
          Button {
            submit(.released, result: result)
          } label: {
            Label(flow.isAwaitingSaveRetry ? "Retry saving" : "Release pack", systemImage: "checkmark.shield")
          }
          .disabled(isSaving)
          Button(role: .destructive) {
            submit(.withheld, result: result)
          } label: {
            Label("Withhold pack", systemImage: "hand.raised")
          }
          .disabled(isSaving || flow.isAwaitingSaveRetry)
        } footer: {
          Text("Your decision is stored in the tamper-evident audit trail and cannot be edited.")
        }
      }
      .navigationTitle("Pharmacist sign-off")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          Button("Result") { flow.showResult() }
            .disabled(isSaving || flow.isAwaitingSaveRetry)
        }
      }
      .alert(
        "Cannot sign off",
        isPresented: Binding(get: { validationMessage != nil }, set: { if !$0 { validationMessage = nil } })
      ) {
        Button("OK", role: .cancel) {}
      } message: {
        Text(validationMessage ?? "")
      }
      .confirmationDialog(
        "Mark count-OK compartments as inspected?", isPresented: $isConfirmingBulk, titleVisibility: .visible
      ) {
        Button("I have inspected each of them") {
          for index in countMatched where outcomes[index] == nil {
            outcomes[index] = .confirmedCorrect
          }
        }
      } message: {
        Text("Only confirm if you have physically checked the medication in every one of these compartments.")
      }
    }
  }

  private func outcomeBinding(_ index: CompartmentIndex) -> Binding<CompartmentReviewOutcome?> {
    Binding(get: { outcomes[index] }, set: { outcomes[index] = $0 })
  }

  private func submit(_ decision: SignOffDecision, result: PackVerificationResult) {
    let reviews = outcomes.map { CompartmentReview(compartment: $0.key, outcome: $0.value) }
      .sorted { $0.compartment < $1.compartment }
    let trimmedNote = note.trimmingCharacters(in: .whitespacesAndNewlines)
    let signOff = PharmacistSignOff(
      pharmacistIdentifier: pharmacist.trimmingCharacters(in: .whitespacesAndNewlines), decision: decision,
      reviews: reviews, acknowledgedPackFindings: acknowledgedPackFindings,
      note: trimmedNote.isEmpty ? nil : trimmedNote)
    if !flow.isAwaitingSaveRetry {
      do {
        try SignOffValidator.validate(signOff, for: result)
      } catch let error as SignOffError {
        validationMessage = error.text(layout: flow.layout)
        return
      } catch {
        validationMessage = error.localizedDescription
        return
      }
    }
    isSaving = true
    Task {
      await flow.complete(with: signOff, app: app)
      isSaving = false
    }
  }
}

struct CompletedView: View {
  let flow: CheckFlowModel
  let onFinish: () -> Void

  var body: some View {
    VStack(spacing: 16) {
      Spacer()
      if let record = flow.savedRecord {
        let released = record.signOff.decision == .released
        Image(systemName: released ? "checkmark.seal.fill" : "hand.raised.fill")
          .font(.system(size: 64))
          .foregroundStyle(released ? .green : .orange)
        Text(released ? "Pack released" : "Pack withheld").font(.title2.bold())
        Text("Signed off by \(record.signOff.pharmacistIdentifier) and saved to the audit trail.")
          .multilineTextAlignment(.center)
          .foregroundStyle(.secondary)
        if let note = flow.learningNote {
          Label(note, systemImage: "brain")
            .font(.callout)
            .foregroundStyle(.purple)
        }
      }
      Spacer()
      Button {
        onFinish()
      } label: {
        Text("Done").frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .controlSize(.large)
    }
    .padding()
    .navigationBarBackButtonHidden()
  }
}
