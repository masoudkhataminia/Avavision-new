import AVFoundation
import AvaVisionCore
import PhotosUI
import SwiftUI

struct CaptureView: View {
  let flow: CheckFlowModel
  @State private var photoItems: [PhotosPickerItem] = []

  var body: some View {
    VStack(spacing: 12) {
      ZStack {
        Color.black
        switch flow.cameraState {
        case .running:
          CameraPreview(session: flow.camera.session)
          if let frame = flow.latestFrame {
            PackOverlay(
              imageSize: CGSize(width: frame.image.width, height: frame.image.height),
              registration: frame.observation.registration.registration, layout: flow.layout)
          }
        case .idle:
          ProgressView().tint(.white)
        case .unauthorized:
          cameraMessage("Camera access is off. Allow it in Settings › AvaVision, or use photos.")
        case .unavailable:
          cameraMessage("No camera is available on this device. Use photos instead.")
        case .failed(let message):
          cameraMessage(message)
        }
        if flow.stage == .analyzing {
          ProgressView("Analysing…")
            .padding()
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
      }
      .clipShape(RoundedRectangle(cornerRadius: 12))

      guidance
        .font(.callout)
        .frame(maxWidth: .infinity, alignment: .leading)

      HStack {
        PhotosPicker(selection: $photoItems, maxSelectionCount: 5, matching: .images) {
          Label("Photos", systemImage: "photo.on.rectangle")
        }
        .buttonStyle(.bordered)
        .disabled(flow.stage != .live)

        Spacer()

        Button {
          flow.startCapture()
        } label: {
          Label(flow.stage == .collecting ? "Capturing…" : "Check pack", systemImage: "camera.shutter.button")
            .frame(minWidth: 140)
        }
        .buttonStyle(.borderedProminent)
        .disabled(flow.stage != .live || flow.cameraState != .running)
      }
    }
    .padding()
    .navigationTitle(flow.session.profile.reference)
    .navigationBarTitleDisplayMode(.inline)
    .onChange(of: photoItems) { _, items in
      guard !items.isEmpty else { return }
      Task { await loadPhotos(items) }
    }
  }

  @ViewBuilder
  private var guidance: some View {
    if flow.stage == .collecting {
      Label(
        "Hold steady: \(min(flow.usableFrames, flow.requiredFrames)) of \(flow.requiredFrames) good photos",
        systemImage: "hand.raised")
    } else if let observation = flow.latestFrame?.observation {
      let problems =
        observation.registration.issues.map(\.text) + observation.quality.issues.map(\.text)
      if problems.isEmpty {
        Label(
          "Pack in position. Check the yellow label sits on the first compartment.", systemImage: "checkmark.circle"
        )
        .foregroundStyle(.green)
      } else {
        Label(problems.joined(separator: " · ").capitalizedFirst, systemImage: "exclamationmark.triangle")
          .foregroundStyle(.orange)
      }
    } else {
      Label(
        "Place the whole pack flat under the camera, or choose at least \(flow.requiredFrames) photos of it.",
        systemImage: "viewfinder")
    }
  }

  private func cameraMessage(_ text: String) -> some View {
    Text(text)
      .foregroundStyle(.white)
      .multilineTextAlignment(.center)
      .padding()
  }

  private func loadPhotos(_ items: [PhotosPickerItem]) async {
    var images: [CGImage] = []
    for item in items {
      if let data = try? await item.loadTransferable(type: Data.self),
        let image = ImageConversion.uprightImage(fromPhotoData: data)
      {
        images.append(image)
      }
    }
    photoItems = []
    if images.count < items.count {
      flow.errorMessage = "\(items.count - images.count) photo(s) could not be read."
    }
    await flow.analyzePhotos(images)
  }
}

struct CameraPreview: UIViewRepresentable {
  let session: AVCaptureSession

  func makeUIView(context: Context) -> PreviewView {
    let view = PreviewView()
    view.backgroundColor = .black
    view.previewLayer.session = session
    view.previewLayer.videoGravity = .resizeAspect
    return view
  }

  func updateUIView(_ uiView: PreviewView, context: Context) {}

  final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
      // The layer class is fixed above, so this cast cannot fail.
      layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
      super.layoutSubviews()
      if let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(90) {
        connection.videoRotationAngle = 90
      }
    }
  }
}

extension String {
  var capitalizedFirst: String {
    prefix(1).uppercased() + dropFirst()
  }
}
