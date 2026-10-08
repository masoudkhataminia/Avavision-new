import AVFoundation
import AvaVisionCore
import SwiftUI

/// Draws the registered pack outline and compartment grid over an image shown with aspect-fit.
/// When `cellColors` is set, compartments are tinted by result.
struct PackOverlay: View {
  let imageSize: CGSize
  let registration: PackRegistration?
  let layout: PackLayout
  var cellColors: [CompartmentIndex: Color] = [:]

  var body: some View {
    Canvas { context, size in
      guard imageSize.width > 0, imageSize.height > 0, let registration,
        let packToImage = registration.imageToPack.inverse
      else { return }
      let fitted = AVMakeRect(aspectRatio: imageSize, insideRect: CGRect(origin: .zero, size: size))

      func viewPoint(_ point: Point2D) -> CGPoint? {
        packToImage.apply(point).map {
          CGPoint(x: fitted.minX + $0.x * fitted.width, y: fitted.minY + $0.y * fitted.height)
        }
      }

      func polygon(_ rect: Rect2D) -> Path? {
        let corners = [
          Point2D(x: rect.minX, y: rect.minY), Point2D(x: rect.maxX, y: rect.minY),
          Point2D(x: rect.maxX, y: rect.maxY), Point2D(x: rect.minX, y: rect.maxY),
        ].compactMap(viewPoint)
        guard corners.count == 4 else { return nil }
        var path = Path()
        path.addLines(corners)
        path.closeSubpath()
        return path
      }

      for index in layout.allCompartments {
        guard let cell = polygon(layout.cellRect(index)) else { continue }
        if let color = cellColors[index] {
          context.fill(cell, with: .color(color.opacity(0.4)))
        }
        context.stroke(cell, with: .color(.white.opacity(0.8)), lineWidth: 1)
      }
      if let outline = polygon(.unit) {
        context.stroke(outline, with: .color(.yellow), lineWidth: 3)
      }
      let first = CompartmentIndex(row: 0, column: 0)
      if let anchor = viewPoint(layout.cellRect(first).center) {
        context.draw(
          Text(layout.label(for: first)).font(.caption2.bold()).foregroundStyle(.yellow),
          at: anchor)
      }
    }
    .allowsHitTesting(false)
  }
}

/// A captured image with the pack overlay on top.
struct AnnotatedImage: View {
  let image: CGImage
  let registration: PackRegistration?
  let layout: PackLayout
  var cellColors: [CompartmentIndex: Color] = [:]

  var body: some View {
    Image(decorative: image, scale: 1)
      .resizable()
      .aspectRatio(contentMode: .fit)
      .overlay {
        PackOverlay(
          imageSize: CGSize(width: image.width, height: image.height), registration: registration, layout: layout,
          cellColors: cellColors)
      }
  }
}
