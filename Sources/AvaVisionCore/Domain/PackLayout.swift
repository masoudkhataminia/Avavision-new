import Foundation

/// Where a point in pack coordinates falls relative to the compartment grid.
public enum CellLocation: Hashable, Sendable {
  /// Clearly inside one compartment.
  case inside(CompartmentIndex)
  /// Within the border band of the listed compartments; the owner cannot be decided safely.
  case border([CompartmentIndex])
  /// Outside the compartment grid (on the card or off the pack).
  case outsideGrid
}

/// Physical geometry of a multi-dose pack, in normalized pack coordinates
/// (0…1 across the registered pack card, origin top-left).
public struct PackLayout: Codable, Hashable, Sendable, Identifiable {
  public var id: String
  public var displayName: String
  public var rows: Int
  public var columns: Int
  public var rowLabels: [String]
  public var columnLabels: [String]
  /// Outer size of the card, used to sanity-check registration once calibrated.
  public var widthMillimetres: Double
  public var heightMillimetres: Double
  /// Region of the card that contains the compartment grid.
  public var gridRegion: Rect2D
  /// Fraction (0..<0.5) of a cell's width/height near its edges where ownership is ambiguous.
  public var borderBand: Double
  /// Only a layout measured against real packs (B01) may produce accepted results.
  public var isCalibrated: Bool

  public init(
    id: String, displayName: String, rows: Int, columns: Int, rowLabels: [String],
    columnLabels: [String], widthMillimetres: Double, heightMillimetres: Double,
    gridRegion: Rect2D, borderBand: Double, isCalibrated: Bool
  ) {
    self.id = id
    self.displayName = displayName
    self.rows = rows
    self.columns = columns
    self.rowLabels = rowLabels
    self.columnLabels = columnLabels
    self.widthMillimetres = widthMillimetres
    self.heightMillimetres = heightMillimetres
    self.gridRegion = gridRegion
    self.borderBand = borderBand
    self.isCalibrated = isCalibrated
  }

  /// Weekly 7-day × 4-dose-time Webster-pak style layout.
  ///
  /// The dimensions are placeholders until measured on a real pack, so it ships uncalibrated.
  public static let weekly7x4 = PackLayout(
    id: "weekly-7x4",
    displayName: "Weekly pack · 7 days × 4 times",
    rows: 4,
    columns: 7,
    rowLabels: ["Morning", "Midday", "Evening", "Bedtime"],
    columnLabels: (1...7).map { "Day \($0)" },
    widthMillimetres: 250,
    heightMillimetres: 170,
    gridRegion: Rect2D(x: 0.06, y: 0.12, width: 0.88, height: 0.8),
    borderBand: 0.12,
    isCalibrated: false
  )

  public var allCompartments: [CompartmentIndex] {
    (0..<max(rows, 0)).flatMap { row in
      (0..<max(columns, 0)).map { CompartmentIndex(row: row, column: $0) }
    }
  }

  public var aspectRatio: Double { widthMillimetres / heightMillimetres }

  public func contains(_ index: CompartmentIndex) -> Bool {
    (0..<rows).contains(index.row) && (0..<columns).contains(index.column)
  }

  public func label(for index: CompartmentIndex) -> String {
    let column = columnLabels.indices.contains(index.column) ? columnLabels[index.column] : "C\(index.column + 1)"
    let row = rowLabels.indices.contains(index.row) ? rowLabels[index.row] : "R\(index.row + 1)"
    return "\(column) · \(row)"
  }

  /// Rectangle of one compartment in pack coordinates.
  public func cellRect(_ index: CompartmentIndex) -> Rect2D {
    let width = gridRegion.width / Double(columns)
    let height = gridRegion.height / Double(rows)
    return Rect2D(
      x: gridRegion.x + Double(index.column) * width,
      y: gridRegion.y + Double(index.row) * height,
      width: width,
      height: height
    )
  }

  /// Decides which compartment owns a point. Points in a border band are reported as ambiguous
  /// rather than guessed, including bands on the outer edge of the grid.
  public func locate(_ point: Point2D) -> CellLocation {
    guard gridRegion.contains(point) else { return .outsideGrid }
    let fx = (point.x - gridRegion.x) / gridRegion.width * Double(columns)
    let fy = (point.y - gridRegion.y) / gridRegion.height * Double(rows)
    let column = min(columns - 1, max(0, Int(fx.rounded(.down))))
    let row = min(rows - 1, max(0, Int(fy.rounded(.down))))
    let u = fx - Double(column)
    let v = fy - Double(row)

    var columnCandidates = [column]
    var rowCandidates = [row]
    var nearOuterEdge = false
    if u < borderBand {
      if column > 0 { columnCandidates.append(column - 1) } else { nearOuterEdge = true }
    } else if u > 1 - borderBand {
      if column < columns - 1 { columnCandidates.append(column + 1) } else { nearOuterEdge = true }
    }
    if v < borderBand {
      if row > 0 { rowCandidates.append(row - 1) } else { nearOuterEdge = true }
    } else if v > 1 - borderBand {
      if row < rows - 1 { rowCandidates.append(row + 1) } else { nearOuterEdge = true }
    }

    let own = CompartmentIndex(row: row, column: column)
    guard columnCandidates.count > 1 || rowCandidates.count > 1 || nearOuterEdge else {
      return .inside(own)
    }
    let candidates = rowCandidates.flatMap { r in
      columnCandidates.map { CompartmentIndex(row: r, column: $0) }
    }
    return .border(candidates.sorted())
  }

  public enum ValidationError: Error, Hashable, Sendable {
    case emptyIdentifier
    case invalidGridSize
    case labelCountMismatch
    case invalidPhysicalSize
    case gridRegionOutsidePack
    case invalidBorderBand
  }

  /// Returns every problem that makes the layout unusable.
  public func validationErrors() -> [ValidationError] {
    var errors: [ValidationError] = []
    if id.trimmingCharacters(in: .whitespaces).isEmpty { errors.append(.emptyIdentifier) }
    if rows < 1 || columns < 1 { errors.append(.invalidGridSize) }
    if rowLabels.count != rows || columnLabels.count != columns { errors.append(.labelCountMismatch) }
    if !(widthMillimetres > 0 && heightMillimetres > 0) { errors.append(.invalidPhysicalSize) }
    let region = gridRegion
    if !(region.width > 0 && region.height > 0 && region.minX >= 0 && region.minY >= 0
      && region.maxX <= 1 && region.maxY <= 1)
    {
      errors.append(.gridRegionOutsidePack)
    }
    if !(borderBand >= 0 && borderBand < 0.5) { errors.append(.invalidBorderBand) }
    return errors
  }
}
