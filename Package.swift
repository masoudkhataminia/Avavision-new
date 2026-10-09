// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "AvaVisionKit",
  platforms: [.iOS(.v17), .macOS(.v14)],
  products: [
    .library(name: "AvaVisionCore", targets: ["AvaVisionCore"]),
    .library(name: "AvaVisionImaging", targets: ["AvaVisionImaging"]),
    .executable(name: "avavision", targets: ["AvaVisionCLI"]),
  ],
  targets: [
    .target(name: "AvaVisionCore"),
    .target(
      name: "AvaVisionImaging",
      dependencies: ["AvaVisionCore"]
    ),
    .executableTarget(
      name: "AvaVisionCLI",
      dependencies: ["AvaVisionCore"]
    ),
    .testTarget(
      name: "AvaVisionCoreTests",
      dependencies: ["AvaVisionCore"]
    ),
    .testTarget(
      name: "AvaVisionImagingTests",
      dependencies: ["AvaVisionImaging"]
    ),
  ]
)
