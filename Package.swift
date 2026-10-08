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
  dependencies: [
    .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"4.0.0")
  ],
  targets: [
    .target(
      name: "AvaVisionCore",
      dependencies: [.product(name: "Crypto", package: "swift-crypto")]
    ),
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
