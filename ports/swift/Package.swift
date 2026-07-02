// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CyclePredictor",
    products: [
        .library(name: "CyclePredictor", targets: ["CyclePredictor"]),
    ],
    targets: [
        .target(name: "CyclePredictor"),
        .executableTarget(name: "Validate", dependencies: ["CyclePredictor"]),
    ]
)
