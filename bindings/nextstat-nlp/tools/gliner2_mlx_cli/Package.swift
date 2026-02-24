// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "gliner2_mlx_cli",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "gliner2_mlx_cli", targets: ["gliner2_mlx_cli"])
    ],
    dependencies: [
        // GLiNER2 on MLX (Metal)
        // Use branch pin because the repo may not publish semver tags consistently.
        .package(url: "https://github.com/MacPaw/Gliner2Swift", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "gliner2_mlx_cli",
            dependencies: [
                .product(name: "GLiNER2Swift", package: "Gliner2Swift"),
            ]
        )
    ]
)
