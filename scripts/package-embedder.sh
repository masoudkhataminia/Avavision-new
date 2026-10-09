#!/usr/bin/env bash
# Compiles an exported embedder, writes its manifest and installs both into the app bundle sources.
# Usage (on a Mac): scripts/package-embedder.sh build/AvaVisionEmbedder.mlpackage <embedder-id> <version>
set -euo pipefail

cd "$(dirname "$0")/.."
package=${1:?path to .mlpackage}
embedder_id=${2:?embedder id, e.g. avavision-dinov2s-v1}
version=${3:?version, e.g. 2026.10.1}
licence=${4:-"Apache-2.0 (facebook/dinov2-with-registers-small)"}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
xcrun coremlcompiler compile "$package" "$work"
compiled=$(find "$work" -maxdepth 1 -name '*.mlmodelc' | head -1)

target=App/AvaVision/Models
rm -rf "$target/AvaVisionEmbedder.mlmodelc"
cp -R "$compiled" "$target/AvaVisionEmbedder.mlmodelc"
swift run --quiet avavision embedder-manifest "$target/AvaVisionEmbedder.mlmodelc" "$embedder_id" "$version" "$licence" \
  > "$target/AvaVisionEmbedder.json"
echo "Installed $embedder_id $version into $target. Regenerate the project: (cd App && xcodegen generate)"
