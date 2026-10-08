#!/usr/bin/env bash
# Generates the Xcode project, then builds and tests the iOS app on the newest available iPhone simulator.
# Usage: scripts/ios-test.sh            (run from the repository root on a Mac with Xcode and XcodeGen)
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v xcodegen >/dev/null; then
  echo "XcodeGen is required: brew install xcodegen" >&2
  exit 1
fi

(cd App && xcodegen generate --quiet)

simulator_id=$(xcrun simctl list devices available --json | python3 -c '
import json, re, sys
devices = json.load(sys.stdin)["devices"]
candidates = []
for runtime, entries in devices.items():
    match = re.search(r"iOS-(\d+)-(\d+)", runtime)
    if not match:
        continue
    version = (int(match.group(1)), int(match.group(2)))
    for device in entries:
        if device.get("isAvailable") and device["name"].startswith("iPhone"):
            candidates.append((version, device["name"], device["udid"]))
if not candidates:
    sys.exit("No available iPhone simulator found")
candidates.sort()
print(candidates[-1][2])
')
echo "Using simulator ${simulator_id}"

xcodebuild \
  -project App/AvaVision.xcodeproj \
  -scheme AvaVision \
  -destination "platform=iOS Simulator,id=${simulator_id}" \
  -derivedDataPath build/DerivedData \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO \
  test
