#!/usr/bin/env bash
# TIDES one-command developer setup (T8.5 observable: <= 30 min).
# Configures + builds + tests the CPU foundation in one shot.
# Usage: bash tides/ci/setup.sh [preset]
# Default preset: "ci" (CPU, fast). Use "cuda" for GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PRESET="${1:-ci}"
echo "=== TIDES setup: preset=$PRESET ==="
echo "=== configure ==="
cmake --preset "$PRESET"
echo "=== build ==="
cmake --build --preset "$PRESET" -j "$(nproc)"
echo "=== test ==="
ctest --preset "$PRESET"
echo "=== setup complete ==="
echo "Build dir: $(ls -d ../build-$PRESET 2>/dev/null || echo 'build')"
