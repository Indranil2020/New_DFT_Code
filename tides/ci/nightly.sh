#!/usr/bin/env bash
# TIDES nightly CI runner (T8.5): CPU + CUDA correctness + profiling.
# Runs on RTX (primary device per 01-hardware-strategy).
# Exit nonzero on any test failure.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== TIDES nightly CI: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo CPU-only) ==="

# Step 1: Configure (CUDA if available, else CPU).
if command -v nvcc &>/dev/null; then
  PRESET="cuda"
else
  PRESET="ci"
fi
echo "=== configure ($PRESET) ==="
cmake --preset "$PRESET"

# Step 2: Build.
echo "=== build ==="
cmake --build --preset "$PRESET" -j "$(nproc)"

# Step 3: Test (all WPs).
echo "=== ctest ==="
ctest --preset "$PRESET" 2>&1 | tee /tmp/tides_ci_test.log

# Step 4: Profile (if the profiler targets exist).
echo "=== profile ==="
BUILD_DIR="../build-$PRESET"
if [ -f "$BUILD_DIR/tides_wp2_profile" ]; then
  MKL_NUM_THREADS=8 "$BUILD_DIR/tides_wp2_profile" > \
    "perf/logs/wp2_cpu_$(date -u +%Y%m%dT%H%M%SZ).jsonl" 2>/dev/null || true
  echo "  WP2 CPU profile saved."
fi
if [ -f "$BUILD_DIR/tides_cuda_gemm_probe" ]; then
  "$BUILD_DIR/tides_cuda_gemm_probe" > \
    "perf/logs/wp1_gpu_$(date -u +%Y%m%dT%H%M%SZ).txt" 2>/dev/null || true
  echo "  WP1 GPU profile saved."
fi

echo "=== CI complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
