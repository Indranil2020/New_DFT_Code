#!/bin/bash
# T8.7: HIP quarterly gate — validates that the HIP/ROCm backend builds and
# passes tests on supported AMD GPU architectures.
#
# This script is intended to run quarterly (or on-demand) to catch regressions
# in the HIP compatibility layer. It requires a ROCm installation.
#
# Usage:
#   ./ci/hip_quarterly_gate.sh [ROCM_PATH]
#
# Exit codes:
#   0 = all HIP tests pass
#   1 = build or test failure
#   77 = ROCm not found (skip)

set -euo pipefail

ROCM_PATH="${1:-${ROCM_PATH:-/opt/rocm}}"
BUILD_DIR="${BUILD_DIR:-build-hip}"

echo "=== T8.7: HIP Quarterly Gate ==="
echo "ROCM_PATH: ${ROCM_PATH}"
echo "BUILD_DIR: ${BUILD_DIR}"

# Check ROCm installation.
if [ ! -d "${ROCM_PATH}" ]; then
  echo "SKIP: ROCm not found at ${ROCM_PATH}"
  exit 77
fi

if [ ! -x "${ROCM_PATH}/bin/hipcc" ]; then
  echo "SKIP: hipcc not found at ${ROCM_PATH}/bin/hipcc"
  exit 77
fi

# Detect AMD GPU.
GPU_ARCHS=""
if command -v rocminfo &>/dev/null; then
  GPU_ARCHS=$(rocminfo | grep "Name:" | grep gfx | awk '{print $2}' | sort -u | tr '\n' ' ')
fi
if [ -z "${GPU_ARCHS}" ]; then
  echo "No AMD GPU detected, using default architectures: gfx906 gfx90a"
  GPU_ARCHS="gfx906 gfx90a"
fi
echo "GPU architectures: ${GPU_ARCHS}"

# Configure.
echo "--- Configuring HIP build ---"
cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTIDES_ENABLE_CUDA=OFF \
  -DTIDES_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="${GPU_ARCHS}" \
  -DROCM_PATH="${ROCM_PATH}"

# Build.
echo "--- Building HIP targets ---"
cmake --build "${BUILD_DIR}" --target tides_hip_tile --parallel 4
cmake --build "${BUILD_DIR}" --target tides_hip_runtime_probe --parallel 4
cmake --build "${BUILD_DIR}" --target tides_hip_gemm_tests --parallel 4
cmake --build "${BUILD_DIR}" --target tides_hip_ozaki_tests --parallel 4
cmake --build "${BUILD_DIR}" --target tides_hip_spgemm_tests --parallel 4

# Test.
echo "--- Running HIP tests ---"
cd "${BUILD_DIR}"
ctest -R hip --output-on-failure -V

echo "=== HIP Quarterly Gate: PASS ==="
exit 0
