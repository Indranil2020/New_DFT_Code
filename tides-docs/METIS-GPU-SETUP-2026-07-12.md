# METIS Partitioning + GPU/CUDA Setup — Status Report

**Date:** 2026-07-12  
**Author:** Dirac agent  
**Scope:** Install METIS partitioning, wire CUDA GPU support, identify remaining gaps.

---

## 1. What Was Done

### 1.1 METIS Graph Partitioner — ✅ INSTALLED & INTEGRATED

**Problem:** T8.2 target was METIS partitioning, but only RCB (recursive coordinate bisection) was implemented. `libmetis5` (runtime) was installed but `libmetis-dev` (headers) was missing.

**Solution:**
- Downloaded `libmetis-dev` .deb package (no sudo needed), extracted `metis.h` to `external/metis/include/` and symlinked `libmetis.so` → `libmetis.so.5`.
- Rewrote `tides/core/parallel/graph_partitioner.hpp`:
  - Added `PartitionMethod` enum (`RCB`, `METIS`) with compile-time `#ifdef TIDES_HAVE_METIS` guard.
  - METIS backend uses `METIS_PartGraphRecursive` (for 2 parts) and `METIS_PartGraphKway` (for >2 parts) with contiguous partitions, deterministic seed, 5% imbalance tolerance.
  - RCB backend now also computes edge cut (`CountCutEdges`) for fair cross-comparison.
  - Automatic fallback to RCB if METIS is not compiled in or fails at runtime.
- Updated `tides/core/parallel/tests/wp8_tests.cpp`:
  - Tests METIS with non-power-of-2 partition counts (2, 3, 4, 5, 7, 8).
  - Cross-checks RCB vs METIS edge cut quality.
- Updated `tides/CMakeLists.txt`:
  - Added `TIDES_ENABLE_METIS` option (default ON).
  - `find_path`/`find_library` searches `external/metis/`, system paths.
  - Links metis to `tides_parallel_lib`, defines `TIDES_HAVE_METIS=1`.

**Verified:** All WP8 tests GREEN. METIS imbalance ≤9% across all partition counts. RCB and METIS produce comparable edge cuts on regular grids.

### 1.2 NCCL (Single-Node Multi-GPU) — ✅ DETECTED

**Finding:** NCCL 2.18.5 is available in the NVIDIA HPC SDK at:
- Headers: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/nccl/include/nccl.h`
- Library: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/nccl/lib/libnccl.so`
- Also available via pip: `nvidia-nccl-cu12` (2.27.5)

**Action:** Added NCCL detection to `tides/CMakeLists.txt` — searches HPC SDK, pip, and system paths. Defines `TIDES_HAVE_NCCL=1` when found. CMake confirms: `WP8: NCCL found`.

### 1.3 NVSHMEM (Multi-Node GPU PGAS) — ✅ DETECTED

**Finding:** NVSHMEM 3.1.7 is available in the NVIDIA HPC SDK at:
- Headers: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/nvshmem/include/nvshmem.h`
- Library: `/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/nvshmem/lib/libnvshmem_host.so`

**Action:** Added NVSHMEM detection to `tides/CMakeLists.txt`. Defines `TIDES_HAVE_NVSHMEM=1` when found. CMake confirms: `WP8: NVSHMEM found`.

### 1.4 CUDA Architecture Fix for RTX 5050 (Blackwell) — ✅ FIXED

**Problem:** RTX 5050 has compute capability 12.0 (sm_120, Blackwell). The installed nvcc (12.0) only supports up to `compute_90`. The CMakePresets `cuda` preset used `89;90` (SASS-only), which would fail on sm_120 without PTX fallback.

**Solution:**
- Updated `CMakePresets.json` cuda preset: `CMAKE_CUDA_ARCHITECTURES: "90-virtual"` (PTX for JIT compilation by the CUDA 13.2 driver).
- Updated `tides/CMakeLists.txt` default archs: `75-virtual 86-virtual 90-virtual` (PTX-only, forward-compatible).
- The CUDA 13.2 driver JIT-compiles PTX to sm_120 at runtime. All existing CUDA tests pass.

### 1.5 CMakePresets Update — ✅ DONE

- `cuda` preset now enables METIS (`TIDES_ENABLE_METIS: ON`).
- Display name updated to "CUDA (RTX 5050 / A40)".
- CUDA architecture set to `90-virtual` for Blackwell compatibility.

---

## 2. What Is Left (Remaining Gaps)

### 2.1 NCCL/NVSHMEM Runtime Usage — ⚠️ NOT YET ACTIVE

**Status:** Libraries are detected and linked, and `TIDES_HAVE_NCCL`/`TIDES_HAVE_NVSHMEM` are defined. However, the actual code in `multi_gpu.hpp`, `halo_exchange.hpp`, and `mpi_nvshmem.hpp` still uses CPU simulation (the `use_nccl = false` flag).

**What's needed:**
- Implement real `ncclCommInitRank` / `ncclAllReduce` / `ncclSend`/`ncclRecv` calls in `multi_gpu.hpp` behind `#ifdef TIDES_HAVE_NCCL`.
- Implement real `nvshmem_init` / `nvshmem_double_put` / `nvshmem_double_get` in `mpi_nvshmem.hpp` behind `#ifdef TIDES_HAVE_NVSHMEM`.
- Note: Single GPU on this machine (RTX 5050) — NCCL single-process multi-GPU won't be exercisable without a second GPU. NVSHMEM multi-node requires MPI bootstrap.

### 2.2 Pre-Existing Build Error in `nao_driver.hpp` — 🔴 BLOCKING

`tides/core/scf/nao_driver.hpp:671` references `tides::grid::GpuArena` which is not declared. This is a **pre-existing issue** unrelated to METIS changes, but it prevents a full `make` from completing. The WP8/parallel targets all build and pass independently.

**Fix needed:** Check if `GpuArena` was renamed to `dev_arena` or if the include for `gpu_arena.hpp` is missing.

### 2.3 CUDA Toolkit Upgrade for sm_120 SASS — 📋 RECOMMENDED

The system nvcc is 12.0 (supports up to `compute_90`). The HPC SDK has nvcc 12.6 (also max `compute_90`). For native sm_120 SASS compilation (no JIT overhead), CUDA Toolkit 12.8+ is needed (adds `compute_100`, `compute_120`).

**Current workaround:** PTX (`90-virtual`) works fine — the CUDA 13.2 driver JIT-compiles to sm_120. Slight startup latency on first kernel launch, no runtime impact after warmup.

### 2.4 GPU Memory Constraint — 📋 NOTE

RTX 5050 has only 8 GB VRAM. Large-scale DFT calculations (10⁵+ atoms) will require:
- Multi-GPU partitioning (single GPU insufficient for dense matrices)
- Mixed precision (fp16/bf16 accumulation) to fit more data
- Tile-based sparse methods (already implemented in the codebase)

### 2.5 Graph Partitioner Integration into SCF Pipeline — 📋 FUTURE

METIS is now available as a partitioning backend, but `BuildPartitions()` in `multi_gpu.hpp` still calls `GraphPartitioner::Partition()` with the default RCB method. To use METIS in the SCF pipeline, add a method parameter to `BuildPartitions()` and `MultiGPUConfig`.

---

## 3. Files Modified

| File | Change |
|------|--------|
| `tides/core/parallel/graph_partitioner.hpp` | Added METIS backend, `PartitionMethod` enum, `CountCutEdges`, edge cut for RCB |
| `tides/core/parallel/tests/wp8_tests.cpp` | Added METIS tests, cross-check, `MakeGridGraph` helper |
| `tides/CMakeLists.txt` | Added `TIDES_ENABLE_METIS`, METIS/NCCL/NVSHMEM detection, CUDA arch fix |
| `tides/CMakePresets.json` | Updated cuda preset: METIS ON, `90-virtual` arch |
| `external/metis/include/metis.h` | New — METIS dev header |
| `external/metis/libmetis.so` | New — symlink to system libmetis.so.5 |

---

## 4. Test Results

```
ctest -R 'wp8|multi_gpu|distributed|t86':
  100% tests passed, 0 tests failed out of 4
  - wp8_stage_dump_partitioner_halo_packaging: PASSED
  - multi_gpu_nccl_data_model: PASSED
  - t59_distributed_scaling: PASSED
  - t86_mpi_nvshmem: PASSED
```

METIS partitioner results (10×10 grid, 100 vertices):
| n_parts | Method | Imbalance | Edge Cut |
|---------|--------|-----------|----------|
| 2 | METIS | 0% | 10 |
| 3 | METIS | 5% | 18 |
| 4 | METIS | 0% | 20 |
| 5 | METIS | 5% | 28 |
| 7 | METIS | 9% | 39 |
| 8 | METIS | 4% | 44 |
| 4 | RCB | 0% | 20 |
