# TIDES Profiling & Ledger Directory

This directory holds the performance and task ledgers for the TIDES project.

## Structure

```
tides/perf/
├── README.md           # this file
├── model-ledger.md     # assumption ledger + measured values (per proposal §7)
├── task-ledger.md      # per-task completion status (per DEPENDENCY-GRAPH.md §9)
└── logs/               # JSON-lines and text profiling logs (timestamped)
    ├── wp1_gpu_*.txt   # WP1 CUDA probe output (GEMM, SpGEMM, f64e, graphs)
    └── wp2_cpu_*.jsonl # WP2 CPU profiling (JSON lines)
```

## How to run profiling

### WP2 CPU profiling
```bash
cmake -S tides -B build -DTIDES_ENABLE_CUDA=ON
cmake --build build --target tides_wp2_profile
MKL_NUM_THREADS=8 ./build/tides_wp2_profile > tides/perf/logs/wp2_cpu_$(date -u +%Y%m%dT%H%M%SZ).jsonl
```

### WP1 GPU profiling
```bash
cmake --build build --target tides_cuda_gemm_probe tides_cuda_spgemm_probe \
    tides_cuda_reduce_f64e_probe tides_cuda_graph_probe tides_tile_ops_probe
cd build
LOG=../tides/perf/logs/wp1_gpu_$(date -u +%Y%m%dT%H%M%SZ).txt
{ ./tides_cuda_gemm_probe; ./tides_cuda_spgemm_probe; ./tides_cuda_reduce_f64e_probe; \
  ./tides_cuda_graph_probe; ./tides_tile_ops_probe; } > "$LOG" 2>&1
```

### Full ctest (WP1+WP2 correctness)
```bash
cmake --build build -j
cd build && ctest --output-on-failure
```

## Log format

### WP2 CPU logs (JSON lines)
Each line is one timing measurement:
```json
{"op":"radial_solve","device":"cpu","variant":"cold","repeat":0,"ms":16.88,
 "params":{"n_r":2000,"n_states":3,"err_1s":2e-04},"ts":"2026-07-05T19:18:37Z"}
```
Fields: `op` (operation name), `variant` (cold/warm), `repeat` (0-2),
`ms` (wall time), `params` (operation-specific), `ts` (ISO 8601 UTC).

### WP1 GPU logs (key=value text)
The probes emit a single key=value line per run. Key metrics:
- `planned_kernel_gflops` / `cublas_kernel_gflops` / `planned_vs_cublas_ratio`
- `mixed_abs_bound` (precision error bound)
- `ledger_bound` (SpGEMM filter error bound)
- `trace_abs_error` (f64e accuracy)

## Model revision policy

Per proposal §7: "every deviation >2x triggers a documented model revision."
Deviations are logged in `model-ledger.md` §6. Current deviations:
- T1.2 GEMM 3.6x slower than cuBLASLt (RTX 5050 mobile; re-benchmark on A40/H100)
- Ne LDA 1.1e-2 vs 1e-6 target (uniform grid; needs log grid)
