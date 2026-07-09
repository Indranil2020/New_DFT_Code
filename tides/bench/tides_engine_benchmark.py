#!/usr/bin/env python3
"""
TIDES Engine Piecewise Benchmark — runs all C++ engine profiles and CUDA probes,
parses output into structured JSON for comparison with gpu4pyscf.

Measures per-engine performance on this RTX 3060:
  E1: Tile substrate (GEMM, SpGEMM, Ozaki, CUDA graphs)
  E2: Basis & integrals (NAO, two-center, three-center)
  E3: Grid (rho_build, vmat_build, Poisson, XC)
  E4: Solvers (batched eig, SP2, ChFSI, FOE, OMM)
  E5: SCF (driver, energy, stress)
  E6: Forces & dynamics (analytic forces, XL-BOMD, optimizers)
  E7: Parallel (partitioner, halo exchange)
  E8: Hybrids (D3, ISDF, ACE)
  E9: Verification ladder

Outputs:
  bench/profiling_results/tides_engine_benchmark.json
  bench/profiling_results/tides_engine_benchmark.md
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BUILD_DIR = Path(__file__).parent.parent / "build"
OUT_DIR = Path(__file__).parent / "profiling_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "tides_engine_benchmark.json"
MD_PATH = OUT_DIR / "tides_engine_benchmark.md"

ENGINE_EXES = {
    "E1_tile": "tides_e1_tile_profile",
    "E2_basis": "tides_e2_basis_profile",
    "E3_grid": "tides_e3_grid_profile",
    "E4_solvers": "tides_e4_solvers_profile",
    "E5_scf": "tides_e5_scf_profile",
    "E6_dynamics": "tides_e6_dynamics_profile",
    "E7_parallel": "tides_e7_parallel_profile",
    "E8_hybrids": "tides_e8_hybrids_profile",
    "E9_verification": "tides_e9_verification_profile",
}

CUDA_PROBES = {
    "cuda_gemm": "tides_cuda_gemm_probe",
    "cuda_ozaki_gemm": "tides_ozaki_gemm_probe",
    "cuda_spgemm": "tides_cuda_spgemm_probe",
    "cuda_reduce_f64e": "tides_cuda_reduce_f64e_probe",
    "cuda_graph": "tides_cuda_graph_probe",
}


def run_exe(name, exe_name, timeout=300):
    exe_path = BUILD_DIR / exe_name
    if not exe_path.exists():
        return {"status": "not_built", "exe": exe_name}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(exe_path)], capture_output=True, text=True, timeout=timeout
    )
    wall = time.perf_counter() - t0
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "wall_s": wall,
        "stdout": proc.stdout,
        "stderr": proc.stderr[:3000],
    }


def parse_profile_lines(stdout):
    """Parse structured profile output lines into entries."""
    entries = []
    for line in stdout.split("\n"):
        line = line.strip()
        # Match lines like: "  Kernel  Variant  Size  Time(ms)  Error  Status"
        # Data lines have multiple whitespace-separated fields
        parts = line.split()
        if len(parts) >= 5:
            # Try to identify data lines by looking for time(ms) and error fields
            try:
                # Find the numeric time field (float)
                time_idx = None
                for i, p in enumerate(parts):
                    if re.match(r"^\d+\.?\d*$", p):
                        time_idx = i
                        break
                if time_idx is not None and time_idx + 1 < len(parts):
                    time_ms = float(parts[time_idx])
                    error_str = parts[time_idx + 1]
                    error = float(error_str) if re.match(r"^[\d.eE+-]+$", error_str) else None
                    kernel = " ".join(parts[:time_idx - 2]) if time_idx >= 3 else parts[0]
                    variant = parts[time_idx - 2] if time_idx >= 3 else parts[1]
                    size = parts[time_idx - 1] if time_idx >= 3 else parts[2]
                    status = " ".join(parts[time_idx + 2:])
                    entries.append({
                        "kernel": kernel,
                        "variant": variant,
                        "size": size,
                        "time_ms": time_ms,
                        "error": error,
                        "status": status,
                    })
            except (ValueError, IndexError):
                continue
    return entries


def parse_cuda_gemm(stdout):
    """Parse cuda_gemm_probe output."""
    data = {}
    for token in stdout.split():
        if "=" in token:
            key, val = token.split("=", 1)
            try:
                data[key] = float(val)
            except ValueError:
                data[key] = val
    return data


def parse_cuda_spgemm(stdout):
    """Parse cuda_spgemm_probe output."""
    lines = stdout.strip().split("\n")
    entries = []
    for line in lines:
        data = {}
        for token in line.split():
            if "=" in token:
                key, val = token.split("=", 1)
                try:
                    data[key] = float(val)
                except ValueError:
                    data[key] = val
        if data:
            entries.append(data)
    return entries


def parse_cuda_graph(stdout):
    """Parse cuda_graph_probe output."""
    entries = []
    for line in stdout.strip().split("\n"):
        data = {}
        for token in line.split():
            if "=" in token:
                key, val = token.split("=", 1)
                try:
                    data[key] = float(val)
                except ValueError:
                    data[key] = val
        if data:
            entries.append(data)
    return entries


def main():
    print("=" * 80)
    print("  TIDES Engine Piecewise Benchmark — RTX 3060")
    print("=" * 80)
    print(f"Build dir: {BUILD_DIR}")
    print()

    results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": "NVIDIA GeForce RTX 3060 (12GB, sm_86)",
            "cuda": "12.9",
        },
        "engines": {},
        "cuda_probes": {},
    }

    # ── Engine profiles ──
    print("── Engine Profiles ──")
    for label, exe_name in ENGINE_EXES.items():
        print(f"  {label}: running {exe_name}...")
        r = run_exe(label, exe_name)
        entries = parse_profile_lines(r.get("stdout", "")) if r["status"] == "pass" else []
        r["parsed_entries"] = entries
        results["engines"][label] = r
        status_str = "PASS" if r["status"] == "pass" else f"FAIL (rc={r.get('returncode', '?')})"
        n_entries = len(entries)
        print(f"    -> {status_str} ({r.get('wall_s', 0):.2f}s, {n_entries} entries)")

    # ── CUDA probes ──
    print("\n── CUDA Probes ──")
    for label, exe_name in CUDA_PROBES.items():
        print(f"  {label}: running {exe_name}...")
        r = run_exe(label, exe_name)
        stdout = r.get("stdout", "")
        if label == "cuda_gemm":
            r["parsed"] = parse_cuda_gemm(stdout)
        elif label == "cuda_spgemm":
            r["parsed"] = parse_cuda_spgemm(stdout)
        elif label == "cuda_graph":
            r["parsed"] = parse_cuda_graph(stdout)
        else:
            r["parsed"] = {}
        results["cuda_probes"][label] = r
        status_str = "PASS" if r["status"] == "pass" else f"FAIL (rc={r.get('returncode', '?')})"
        print(f"    -> {status_str} ({r.get('wall_s', 0):.2f}s)")

    # ── Write JSON ──
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON written to: {JSON_PATH}")

    # ── Generate Markdown ──
    generate_markdown(results)
    print(f"Markdown written to: {MD_PATH}")


def generate_markdown(results):
    lines = []
    lines.append("# TIDES Engine Piecewise Benchmark — RTX 3060")
    lines.append("")
    meta = results["metadata"]
    lines.append(f"**Date**: {meta['date']}")
    lines.append(f"**GPU**: {meta['gpu']} | **CUDA**: {meta['cuda']}")
    lines.append("")

    # Engine summary
    lines.append("## Engine Profile Summary")
    lines.append("")
    lines.append("| Engine | Status | Wall (s) | Entries |")
    lines.append("|---|---|---|---|")
    for label, data in results["engines"].items():
        status = data.get("status", "?")
        wall = data.get("wall_s", 0)
        n = len(data.get("parsed_entries", []))
        lines.append(f"| {label} | {status} | {wall:.2f} | {n} |")
    lines.append("")

    # CUDA probes summary
    lines.append("## CUDA Probe Summary")
    lines.append("")
    lines.append("| Probe | Status | Wall (s) | Key Metrics |")
    lines.append("|---|---|---|---|")
    for label, data in results["cuda_probes"].items():
        status = data.get("status", "?")
        wall = data.get("wall_s", 0)
        parsed = data.get("parsed", {})
        metrics = ""
        if label == "cuda_gemm" and isinstance(parsed, dict):
            gflops = parsed.get("planned_kernel_gflops", "?")
            cublas = parsed.get("cublaslt_kernel_gflops", "?")
            metrics = f"planned={gflops} GFLOPS, cuBLASLt={cublas} GFLOPS"
        elif label == "cuda_spgemm" and isinstance(parsed, list) and parsed:
            d = parsed[0]
            metrics = f"n={d.get('n','?')}, gpu_kernel={d.get('gpu_kernel_ms','?')} ms"
        elif label == "cuda_graph" and isinstance(parsed, list) and parsed:
            d = parsed[0]
            metrics = f"graph_replay={d.get('graph_replay_ms','?')} ms"
        lines.append(f"| {label} | {status} | {wall:.2f} | {metrics} |")
    lines.append("")

    # Detailed engine entries
    for label, data in results["engines"].items():
        entries = data.get("parsed_entries", [])
        if not entries:
            continue
        lines.append(f"## {label} — Detailed Entries")
        lines.append("")
        lines.append("| Kernel | Variant | Size | Time (ms) | Error | Status |")
        lines.append("|---|---|---|---|---|---|")
        for e in entries:
            err_str = f"{e['error']:.2e}" if e["error"] is not None else "—"
            lines.append(f"| {e['kernel']} | {e['variant']} | {e['size']} | {e['time_ms']:.3f} | {err_str} | {e['status']} |")
        lines.append("")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
