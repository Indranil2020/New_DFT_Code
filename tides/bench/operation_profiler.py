#!/usr/bin/env python3
"""
Per-Operation Profiler for TIDES vs PySCF (CPU) vs gpu4pyscf (GPU)

Instruments each SCF step to measure individual operation times:
  - Integral computation (overlap, kinetic, nuclear attraction)
  - Coulomb (J) matrix build
  - XC evaluation (grid integration)
  - Fock matrix assembly
  - Eigendecomposition
  - Density matrix construction
  - DIIS/mixing
  - Energy evaluation

Outputs:
  bench/profiling_results/operation_profile.json
  bench/profiling_results/operation_profile.md
  bench/profiling_results/operation_profile.log
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Fix cuSPARSE/nvJitLink version mismatch
_nvjitlink = Path.home() / ".local" / "lib" / "python3.12" / "site-packages" / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
if _nvjitlink.exists():
    _nvjitlink_str = str(_nvjitlink)
    current_preload = os.environ.get("LD_PRELOAD", "")
    if _nvjitlink_str not in current_preload:
        os.environ["LD_PRELOAD"] = _nvjitlink_str + (":" + current_preload if current_preload else "")
        os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], os.environ)

import pyscf
from pyscf import gto, dft, scf
from pyscf.grad import rks as rks_grad

GPU_AVAILABLE = False
CUPY = None
try:
    import cupy
    CUPY = cupy
    GPU_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    pass

GPU4PYSCF_VERSION = "unknown"
if GPU_AVAILABLE:
    try:
        import gpu4pyscf
        GPU4PYSCF_VERSION = gpu4pyscf.__version__
    except (ImportError, AttributeError):
        pass

BUILD_DIR = Path(__file__).parent.parent / "build"
OUT_DIR = Path(__file__).parent / "profiling_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "operation_profile.json"
MD_PATH = OUT_DIR / "operation_profile.md"
LOG_PATH = OUT_DIR / "operation_profile.log"


class OperationProfiler:
    """Wraps an SCF object and instruments each method with timing."""

    def __init__(self, label, device="cpu"):
        self.label = label
        self.device = device
        self.timings = defaultdict(list)  # op_name -> [times in seconds]
        self._sync = self._make_sync()

    def _make_sync(self):
        if self.device == "gpu" and GPU_AVAILABLE:
            def sync():
                CUPY.cuda.Stream.null.synchronize()
            return sync
        return lambda: None

    def _wrap(self, obj, method_name, display_name=None):
        """Monkey-patch a method on obj to record timing."""
        display_name = display_name or method_name
        if not hasattr(obj, method_name):
            return
        original = getattr(obj, method_name)
        profiler = self

        def timed(*args, **kwargs):
            profiler._sync()
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            profiler._sync()
            t1 = time.perf_counter()
            profiler.timings[display_name].append(t1 - t0)
            return result

        setattr(obj, method_name, timed)

    def instrument_scf(self, mf):
        """Instrument all key SCF methods."""
        self._wrap(mf, "get_hcore", "hcore")
        self._wrap(mf, "get_ovlp", "overlap")
        self._wrap(mf, "get_veff", "veff_coulomb_xc")
        self._wrap(mf, "get_fock", "fock_assembly")
        self._wrap(mf, "get_jk", "jk_matrix")
        self._wrap(mf, "get_j", "j_matrix")
        self._wrap(mf, "eig", "eigendecomposition")
        self._wrap(mf, "make_rdm1", "density_matrix")
        self._wrap(mf, "get_occ", "occupation")

        if hasattr(mf, "with_df") and mf.with_df is not None:
            self._wrap(mf.with_df, "_call_df_jk", "df_jk_integral")

        if hasattr(mf, "grids"):
            self._wrap(mf.grids, "build", "grid_build")

        if hasattr(mf, "xc") and hasattr(mf, "_numint"):
            self._wrap(mf._numint, "nr_rks", "xc_nr_rks")
            self._wrap(mf._numint, "eval_ao", "xc_eval_ao")

    def summary(self):
        """Return per-operation summary stats."""
        result = {}
        for op, times in self.timings.items():
            arr = np.array(times)
            result[op] = {
                "calls": len(arr),
                "total_ms": arr.sum() * 1000,
                "mean_ms": arr.mean() * 1000,
                "max_ms": arr.max() * 1000,
                "min_ms": arr.min() * 1000,
                "std_ms": arr.std() * 1000 if len(arr) > 1 else 0.0,
            }
        return result


def build_mol(atom_str, basis="cc-pVDZ"):
    mol = gto.M()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = 0
    mol.spin = None
    mol.build()
    n_elec = sum(mol.atom_charges())
    is_open = (n_elec % 2 != 0)
    if is_open:
        mol = gto.M(atom=atom_str, basis=basis, verbose=0, spin=1)
        mol.build()
    return mol, is_open


def profile_scf(atom_str, basis, xc="LDA", device="cpu", density_fit=True):
    """Run SCF with per-operation profiling."""
    mol, is_open = build_mol(atom_str, basis)

    if xc.upper() == "HF":
        mf = scf.UHF(mol) if is_open else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if is_open else dft.RKS(mol)
        mf.xc = xc.lower()
        mf.grids.level = 4
        mf.grids.atom_grid = (99, 590)

    if density_fit:
        mf = mf.density_fit()
    mf.verbose = 0

    if device == "gpu":
        mf = mf.to_gpu()
        mf.verbose = 0

    profiler = OperationProfiler(f"{atom_str[:20]}/{basis}/{xc}", device)
    profiler.instrument_scf(mf)

    # Warmup run
    mf.kernel()

    # Clear timings from warmup
    profiler.timings.clear()

    # Timed run
    t0 = time.perf_counter()
    e = mf.kernel()
    t_total = time.perf_counter() - t0

    summary = profiler.summary()
    summary["_total_scf_ms"] = t_total * 1000
    summary["_energy_ha"] = float(e)
    summary["_n_basis"] = mol.nao
    summary["_n_atoms"] = mol.natm
    summary["_n_iterations"] = len(profiler.timings.get("eigendecomposition", []))
    summary["_device"] = device
    summary["_system"] = atom_str[:30]
    summary["_basis"] = basis
    summary["_xc"] = xc

    return summary


def profile_tides_engines():
    """Run TIDES C++ engine profiles and capture per-kernel output."""
    import subprocess

    engine_exes = {
        "E1_tile": "tides_e1_tile_profile",
        "E3_grid": "tides_e3_grid_profile",
        "E4_solvers": "tides_e4_solvers_profile",
        "E5_scf": "tides_e5_scf_profile",
    }
    results = {}
    for label, exe_name in engine_exes.items():
        exe_path = BUILD_DIR / exe_name
        if not exe_path.exists():
            results[label] = {"status": "not_built"}
            continue
        proc = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=120)
        results[label] = {
            "status": "pass" if proc.returncode == 0 else "fail",
            "stdout": proc.stdout,
            "stderr": proc.stderr[:2000],
        }
    return results


def parse_e1_kernels(e1_stdout):
    """Parse E1 profile output to extract per-kernel timing."""
    kernels = []
    for line in e1_stdout.split("\n"):
        line = line.strip()
        if not line or line.startswith("===") or line.startswith("╔") or line.startswith("╚"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        # Status is last token, or "PASS" if last is "(xN.NN)" speedup factor
        status = parts[-1]
        offset = 0
        if status not in ("PASS", "FAIL", "ref"):
            if len(parts) >= 8 and parts[-2] == "PASS":
                status = "PASS"
                offset = 1  # extra token for speedup factor
            else:
                continue
        # Columns from end: status(-1), gflops(-2), maxdiff(-3), wall_ms(-4), kern_ms(-5)
        try:
            kern_ms = float(parts[-5 - offset])
            wall_ms = float(parts[-4 - offset])
            maxdiff = float(parts[-3 - offset])
            gflops = float(parts[-2 - offset])
            kernel = parts[0]
            variant = parts[1]
            size = parts[2]
            kernels.append({
                "kernel": kernel,
                "variant": variant,
                "size": size,
                "kernel_ms": kern_ms,
                "wall_ms": wall_ms,
                "max_diff": maxdiff,
                "gflops": gflops,
                "status": status,
            })
        except (ValueError, IndexError):
            pass
    return kernels


# ─── Test systems ───

PROFILE_SYSTEMS = [
    ("H2O",    "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",                       3),
    ("NH3",    "N 0 0 0; H 0.939 0 -0.341; H -0.470 0.813 -0.341; H -0.470 -0.813 -0.341", 4),
    ("C6H6",   "C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; H 1.240 2.149 0; H -1.240 2.149 0; H -2.479 0 0; H -1.240 -2.149 0; H 1.240 -2.149 0", 12),
    ("H2O_8mer", "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(8)]), 24),
    ("H2O_16mer", "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(16)]), 48),
]


def main():
    print("=" * 80)
    print("  Per-Operation Profiler — TIDES vs PySCF CPU vs gpu4pyscf GPU")
    print("=" * 80)
    print(f"PySCF: {pyscf.__version__} | gpu4pyscf: {GPU4PYSCF_VERSION} | GPU: {GPU_AVAILABLE}")
    print(f"NumPy: {np.__version__}")
    print()

    all_results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pyscf_version": pyscf.__version__,
            "gpu4pyscf_version": GPU4PYSCF_VERSION,
            "gpu_available": GPU_AVAILABLE,
            "gpu": "NVIDIA GeForce RTX 3060 (12GB, sm_86)",
            "cuda": "12.9",
            "numpy_version": np.__version__,
        },
        "scf_profiles": {"cpu": [], "gpu": []},
        "tides_engines": {},
        "tides_kernels": [],
    }

    # ── 1. Per-operation SCF profiling ──
    for label, atom_str, n_atoms in PROFILE_SYSTEMS:
        for device in ["cpu", "gpu"]:
            if device == "gpu" and not GPU_AVAILABLE:
                continue
            dev_label = "GPU" if device == "gpu" else "CPU"
            print(f"\n── Profiling {label} ({dev_label}, cc-pVDZ, LDA) ──")
            r = profile_scf(atom_str, "cc-pVDZ", "LDA", device=device)
            r["_label"] = label
            all_results["scf_profiles"][device].append(r)

            total = r["_total_scf_ms"]
            n_iter = r["_n_iterations"]
            nao = r["_n_basis"]
            print(f"  Total: {total:.1f} ms | {n_iter} iterations | nao={nao}")
            print(f"  {'Operation':<25s} {'Calls':>5s} {'Total ms':>10s} {'Mean ms':>10s} {'Max ms':>10s} {'% SCF':>7s}")
            print(f"  {'-'*25} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
            for op, stats in sorted(r.items()):
                if op.startswith("_"):
                    continue
                pct = stats["total_ms"] / total * 100 if total > 0 else 0
                print(f"  {op:<25s} {stats['calls']:5d} {stats['total_ms']:10.2f} {stats['mean_ms']:10.3f} {stats['max_ms']:10.3f} {pct:6.1f}%")

    # ── 2. TIDES C++ engine profiles ──
    print("\n\n── TIDES C++ Engine Profiles ──")
    tides_engines = profile_tides_engines()
    all_results["tides_engines"] = tides_engines

    for label, data in tides_engines.items():
        status = data.get("status", "?")
        print(f"  {label}: {status}")
        if status == "pass" and label == "E1_tile":
            kernels = parse_e1_kernels(data.get("stdout", ""))
            all_results["tides_kernels"] = kernels
            print(f"    Parsed {len(kernels)} kernel entries")

    # ── Write JSON ──
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nJSON written to: {JSON_PATH}")

    # ── Generate Markdown ──
    generate_markdown(all_results)
    print(f"Markdown written to: {MD_PATH}")


def generate_markdown(results):
    lines = []
    lines.append("# Per-Operation Profiling Report")
    lines.append("")
    meta = results["metadata"]
    lines.append(f"**Date**: {meta['date']}")
    lines.append(f"**GPU**: {meta['gpu']} | **CUDA**: {meta['cuda']}")
    lines.append(f"**PySCF**: {meta['pyscf_version']} | **gpu4pyscf**: {meta['gpu4pyscf_version']}")
    lines.append("")

    # ── Section 1: Per-operation SCF breakdown ──
    lines.append("## 1. Per-Operation SCF Breakdown")
    lines.append("")
    lines.append("### 1.1 PySCF CPU (cc-pVDZ, LDA, density fitting)")
    lines.append("")
    lines.append("| System | Atoms | nao | Iter | Total ms |")
    lines.append("|---|---|---|---|---|")
    for r in results["scf_profiles"]["cpu"]:
        lines.append(f"| {r['_label']} | {r['_n_atoms']} | {r['_n_basis']} | {r['_n_iterations']} | {r['_total_scf_ms']:.1f} |")
    lines.append("")

    # Per-operation detail for CPU
    lines.append("### 1.2 CPU Per-Operation Detail (ms)")
    lines.append("")
    all_ops = set()
    for r in results["scf_profiles"]["cpu"]:
        for k in r:
            if not k.startswith("_"):
                all_ops.add(k)
    op_list = sorted(all_ops)
    header = "| System | " + " | ".join(op_list) + " |"
    sep = "|---|" + "|".join(["---"] * len(op_list)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results["scf_profiles"]["cpu"]:
        row = [r["_label"]]
        for op in op_list:
            if op in r:
                row.append(f"{r[op]['total_ms']:.1f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-operation percentage
    lines.append("### 1.3 CPU Per-Operation % of Total SCF Time")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in results["scf_profiles"]["cpu"]:
        total = r["_total_scf_ms"]
        row = [r["_label"]]
        for op in op_list:
            if op in r and total > 0:
                pct = r[op]["total_ms"] / total * 100
                row.append(f"{pct:.1f}%")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # GPU sections
    if results["scf_profiles"]["gpu"]:
        lines.append("### 1.4 gpu4pyscf GPU (cc-pVDZ, LDA, density fitting)")
        lines.append("")
        lines.append("| System | Atoms | nao | Iter | Total ms |")
        lines.append("|---|---|---|---|---|")
        for r in results["scf_profiles"]["gpu"]:
            lines.append(f"| {r['_label']} | {r['_n_atoms']} | {r['_n_basis']} | {r['_n_iterations']} | {r['_total_scf_ms']:.1f} |")
        lines.append("")

        lines.append("### 1.5 GPU Per-Operation Detail (ms)")
        lines.append("")
        all_ops_gpu = set()
        for r in results["scf_profiles"]["gpu"]:
            for k in r:
                if not k.startswith("_"):
                    all_ops_gpu.add(k)
        op_list_gpu = sorted(all_ops_gpu)
        header_gpu = "| System | " + " | ".join(op_list_gpu) + " |"
        sep_gpu = "|---|" + "|".join(["---"] * len(op_list_gpu)) + "|"
        lines.append(header_gpu)
        lines.append(sep_gpu)
        for r in results["scf_profiles"]["gpu"]:
            row = [r["_label"]]
            for op in op_list_gpu:
                if op in r:
                    row.append(f"{r[op]['total_ms']:.1f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("### 1.6 GPU Per-Operation % of Total SCF Time")
        lines.append("")
        lines.append(header_gpu)
        lines.append(sep_gpu)
        for r in results["scf_profiles"]["gpu"]:
            total = r["_total_scf_ms"]
            row = [r["_label"]]
            for op in op_list_gpu:
                if op in r and total > 0:
                    pct = r[op]["total_ms"] / total * 100
                    row.append(f"{pct:.1f}%")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ── Section 2: CPU vs GPU comparison ──
    lines.append("## 2. CPU vs GPU Per-Operation Comparison")
    lines.append("")
    lines.append("| System | Device | Total ms | Top Operation | Top ms | Top % |")
    lines.append("|---|---|---|---|---|---|")
    for device in ["cpu", "gpu"]:
        dev_label = "CPU" if device == "cpu" else "GPU"
        for r in results["scf_profiles"][device]:
            total = r["_total_scf_ms"]
            ops = {k: v["total_ms"] for k, v in r.items() if not k.startswith("_")}
            if ops:
                top_op = max(ops, key=ops.get)
                top_ms = ops[top_op]
                top_pct = top_ms / total * 100 if total > 0 else 0
                lines.append(f"| {r['_label']} | {dev_label} | {total:.1f} | {top_op} | {top_ms:.1f} | {top_pct:.1f}% |")
    lines.append("")

    # ── Section 3: TIDES E1 Kernel Profile ──
    lines.append("## 3. TIDES E1 Tile Engine — Per-Kernel Profile")
    lines.append("")
    kernels = results.get("tides_kernels", [])
    if kernels:
        lines.append("| Kernel | Variant | Size | Kernel ms | GFLOPS | Status |")
        lines.append("|---|---|---|---|---|---|")
        for k in kernels:
            lines.append(f"| {k['kernel']} | {k['variant']} | {k['size']} | {k['kernel_ms']:.3f} | {k['gflops']:.1f} | {k['status']} |")
    else:
        lines.append("No E1 kernel data available.")
    lines.append("")

    # ── Section 4: Bottleneck Analysis ──
    lines.append("## 4. Bottleneck Analysis")
    lines.append("")

    # CPU bottlenecks
    lines.append("### 4.1 CPU Bottlenecks (largest time consumers)")
    lines.append("")
    cpu_bottlenecks = []
    for r in results["scf_profiles"]["cpu"]:
        for op, stats in r.items():
            if not op.startswith("_"):
                cpu_bottlenecks.append((r["_label"], op, stats["total_ms"], stats["calls"]))
    cpu_bottlenecks.sort(key=lambda x: -x[2])
    lines.append("| System | Operation | Total ms | Calls |")
    lines.append("|---|---|---|---|")
    for sys_name, op, ms, calls in cpu_bottlenecks[:15]:
        lines.append(f"| {sys_name} | {op} | {ms:.1f} | {calls} |")
    lines.append("")

    # GPU bottlenecks
    if results["scf_profiles"]["gpu"]:
        lines.append("### 4.2 GPU Bottlenecks (largest time consumers)")
        lines.append("")
        gpu_bottlenecks = []
        for r in results["scf_profiles"]["gpu"]:
            for op, stats in r.items():
                if not op.startswith("_"):
                    gpu_bottlenecks.append((r["_label"], op, stats["total_ms"], stats["calls"]))
        gpu_bottlenecks.sort(key=lambda x: -x[2])
        lines.append("| System | Operation | Total ms | Calls |")
        lines.append("|---|---|---|---|")
        for sys_name, op, ms, calls in gpu_bottlenecks[:15]:
            lines.append(f"| {sys_name} | {op} | {ms:.1f} | {calls} |")
        lines.append("")

    # TIDES kernel bottlenecks
    if kernels:
        lines.append("### 4.3 TIDES GPU Kernel Bottlenecks (slowest kernels)")
        lines.append("")
        gpu_kernels = [k for k in kernels if k["status"] == "PASS" and k["kernel_ms"] > 0]
        gpu_kernels.sort(key=lambda x: -x["kernel_ms"])
        lines.append("| Kernel | Variant | Size | Kernel ms | GFLOPS |")
        lines.append("|---|---|---|---|---|")
        for k in gpu_kernels[:10]:
            lines.append(f"| {k['kernel']} | {k['variant']} | {k['size']} | {k['kernel_ms']:.3f} | {k['gflops']:.1f} |")
        lines.append("")

    # ── Section 5: Call counts ──
    lines.append("## 5. Per-Operation Call Counts")
    lines.append("")
    lines.append("### 5.1 CPU Call Counts")
    lines.append("")
    lines.append("| System | " + " | ".join(op_list) + " |")
    lines.append("|---|" + "|".join(["---"] * len(op_list)) + "|")
    for r in results["scf_profiles"]["cpu"]:
        row = [r["_label"]]
        for op in op_list:
            if op in r:
                row.append(str(r[op]["calls"]))
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if results["scf_profiles"]["gpu"] and op_list_gpu:
        lines.append("### 5.2 GPU Call Counts")
        lines.append("")
        lines.append("| System | " + " | ".join(op_list_gpu) + " |")
        lines.append("|---|" + "|".join(["---"] * len(op_list_gpu)) + "|")
        for r in results["scf_profiles"]["gpu"]:
            row = [r["_label"]]
            for op in op_list_gpu:
                if op in r:
                    row.append(str(r[op]["calls"]))
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
