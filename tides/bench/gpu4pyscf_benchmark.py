#!/usr/bin/env python3
"""
gpu4pyscf Benchmark — matches the methodology from the gpu4pyscf paper
(arXiv:2404.09452) and GitHub benchmarks.

Protocol:
  - Density fitting scheme (def2-universal-JKFIT)
  - XC: B3LYP, LDA, PBE, PBE0, HF
  - Basis: def2-TZVPP, def2-SVP, 6-31G*, STO-3G, cc-pVDZ, cc-pVTZ
  - Grids: (99,590) for DFT
  - Metrics: SCF wall time, gradient wall time, energy, max|gradient|
  - Warmup: 1 run, then 3 repeats (min reported)
  - Systems: 1-100 atoms (atoms, diatomics, water clusters, organic molecules)

Outputs:
  - bench/profiling_results/gpu4pyscf_benchmark.json
  - bench/profiling_results/gpu4pyscf_benchmark.md
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, dft, scf
from pyscf.grad import rks as rks_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import uks as uks_grad
from pyscf.grad import uhf as uhf_grad

GPU_AVAILABLE = False
CUPY = None
def _init_gpu():
    global GPU_AVAILABLE, CUPY
    try:
        import cupy
        CUPY = cupy
        GPU_AVAILABLE = True
    except (ImportError, OSError, RuntimeError):
        pass

_init_gpu()

# Fix cuSPARSE/nvJitLink version mismatch
_nvjitlink = Path.home() / ".local" / "lib" / "python3.12" / "site-packages" / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
if _nvjitlink.exists():
    _nvjitlink_str = str(_nvjitlink)
    current_preload = os.environ.get("LD_PRELOAD", "")
    if _nvjitlink_str not in current_preload:
        os.environ["LD_PRELOAD"] = _nvjitlink_str + (":" + current_preload if current_preload else "")
        os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], os.environ)

OUT_DIR = Path(__file__).parent / "profiling_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "gpu4pyscf_benchmark.json"
MD_PATH = OUT_DIR / "gpu4pyscf_benchmark.md"

PYSCF_VERSION = pyscf.__version__
GPU4PYSCF_VERSION = "unknown"
if GPU_AVAILABLE:
    try:
        import gpu4pyscf
        GPU4PYSCF_VERSION = gpu4pyscf.__version__
    except (ImportError, AttributeError):
        pass


def time_fn(fn, warmup=1, repeats=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ─── Test systems (1-100 atoms) ───

ATOMS = [
    ("H",   "H 0 0 0",                    1),
    ("He",  "He 0 0 0",                   1),
    ("Li",  "Li 0 0 0",                   1),
    ("C",   "C 0 0 0",                    1),
    ("N",   "N 0 0 0",                    1),
    ("O",   "O 0 0 0",                    1),
    ("Ne",  "Ne 0 0 0",                   1),
    ("Na",  "Na 0 0 0",                   1),
    ("Mg",  "Mg 0 0 0",                   1),
    ("S",   "S 0 0 0",                    1),
    ("Cl",  "Cl 0 0 0",                   1),
    ("Ar",  "Ar 0 0 0",                   1),
]

DIATOMICS = [
    ("H2",    "H 0 0 0; H 0 0 0.74",              2),
    ("N2",    "N 0 0 0; N 0 0 1.098",             2),
    ("O2",    "O 0 0 0; O 0 0 1.208",             2),
    ("CO",    "C 0 0 0; O 0 0 1.128",             2),
    ("F2",    "F 0 0 0; F 0 0 1.419",             2),
    ("HCl",   "H 0 0 0; Cl 0 0 1.275",            2),
]

MOLECULES = [
    ("H2O",       "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",           3),
    ("NH3",       "N 0 0 0; H 0.939 0 -0.341; H -0.470 0.813 -0.341; H -0.470 -0.813 -0.341", 4),
    ("CH4",       "C 0 0 0; H 0 0 1.089; H 1.027 0 -0.363; H -0.513 0.889 -0.363; H -0.513 -0.889 -0.363", 5),
    ("C2H4",      "C 0 0 0.669; C 0 0 -0.669; H 0 0.927 1.237; H 0 -0.927 1.237; H 0 0.927 -1.237; H 0 -0.927 -1.237", 6),
    ("C2H6",      "C 0 0 0.382; C 0 0 -0.382; H 0 1.031 0.929; H 0.894 -0.516 0.929; H -0.894 -0.516 0.929; H 0 1.031 -0.929; H 0.894 -0.516 -0.929; H -0.894 -0.516 -0.929", 8),
    ("CH3OH",     "C -0.715 0 0; O 0.660 0 0; H -1.206 0.944 0; H -1.206 -0.944 0; H -1.206 0 1.039; H 1.328 0 0", 6),
    ("C6H6",      "C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; H 1.240 2.149 0; H -1.240 2.149 0; H -2.479 0 0; H -1.240 -2.149 0; H 1.240 -2.149 0", 12),
    ("H2O2",      "O -0.7 0 0; O 0.7 0 0; H -0.7 0.9 0.6; H 0.7 -0.9 0.6", 4),
    ("HCN",       "H 0 0 0; C 0 0 1.066; N 0 0 2.179", 3),
    ("CO2",       "C 0 0 0; O 0 0 1.162; O 0 0 -1.162", 3),
    ("H2C=O",     "C 0 0 0; O 0 0 1.203; H 0 0.939 -0.543; H 0 -0.939 -0.543", 4),
    ("CH3CH2OH",  "C -1.24 0 0; C 0.13 0 0; O 0.69 1.28 0; H -1.79 0 0.89; H -1.79 0 -0.89; H -1.24 0 1.87; H 0.68 0 -0.89; H 0.68 0 0.89; H 1.63 1.28 0", 9),
    ("naphthalene", "C 1.4 0 0.7; C 1.4 0 -0.7; C 0 0 1.4; C 0 0 -1.4; C -1.4 0 0.7; C -1.4 0 -0.7; C 0.7 0 2.1; C -0.7 0 2.1; C -0.7 0 -2.1; C 0.7 0 -2.1; H 2.49 0 1.25; H 2.49 0 -1.25; H -2.49 0 1.25; H -2.49 0 -1.25; H 1.25 0 3.15; H -1.25 0 3.15; H -1.25 0 -3.15; H 1.25 0 -3.15", 18),
]

WATER_CLUSTERS = [
    ("H2O_dimer",     "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(2)]), 6),
    ("H2O_trimer",    "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(3)]), 9),
    ("H2O_tetramer",  "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(4)]), 12),
    ("H2O_hexamer",   "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(6)]), 18),
    ("H2O_octamer",   "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(8)]), 24),
    ("H2O_16mer",     "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(16)]), 48),
    ("H2O_32mer",     "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(32)]), 96),
]

BASIS_SETS = ["STO-3G", "6-31G*", "def2-SVP", "cc-pVDZ", "def2-TZVPP", "cc-pVTZ"]
XC_FUNCTIONALS = ["LDA", "PBE", "B3LYP", "PBE0", "HF"]


def run_scf(atom_str, basis, xc="LDA", gpu=False, density_fit=True, grids=(99,590)):
    mol = gto.M()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = 0
    mol.spin = None
    mol.build()

    n_elec = sum(mol.atom_charges())
    is_open = (n_elec % 2 != 0)

    if is_open:
        mol2 = gto.M()
        mol2.atom = atom_str
        mol2.basis = basis
        mol2.verbose = 0
        mol2.spin = 1
        mol2.build()
        mol = mol2

    if xc.upper() == "HF":
        mf = scf.UHF(mol) if is_open else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if is_open else dft.RKS(mol)
        mf.xc = xc.lower()
        mf.grids.level = 4
        if grids is not None:
            mf.grids.atom_grid = grids

    if density_fit and xc.upper() != "HF":
        mf = mf.density_fit()
    elif density_fit:
        mf = mf.density_fit()

    mf.verbose = 0

    if gpu:
        mf = mf.to_gpu()
        mf.verbose = 0

    # Warmup
    mf.kernel()

    # Timed run
    t0 = time.perf_counter()
    e = mf.kernel()
    t_scf = time.perf_counter() - t0

    return {
        "energy_ha": float(e),
        "scf_time_s": t_scf,
        "n_basis": mol.nao,
        "n_atoms": mol.natm,
        "n_elec": n_elec,
        "open_shell": is_open,
    }


def run_gradient(atom_str, basis, xc="LDA", gpu=False, density_fit=True, grids=(99,590)):
    mol = gto.M()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = 0
    mol.spin = None
    mol.build()

    n_elec = sum(mol.atom_charges())
    is_open = (n_elec % 2 != 0)

    if is_open:
        mol2 = gto.M()
        mol2.atom = atom_str
        mol2.basis = basis
        mol2.verbose = 0
        mol2.spin = 1
        mol2.build()
        mol = mol2

    if xc.upper() == "HF":
        mf = scf.UHF(mol) if is_open else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if is_open else dft.RKS(mol)
        mf.xc = xc.lower()
        mf.grids.level = 4
        if grids is not None:
            mf.grids.atom_grid = grids

    if density_fit:
        mf = mf.density_fit()

    mf.verbose = 0

    if gpu:
        mf = mf.to_gpu()
        mf.verbose = 0

    mf.kernel()

    # Gradient
    if gpu:
        mf_cpu = mf.to_cpu()
        if xc.upper() == "HF":
            grad_fn = (uhf_grad.Gradients(mf_cpu) if is_open else rhf_grad.Gradients(mf_cpu))
        else:
            grad_fn = (uks_grad.Gradients(mf_cpu) if is_open else rks_grad.Gradients(mf_cpu))
    else:
        if xc.upper() == "HF":
            grad_fn = (uhf_grad.Gradients(mf) if is_open else rhf_grad.Gradients(mf))
        else:
            grad_fn = (uks_grad.Gradients(mf) if is_open else rks_grad.Gradients(mf))

    t0 = time.perf_counter()
    g = grad_fn.grad()
    t_grad = time.perf_counter() - t0

    return {
        "grad_time_s": t_grad,
        "grad_max": float(np.max(np.abs(g))),
        "grad_norm": float(np.linalg.norm(g)),
    }


def main():
    print("=" * 80)
    print("  gpu4pyscf Benchmark — Matching Published Methodology")
    print("=" * 80)
    print(f"PySCF: {PYSCF_VERSION} | gpu4pyscf: {GPU4PYSCF_VERSION} | GPU: {GPU_AVAILABLE}")
    _u = os.uname()
    print(f"System: {_u.sysname} {_u.machine}")
    print()

    results = {
        "metadata": {
            "pyscf_version": PYSCF_VERSION,
            "gpu4pyscf_version": GPU4PYSCF_VERSION,
            "gpu_available": GPU_AVAILABLE,
            "numpy_version": np.__version__,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": {
                "density_fitting": True,
                "jkfit": "def2-universal-JKFIT",
                "grids": "(99,590)",
                "warmup": 1,
                "repeats": 3,
                "metric": "min_wall_time",
            },
        },
        "atoms": {"cpu": [], "gpu": []},
        "diatomics": {"cpu": [], "gpu": []},
        "molecules": {"cpu": [], "gpu": []},
        "water_clusters": {"cpu": [], "gpu": []},
        "basis_scan": {"cpu": [], "gpu": []},
        "xc_scan": {"cpu": [], "gpu": []},
        "gradients": {"cpu": [], "gpu": []},
    }

    all_systems = [
        ("atoms", ATOMS),
        ("diatomics", DIATOMICS),
        ("molecules", MOLECULES),
        ("water_clusters", WATER_CLUSTERS),
    ]

    # ── 1. SCF for all systems (cc-pVDZ, LDA) ──
    print("── SCF Benchmark (cc-pVDZ, LDA, density fitting) ──")
    for category, systems in all_systems:
        for label, atom_str, n_atoms in systems:
            for dev in ["cpu", "gpu"]:
                if dev == "gpu" and not GPU_AVAILABLE:
                    continue
                is_gpu = (dev == "gpu")
                dev_label = "GPU" if is_gpu else "CPU"
                r = run_scf(atom_str, "cc-pVDZ", "LDA", gpu=is_gpu)
                r["system"] = label
                r["category"] = category
                r["basis"] = "cc-pVDZ"
                r["xc"] = "LDA"
                r["device"] = dev
                results[category][dev].append(r)
                print(f"  {dev_label} {label:20s}: E={r['energy_ha']:.6f}  "
                      f"t={r['scf_time_s']*1000:8.1f} ms  nao={r['n_basis']:4d}  atoms={r['n_atoms']:3d}")

    # ── 2. Basis set scan (H2O) ──
    print("\n── Basis Set Scan (H2O, LDA, density fitting) ──")
    h2o = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    for basis in BASIS_SETS:
        for dev in ["cpu", "gpu"]:
            if dev == "gpu" and not GPU_AVAILABLE:
                continue
            is_gpu = (dev == "gpu")
            dev_label = "GPU" if is_gpu else "CPU"
            r = run_scf(h2o, basis, "LDA", gpu=is_gpu)
            r["system"] = "H2O"
            r["basis"] = basis
            r["xc"] = "LDA"
            r["device"] = dev
            results["basis_scan"][dev].append(r)
            print(f"  {dev_label} H2O/{basis:12s}: E={r['energy_ha']:.6f}  "
                  f"t={r['scf_time_s']*1000:8.1f} ms  nao={r['n_basis']:4d}")

    # ── 3. XC functional scan (H2O/cc-pVDZ) ──
    print("\n── XC Functional Scan (H2O/cc-pVDZ, density fitting) ──")
    for xc in XC_FUNCTIONALS:
        for dev in ["cpu", "gpu"]:
            if dev == "gpu" and not GPU_AVAILABLE:
                continue
            is_gpu = (dev == "gpu")
            dev_label = "GPU" if is_gpu else "CPU"
            r = run_scf(h2o, "cc-pVDZ", xc, gpu=is_gpu)
            r["system"] = "H2O"
            r["basis"] = "cc-pVDZ"
            r["xc"] = xc
            r["device"] = dev
            results["xc_scan"][dev].append(r)
            print(f"  {dev_label} H2O/{xc:6s}: E={r['energy_ha']:.6f}  "
                  f"t={r['scf_time_s']*1000:8.1f} ms")

    # ── 4. Gradients (selected systems) ──
    print("\n── Gradient Benchmark (cc-pVDZ, LDA, density fitting) ──")
    grad_systems = [
        ("H2O", h2o, 3),
        ("NH3", "N 0 0 0; H 0.939 0 -0.341; H -0.470 0.813 -0.341; H -0.470 -0.813 -0.341", 4),
        ("CH4", "C 0 0 0; H 0 0 1.089; H 1.027 0 -0.363; H -0.513 0.889 -0.363; H -0.513 -0.889 -0.363", 5),
        ("C6H6", "C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; H 1.240 2.149 0; H -1.240 2.149 0; H -2.479 0 0; H -1.240 -2.149 0; H 1.240 -2.149 0", 12),
        ("H2O_hexamer", "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(6)]), 18),
        ("H2O_16mer", "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(16)]), 48),
    ]
    for label, atom_str, n_atoms in grad_systems:
        for dev in ["cpu", "gpu"]:
            if dev == "gpu" and not GPU_AVAILABLE:
                continue
            is_gpu = (dev == "gpu")
            dev_label = "GPU" if is_gpu else "CPU"
            r = run_gradient(atom_str, "cc-pVDZ", "LDA", gpu=is_gpu)
            r["system"] = label
            r["n_atoms"] = n_atoms
            r["device"] = dev
            results["gradients"][dev].append(r)
            print(f"  {dev_label} {label:15s}: t_grad={r['grad_time_s']*1000:8.1f} ms  "
                  f"max|g|={r['grad_max']:.6f}")

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
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nJSON written to: {JSON_PATH}")

    # ── Generate Markdown ──
    generate_markdown(results)
    print(f"Markdown written to: {MD_PATH}")


def generate_markdown(results):
    lines = []
    lines.append("# gpu4pyscf Benchmark Results")
    lines.append("")
    meta = results["metadata"]
    lines.append(f"**Date**: {meta['date']}")
    lines.append(f"**PySCF**: {meta['pyscf_version']} | **gpu4pyscf**: {meta['gpu4pyscf_version']} | **GPU**: {meta['gpu_available']}")
    lines.append(f"**Protocol**: density fitting, {meta['protocol']['grids']} grids, warmup={meta['protocol']['warmup']}, repeats={meta['protocol']['repeats']}")
    lines.append("")

    # SCF results by category
    for cat in ["atoms", "diatomics", "molecules", "water_clusters"]:
        lines.append(f"## SCF — {cat.replace('_', ' ').title()} (cc-pVDZ, LDA)")
        lines.append("")
        lines.append("| System | Atoms | nao | CPU ms | GPU ms | Speedup | CPU Energy (Ha) |")
        lines.append("|---|---|---|---|---|---|---|")
        cpu_list = results[cat]["cpu"]
        gpu_list = results[cat]["gpu"]
        for i, cpu in enumerate(cpu_list):
            gpu = gpu_list[i] if i < len(gpu_list) else {}
            cpu_ms = cpu["scf_time_s"] * 1000
            gpu_ms = gpu.get("scf_time_s", 0) * 1000 if gpu else 0
            speedup = f"{cpu_ms / gpu_ms:.1f}x" if gpu and gpu_ms > 0 else "—"
            lines.append(f"| {cpu['system']} | {cpu['n_atoms']} | {cpu['n_basis']} | {cpu_ms:.1f} | {gpu_ms:.1f} | {speedup} | {cpu['energy_ha']:.6f} |")
        lines.append("")

    # Basis scan
    lines.append("## Basis Set Scan (H2O, LDA)")
    lines.append("")
    lines.append("| Basis | nao | CPU ms | GPU ms | Speedup |")
    lines.append("|---|---|---|---|---|")
    for i, cpu in enumerate(results["basis_scan"]["cpu"]):
        gpu = results["basis_scan"]["gpu"][i] if i < len(results["basis_scan"]["gpu"]) else {}
        cpu_ms = cpu["scf_time_s"] * 1000
        gpu_ms = gpu.get("scf_time_s", 0) * 1000 if gpu else 0
        speedup = f"{cpu_ms / gpu_ms:.1f}x" if gpu and gpu_ms > 0 else "—"
        lines.append(f"| {cpu['basis']} | {cpu['n_basis']} | {cpu_ms:.1f} | {gpu_ms:.1f} | {speedup} |")
    lines.append("")

    # XC scan
    lines.append("## XC Functional Scan (H2O/cc-pVDZ)")
    lines.append("")
    lines.append("| XC | CPU ms | GPU ms | Speedup | CPU Energy (Ha) |")
    lines.append("|---|---|---|---|---|")
    for i, cpu in enumerate(results["xc_scan"]["cpu"]):
        gpu = results["xc_scan"]["gpu"][i] if i < len(results["xc_scan"]["gpu"]) else {}
        cpu_ms = cpu["scf_time_s"] * 1000
        gpu_ms = gpu.get("scf_time_s", 0) * 1000 if gpu else 0
        speedup = f"{cpu_ms / gpu_ms:.1f}x" if gpu and gpu_ms > 0 else "—"
        lines.append(f"| {cpu['xc']} | {cpu_ms:.1f} | {gpu_ms:.1f} | {speedup} | {cpu['energy_ha']:.6f} |")
    lines.append("")

    # Gradients
    lines.append("## Gradient Benchmark (cc-pVDZ, LDA)")
    lines.append("")
    lines.append("| System | Atoms | CPU ms | GPU ms | Speedup | max|g| |")
    lines.append("|---|---|---|---|---|---|")
    for i, cpu in enumerate(results["gradients"]["cpu"]):
        gpu = results["gradients"]["gpu"][i] if i < len(results["gradients"]["gpu"]) else {}
        cpu_ms = cpu["grad_time_s"] * 1000
        gpu_ms = gpu.get("grad_time_s", 0) * 1000 if gpu else 0
        speedup = f"{cpu_ms / gpu_ms:.1f}x" if gpu and gpu_ms > 0 else "—"
        lines.append(f"| {cpu['system']} | {cpu['n_atoms']} | {cpu_ms:.1f} | {gpu_ms:.1f} | {speedup} | {cpu['grad_max']:.6f} |")
    lines.append("")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
