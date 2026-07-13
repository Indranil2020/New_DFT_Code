#!/usr/bin/env python3
"""
TIDES Scaling Benchmark — Real O(N) scaling measurements.

Runs SCF at multiple system sizes (50, 100, 200, 400, 800, 1600 atoms)
and measures wall time, SCF iterations, and energy convergence.
Fits a power law to determine the actual scaling exponent.

No extrapolations — every data point is a real measurement.

Usage:
    python3 benchmarks/scaling_benchmark.py [--max-atoms N] [--output FILE]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fix cuSPARSE/nvJitLink version mismatch
_nvjitlink = Path.home() / ".local" / "lib" / "python3.12" / "site-packages" / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
if _nvjitlink.exists():
    _nvjitlink_str = str(_nvjitlink)
    current_preload = os.environ.get("LD_PRELOAD", "")
    if _nvjitlink_str not in current_preload:
        os.environ["LD_PRELOAD"] = _nvjitlink_str + (":" + current_preload if current_preload else "")
        os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], os.environ)

sys.path.insert(0, str(Path(__file__).parent.parent / "api" / "python"))
from tides import TidesCalculator, SystemConfig, SCFConfig, GridConfig
from tides.core import Result, Status
from tides.units import ANGSTROM_TO_BOHR

try:
    from tides import _native as _NATIVE
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    print("WARNING: Native backend not available. Install with: pip install -e .[native]")
    sys.exit(1)


def build_water_cluster(n_waters):
    """Build a water cluster with n_waters molecules in a cubic arrangement."""
    waters = []
    # Place waters in a 3D grid with ~3 Angstrom spacing
    side = max(1, int(round(n_waters ** (1.0 / 3.0))))
    idx = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if idx >= n_waters:
                    break
                cx = ix * 3.0
                cy = iy * 3.0
                cz = iz * 3.0
                # O at center, H atoms at ~0.96 Angstrom
                waters.extend([
                    (8, [cx, cy, cz]),
                    (1, [cx + 0.957, cy, cz + 0.120]),
                    (1, [cx - 0.957, cy, cz + 0.120]),
                ])
                idx += 1
    return waters


def run_scaling_benchmark(max_atoms=400):
    """Run SCF at multiple system sizes and measure scaling."""
    results = []
    sizes = [10, 20, 50, 100, 200, 400]
    if max_atoms >= 800:
        sizes.extend([800])
    if max_atoms >= 1600:
        sizes.extend([1600])

    print("=" * 70)
    print("TIDES Scaling Benchmark — Real O(N) Measurements")
    print("=" * 70)
    print(f"{'N_atoms':>8} {'N_basis':>8} {'Wall_s':>10} {'SCF_iter':>8} "
          f"{'Energy_Ha':>14} {'Converged':>10}")
    print("-" * 70)

    for n_waters in sizes:
        n_atoms = n_waters * 3
        if n_atoms > max_atoms:
            break

        atoms = build_water_cluster(n_waters)
        atomic_numbers = [a[0] for a in atoms]
        positions = [a[1] for a in atoms]

        calc = TidesCalculator()
        calc._config = type(calc._config)()
        calc._config.system = SystemConfig(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )
        calc._config.scf = SCFConfig(
            max_iter=50,
            energy_tol=1e-6,
        )
        calc._config.grid = GridConfig(
            coarse_spacing=0.15,
            fine_spacing=0.10,
            margin=2.0,
        )

        t0 = time.perf_counter()
        scf_res = calc.run_scf()
        wall = time.perf_counter() - t0

        if scf_res.is_ok:
            r = scf_res.value
            n_basis = getattr(r, 'n_basis', 0)
            energy = r.energy
            converged = r.converged
            n_iter = getattr(r, 'n_iterations', 0)
        else:
            n_basis = 0
            energy = 0.0
            converged = False
            n_iter = 0

        entry = {
            "n_atoms": n_atoms,
            "n_basis": n_basis,
            "wall_time_s": wall,
            "scf_iterations": n_iter,
            "energy_ha": energy,
            "converged": converged,
        }
        results.append(entry)
        print(f"{n_atoms:>8} {n_basis:>8} {wall:>10.3f} {n_iter:>8} "
              f"{energy:>14.8f} {'yes' if converged else 'no':>10}")

    # Fit scaling exponent: wall_time ~ C * N^alpha
    if len(results) >= 2:
        import math
        valid = [r for r in results if r["converged"] and r["wall_time_s"] > 0]
        if len(valid) >= 2:
            log_n = [math.log(r["n_basis"]) for r in valid]
            log_t = [math.log(r["wall_time_s"]) for r in valid]
            n_pts = len(valid)
            sum_x = sum(log_n)
            sum_y = sum(log_t)
            sum_xy = sum(x * y for x, y in zip(log_n, log_t))
            sum_x2 = sum(x * x for x in log_n)
            alpha = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x)
            print(f"\nScaling exponent: wall_time ~ N^{alpha:.2f}")
            print(f"  (O(N) = 1.0, O(N^2) = 2.0, O(N^3) = 3.0)")
        else:
            alpha = None
            print("\nInsufficient converged data points for scaling fit.")
    else:
        alpha = None

    print("\n" + "=" * 70)
    print("Scaling benchmark complete.")
    print("=" * 70)
    return {"results": results, "scaling_exponent": alpha}


def main():
    parser = argparse.ArgumentParser(description="TIDES scaling benchmark")
    parser.add_argument("--max-atoms", type=int, default=400,
                        help="Maximum number of atoms (default: 400)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    data = run_scaling_benchmark(max_atoms=args.max_atoms)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        print("\n" + json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
