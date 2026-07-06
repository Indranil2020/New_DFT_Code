# TIDES Tutorial 2: Forces and geometry optimization
#
# This tutorial demonstrates:
#   1. Computing analytic forces
#   2. Running a geometry optimization with FIRE
#   3. Checking convergence
#
# This file doubles as an integration test.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api", "python"))

from tides import TidesConfig, SystemConfig, BasisConfig, XCConfig, SCFConfig, MDConfig
from tides import TidesCalculator


def run_forces_and_optimize() -> dict:
    """Compute forces on H2 and run a geometry optimization."""
    # First: compute forces at a stretched geometry
    config = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],  # stretched
        ),
        basis=BasisConfig(kind="DZP"),
        xc=XCConfig(functional="PBE"),
    )

    calc = TidesCalculator(config)
    scf_res = calc.run_scf()
    assert scf_res.is_ok, f"SCF failed: {scf_res.status}"

    f_res = calc.compute_forces()
    assert f_res.is_ok, f"Forces failed: {f_res.status}"

    print(f"Forces at R=2.0 Angstrom:")
    for i, f in enumerate(f_res.value.forces):
        print(f"  Atom {i}: [{f[0]:+.6e}, {f[1]:+.6e}, {f[2]:+.6e}] Ha/Bohr")
    print(f"  Max force: {f_res.value.max_force:.6e} Ha/Bohr")

    # Second: geometry optimization
    opt_config = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],  # start from 1.8 Ang
        ),
        basis=BasisConfig(kind="DZP"),
        xc=XCConfig(functional="PBE"),
        md=MDConfig(mode="optimize", n_steps=50, optimizer="fire", f_max=1e-3),
    )

    opt_calc = TidesCalculator(opt_config)
    opt_res = opt_calc.run_md()
    assert opt_res.is_ok, f"Optimization failed: {opt_res.status}"

    print(f"\nOptimization: {opt_res.value.n_steps} steps")
    print(f"  Converged: {opt_res.value.converged}")
    print(f"  Final energy: {opt_res.value.final_energy:.10f} Ha")

    return {
        "max_force": f_res.value.max_force,
        "opt_converged": opt_res.value.converged,
        "opt_n_steps": opt_res.value.n_steps,
        "final_energy": opt_res.value.final_energy,
    }


if __name__ == "__main__":
    results = run_forces_and_optimize()
    print("\nTutorial 2 complete.")

    assert results["max_force"] > 0, "Forces must be non-zero at stretched geometry"
    print("Integration test: PASSED")
