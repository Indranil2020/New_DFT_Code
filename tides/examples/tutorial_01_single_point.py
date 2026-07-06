# TIDES Tutorial 1: Single-point SCF energy of H2
#
# This tutorial demonstrates the basic TIDES workflow:
#   1. Create a TidesConfig
#   2. Run a single-point SCF calculation
#   3. Inspect energy and eigenvalues
#
# This file doubles as an integration test (run with pytest).
#
# T10.6 observable: a new user reproduces this from docs alone in <1 hour.

import sys
import os

# Add the API package to the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api", "python"))

from tides import TidesConfig, SystemConfig, BasisConfig, XCConfig, SCFConfig
from tides import TidesCalculator


def run_h2_scf() -> dict:
    """Run SCF on H2 at 1.4 Angstrom and return results."""
    config = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            boundary_conditions="free",
        ),
        basis=BasisConfig(kind="DZP"),
        xc=XCConfig(functional="PBE", dispersion="none"),
        scf=SCFConfig(max_iter=100, energy_tol=1e-8, mixing="pulay"),
    )

    calc = TidesCalculator(config)
    result = calc.run_scf()

    assert result.is_ok, f"SCF failed: {result.status}"
    scf = result.value
    assert scf.converged, "SCF did not converge"
    assert abs(scf.energy) > 0, "Energy is zero"

    print(f"H2 SCF Energy: {scf.energy:.10f} Ha")
    print(f"Converged in {scf.n_iterations} iterations")
    print(f"Eigenvalues: {scf.eigenvalues}")

    return {
        "energy": scf.energy,
        "converged": scf.converged,
        "n_iterations": scf.n_iterations,
    }


if __name__ == "__main__":
    results = run_h2_scf()
    print("\nTutorial 1 complete.")

    # Integration test assertions
    assert results["converged"], "SCF must converge"
    assert results["energy"] < 0, "H2 energy must be negative (bound)"
    print("Integration test: PASSED")
