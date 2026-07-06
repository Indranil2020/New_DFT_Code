# TIDES Tutorial 3: XL-BOMD molecular dynamics
#
# This tutorial demonstrates:
#   1. Running XL-BOMD shadow dynamics
#   2. Monitoring energy drift (NVE conservation)
#   3. Checking the ~1 solve/step observable
#
# T10.6 / GB2 observable: NVE drift <= 30 uHa/atom/ps with ~1 solve/step.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api", "python"))

from tides import TidesConfig, SystemConfig, BasisConfig, XCConfig, SCFConfig, MDConfig
from tides import TidesCalculator


def run_xlbomd() -> dict:
    """Run a short XL-BOMD trajectory on H2."""
    config = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],  # slightly displaced
        ),
        basis=BasisConfig(kind="DZP"),
        xc=XCConfig(functional="PBE"),
        md=MDConfig(
            mode="xl-bomd",
            n_steps=20,
            timestep=0.5,  # fs
            thermostat="none",  # NVE
        ),
    )

    calc = TidesCalculator(config)
    md_res = calc.run_md()
    assert md_res.is_ok, f"MD failed: {md_res.status}"

    result = md_res.value
    print(f"XL-BOMD: {result.n_steps} steps")
    print(f"  Final energy: {result.final_energy:.10f} Ha")
    print(f"  Drift: {result.drift_uHa_per_atom_per_ps:.4e} uHa/atom/ps")
    print(f"  Avg solves/step: {result.avg_solves_per_step:.2f}")
    print(f"  Energy history (first 5): {result.energy_history[:5]}")

    return {
        "n_steps": result.n_steps,
        "drift": result.drift_uHa_per_atom_per_ps,
        "avg_solves_per_step": result.avg_solves_per_step,
        "final_energy": result.final_energy,
    }


if __name__ == "__main__":
    results = run_xlbomd()
    print("\nTutorial 3 complete.")

    assert results["avg_solves_per_step"] <= 2.0, "XL-BOMD should use ~1 solve/step"
    print("Integration test: PASSED")
