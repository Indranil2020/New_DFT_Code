# TIDES Tutorial 4: TOML input file and CLI usage
#
# This tutorial demonstrates:
#   1. Writing a TIDES TOML input file
#   2. Loading and validating it via the Python API
#   3. Running the calculation from the config
#
# T10.4 observable: every key documented; invalid input yields precise Status.

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api", "python"))

from tides import load_config, TidesCalculator


H2_TOML = """\
[System]
n_atoms = 2
atomic_numbers = [1, 1]
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
boundary_conditions = "free"

[Basis]
kind = "DZP"
confining_radius = 5.0

[XC]
functional = "PBE"
dispersion = "none"

[SCF]
max_iter = 100
energy_tol = 1e-8
mixing = "pulay"
mixing_alpha = 0.3

[Solver]
regime = "auto"

[MD]
mode = "static"

[Precision]
tile_storage = "bf16"
accumulate = "fp32"
critical_reductions = "f64e"

[Grid]
coarse_spacing = 0.20
fine_spacing = 0.15

[Output]
verbosity = "normal"
"""

INVALID_TOML = """\
[System]
n_atoms = 2
atomic_numbers = [1, 1]
positions = [[0.0, 0.0, 0.0]]
boundary_conditions = "invalid_bc"

[XC]
functional = "NONEXISTENT"
"""


def run_toml_workflow() -> dict:
    """Write a TOML config, load it, validate it, and run SCF."""
    # Write valid config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(H2_TOML)
        valid_path = f.name

    result = load_config(valid_path)
    assert result.is_ok, f"Valid config failed to load: {result.status}"
    cfg = result.value
    print(f"Loaded config: {cfg.system.n_atoms} atoms, basis={cfg.basis.kind}")
    print(f"  XC: {cfg.xc.functional}, dispersion: {cfg.xc.dispersion}")
    print(f"  SCF: max_iter={cfg.scf.max_iter}, tol={cfg.scf.energy_tol}")

    calc = TidesCalculator(cfg)
    scf_res = calc.run_scf()
    assert scf_res.is_ok, f"SCF failed: {scf_res.status}"
    print(f"  SCF energy: {scf_res.value.energy:.10f} Ha")

    # Write invalid config and verify validation catches errors
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(INVALID_TOML)
        invalid_path = f.name

    invalid_result = load_config(invalid_path)
    assert not invalid_result.is_ok, "Invalid config should fail validation"
    print(f"\nInvalid config correctly rejected:")
    print(f"  Status: {invalid_result.status.code.name}")
    # Show first few error lines
    for line in invalid_result.status.message.split("\n")[:3]:
        print(f"  {line}")

    os.unlink(valid_path)
    os.unlink(invalid_path)

    return {
        "valid_loaded": True,
        "scf_energy": scf_res.value.energy,
        "invalid_rejected": not invalid_result.is_ok,
    }


if __name__ == "__main__":
    results = run_toml_workflow()
    print("\nTutorial 4 complete.")

    assert results["valid_loaded"], "Valid config must load"
    assert results["invalid_rejected"], "Invalid config must be rejected"
    print("Integration test: PASSED")
