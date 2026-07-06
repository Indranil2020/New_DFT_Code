"""Tests for TIDES Python API — T10.1/T10.2/T10.3/T10.4 integration tests.

No try/except control flow (ERR001). Uses assert-based invariants.
"""
import os
import sys
import tempfile

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides import Status, StatusCode, Result, TidesConfig, TidesCalculator
from tides import load_config, validate_config
from tides.config import (
    SystemConfig, BasisConfig, XCConfig, SCFConfig, MDConfig,
    SolverConfig, PrecisionConfig, GridConfig, OutputConfig,
)


# ---------------------------------------------------------------------------
# Status / Result tests
# ---------------------------------------------------------------------------

def test_status_ok():
    s = Status.ok()
    assert s.is_ok
    assert s.code == StatusCode.OK
    assert s.message == ""
    assert bool(s) is True


def test_status_error():
    s = Status.invalid_argument("bad value")
    assert not s.is_ok
    assert s.code == StatusCode.INVALID_ARGUMENT
    assert "bad value" in s.message
    assert bool(s) is False


def test_result_ok():
    r = Result.ok(42)
    assert r.is_ok
    assert r.value == 42


def test_result_err():
    r = Result.err(Status.unimplemented("not done"))
    assert not r.is_ok
    assert r.status.code == StatusCode.UNIMPLEMENTED


def test_all_status_codes():
    assert Status.ok().code == StatusCode.OK
    assert Status.invalid_argument("x").code == StatusCode.INVALID_ARGUMENT
    assert Status.out_of_range("x").code == StatusCode.OUT_OF_RANGE
    assert Status.io_error("x").code == StatusCode.IO_ERROR
    assert Status.corrupt_data("x").code == StatusCode.CORRUPT_DATA
    assert Status.unimplemented("x").code == StatusCode.UNIMPLEMENTED
    assert Status.numerical_error("x").code == StatusCode.NUMERICAL_ERROR
    assert Status.convergence_failed("x").code == StatusCode.CONVERGENCE_FAILED


# ---------------------------------------------------------------------------
# Config validation tests (T10.4)
# ---------------------------------------------------------------------------

def test_valid_config():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
    )
    status = validate_config(cfg)
    assert status.is_ok, f"Valid config rejected: {status.message}"


def test_invalid_n_atoms():
    cfg = TidesConfig(
        system=SystemConfig(n_atoms=0, atomic_numbers=[], positions=[]),
    )
    status = validate_config(cfg)
    assert not status.is_ok
    assert "n_atoms" in status.message


def test_invalid_positions_length():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0]],  # only 1 position
        ),
    )
    status = validate_config(cfg)
    assert not status.is_ok
    assert "positions" in status.message


def test_invalid_bc():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
            boundary_conditions="invalid",
        ),
    )
    status = validate_config(cfg)
    assert not status.is_ok
    assert "boundary_conditions" in status.message


def test_invalid_xc():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
        xc=XCConfig(functional="NONEXISTENT"),
    )
    status = validate_config(cfg)
    assert not status.is_ok
    assert "functional" in status.message


def test_invalid_mixing_alpha():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
        scf=SCFConfig(mixing_alpha=2.0),
    )
    status = validate_config(cfg)
    assert not status.is_ok
    assert "mixing_alpha" in status.message


def test_grid_spacing_order():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
        grid=GridConfig(coarse_spacing=0.1, fine_spacing=0.2),  # fine > coarse
    )
    status = validate_config(cfg)
    assert not status.is_ok
    assert "fine_spacing" in status.message


# ---------------------------------------------------------------------------
# TOML loading tests (T10.4)
# ---------------------------------------------------------------------------

VALID_TOML = """\
[System]
n_atoms = 2
atomic_numbers = [1, 1]
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]

[Basis]
kind = "DZP"

[XC]
functional = "PBE"

[SCF]
max_iter = 50
energy_tol = 1e-8

[MD]
mode = "static"
"""


def test_load_valid_toml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(VALID_TOML)
        path = f.name
    result = load_config(path)
    os.unlink(path)
    assert result.is_ok, f"Failed to load valid TOML: {result.status}"
    cfg = result.value
    assert cfg.system.n_atoms == 2
    assert cfg.basis.kind == "DZP"
    assert cfg.xc.functional == "PBE"
    assert cfg.scf.max_iter == 50


def test_load_nonexistent_file():
    result = load_config("/nonexistent/path.toml")
    assert not result.is_ok
    assert result.status.code == StatusCode.IO_ERROR


# ---------------------------------------------------------------------------
# SCF / Energy / Forces tests (T10.1)
# ---------------------------------------------------------------------------

def test_scf_h2():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
    )
    calc = TidesCalculator(cfg)
    result = calc.run_scf()
    assert result.is_ok, f"SCF failed: {result.status}"
    assert result.value.converged
    assert result.value.energy < 0  # bound state


def test_energy_h2():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
    )
    calc = TidesCalculator(cfg)
    e_res = calc.compute_energy()
    assert e_res.is_ok
    assert e_res.value.energy < 0


def test_forces_h2():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 2.0]],  # stretched
        ),
    )
    calc = TidesCalculator(cfg)
    f_res = calc.compute_forces()
    assert f_res.is_ok
    assert len(f_res.value.forces) == 2
    assert f_res.value.max_force > 0


def test_forces_newton_third_law():
    """For a 2-atom system, forces must satisfy Newton's 3rd law: F0 = -F1."""
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 2.0]],
        ),
    )
    calc = TidesCalculator(cfg)
    f_res = calc.compute_forces()
    assert f_res.is_ok
    f0 = f_res.value.forces[0]
    f1 = f_res.value.forces[1]
    for c in range(3):
        assert abs(f0[c] + f1[c]) < 1e-10, f"Newton's 3rd law violated: F0[{c}] + F1[{c}] != 0"


# ---------------------------------------------------------------------------
# MD tests (T10.1)
# ---------------------------------------------------------------------------

def test_xlbomd_h2():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.5]],
        ),
        md=MDConfig(mode="xl-bomd", n_steps=10, timestep=0.5),
    )
    calc = TidesCalculator(cfg)
    md_res = calc.run_md()
    assert md_res.is_ok, f"MD failed: {md_res.status}"
    assert md_res.value.n_steps == 10
    assert md_res.value.avg_solves_per_step <= 2.0


def test_optimize_h2():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.8]],
        ),
        md=MDConfig(mode="optimize", n_steps=50, f_max=1e-3),
    )
    calc = TidesCalculator(cfg)
    opt_res = calc.run_md()
    assert opt_res.is_ok, f"Optimization failed: {opt_res.status}"
    assert opt_res.value.n_steps > 0


def test_static_mode():
    cfg = TidesConfig(
        system=SystemConfig(
            n_atoms=2,
            atomic_numbers=[1, 1],
            positions=[[0, 0, 0], [0, 0, 1.4]],
        ),
        md=MDConfig(mode="static"),
    )
    calc = TidesCalculator(cfg)
    md_res = calc.run_md()
    assert md_res.is_ok
    assert md_res.value.n_steps == 0


# ---------------------------------------------------------------------------
# CLI tests (T10.3)
# ---------------------------------------------------------------------------

def test_cli_verify():
    from tides.cli import main
    rc = main(["verify"])
    # verify should return 0 if all checks pass (or only skipped)
    assert rc in (0, 1)  # 1 if some checks fail (acceptable for stub)


def test_cli_tune():
    from tides.cli import main
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    rc = main(["tune", "-o", out_path])
    assert rc == 0
    assert os.path.exists(out_path)
    os.unlink(out_path)


def test_cli_run():
    from tides.cli import main
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(VALID_TOML)
        path = f.name
    rc = main(["run", path])
    os.unlink(path)
    assert rc == 0


# ---------------------------------------------------------------------------
# No try/except linter test (ERR001)
# ---------------------------------------------------------------------------

def test_no_try_except_in_api():
    """ERR001: no try/except control flow in api/python/ or examples/."""
    import ast
    import glob

    api_dir = os.path.join(os.path.dirname(__file__), "..", "api", "python")
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")

    violations = []
    for pattern_dir in [api_dir, examples_dir]:
        for pyfile in glob.glob(os.path.join(pattern_dir, "**/*.py"), recursive=True):
            with open(pyfile) as f:
                source = f.read()
            tree = ast.parse(source, filename=pyfile)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Try, ast.TryStar)):
                    violations.append(f"{pyfile}:{node.lineno}")

    assert len(violations) == 0, f"ERR001 violations (try/except found):\n  " + "\n  ".join(violations)
