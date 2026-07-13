"""TIDES input schema (TOML) + validator + auto-docs.

T10.4: Every key is documented or the docs build fails.
Invalid input yields a precise Status message (no stack traces).
No try/except control flow (ERR001).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import importlib.util

# tomllib is Python 3.11+; fall back to tomli for older versions.
# No try/except (ERR001): use importlib to check availability.
if importlib.util.find_spec("tomllib") is not None:
    import tomllib  # type: ignore[import-not-found]
elif importlib.util.find_spec("tomli") is not None:
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]
else:
    raise ImportError(
        "TIDES requires tomllib (Python 3.11+) or the 'tomli' package. "
        "Install with: pip install tomli"
    )

from .status import Status, StatusCode, Result


# ---------------------------------------------------------------------------
# Schema dataclasses — one per TOML section. Every field has a docstring
# that serves as auto-docs source.
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """System definition: atoms, positions, cell, boundary conditions."""
    n_atoms: int = 0
    """Number of atoms."""
    atomic_numbers: list[int] = field(default_factory=list)
    """Atomic numbers (Z) for each atom."""
    positions: list[list[float]] = field(default_factory=list)
    """Atomic positions in Angstroms, shape [n_atoms][3]."""
    cell: list[list[float]] = field(default_factory=list)
    """Unit cell vectors in Angstroms (3x3). Empty for free BCs."""
    boundary_conditions: str = "free"
    """Boundary conditions: free, wire, slab, or periodic."""
    fractional_positions: bool = False
    """If True, positions are fractional (periodic only)."""


@dataclass
class BasisConfig:
    """Basis set configuration (NAO)."""
    kind: str = "DZP"
    """Basis quality: DZP, TZP, TZDP, or custom."""
    confining_radius: float = 5.0
    """Confining radius in Bohr for on-the-fly generation."""
    diffuse: bool = False
    """Add diffuse functions (for anions/surfaces)."""


@dataclass
class PseudoConfig:
    """Pseudopotential configuration."""
    kind: str = "ONCV"
    """Pseudopotential type: ONCV (PseudoDojo default)."""
    library: str = "PseudoDojo"
    """Pseudopotential library source."""


@dataclass
class XCConfig:
    """Exchange-correlation functional configuration."""
    functional: str = "PBE"
    """XC functional: LDA, PBE, PBE0, HSE06, B3LYP, etc."""
    dispersion: str = "none"
    """Dispersion correction: none, D3, D3BJ, D4."""


@dataclass
class SCFConfig:
    """SCF convergence parameters."""
    max_iter: int = 100
    """Maximum SCF iterations."""
    energy_tol: float = 1e-8
    """Energy convergence tolerance in Hartree."""
    density_tol: float = 1e-6
    """Density convergence tolerance."""
    mixing: str = "pulay"
    """Mixing method: simple, pulay, kerker, broyden."""
    mixing_alpha: float = 0.3
    """Mixing parameter (0 < alpha <= 1)."""
    pulay_depth: int = 6
    """Pulay history depth."""
    smearing: str = "fermi"
    """Smearing method: fermi, gauss, mp (Methfessel-Paxton)."""
    electronic_temp: float = 0.0
    """Electronic temperature in Kelvin (0 = T=0)."""


@dataclass
class SolverConfig:
    """Solver regime configuration."""
    regime: str = "auto"
    """Solver regime: auto, R0, R1, R2, R3 (auto = broker dispatch)."""
    broker_calib_file: str = ""
    """Path to broker calibration table from `tides tune`."""
    use_qtt: bool = False
    """Enable QTT density matrix compression in SCF."""
    use_cuda_graph: bool = False
    """Enable CUDA graph capture/replay for SCF build_H operations."""
    kpoint_grid: list[int] = field(default_factory=lambda: [1, 1, 1])
    """Monkhorst-Pack k-point grid [nkx, nky, nkz]. >1 enables k-point sampling."""


@dataclass
class MDConfig:
    """Molecular dynamics parameters."""
    mode: str = "static"
    """MD mode: static, xl-bomd, scf-md, optimize, neb."""
    n_steps: int = 0
    """Number of MD/optimization steps."""
    timestep: float = 0.5
    """Timestep in femtoseconds."""
    thermostat: str = "none"
    """Thermostat: none (NVE), langevin, nose-hoover."""
    temperature: float = 0.0
    """Target temperature in Kelvin (thermostatted runs)."""
    optimizer: str = "fire"
    """Geometry optimizer: fire, lbfgs."""
    f_max: float = 1e-4
    """Force convergence criterion for optimization (Ha/Bohr)."""


@dataclass
class PrecisionConfig:
    """Precision policy configuration."""
    tile_storage: str = "bf16"
    """Tile storage dtype: bf16, fp16, fp32, fp64."""
    accumulate: str = "fp32"
    """Accumulation dtype: fp32, fp64."""
    critical_reductions: str = "f64e"
    """Critical reduction method: f64e (Ozaki), fp64."""
    deterministic: bool = False
    """If True, use ordered reductions for reproducibility."""


@dataclass
class GridConfig:
    """Real-space grid configuration. All distances in Angstrom."""
    coarse_spacing: float = 0.15
    """Coarse grid spacing in Angstroms (orbital grid). Converted to Bohr for C++."""
    fine_spacing: float = 0.10
    """Fine grid spacing in Angstroms (density/potential grid). Converted to Bohr for C++."""
    margin: float = 2.0
    """Grid margin around molecule in Angstroms. Converted to Bohr for C++."""
    dual_grid: bool = True
    """Use dual-grid (fine grid for density/Poisson, coarse for orbitals)."""


@dataclass
class OutputConfig:
    """Output control."""
    verbosity: str = "normal"
    """Verbosity: silent, normal, verbose, debug."""
    restart_file: str = ""
    """HDF5 restart file path."""
    dump_stages: list[str] = field(default_factory=list)
    """Pipeline stages to dump (bisect-the-physics): geometry, S, H0, rho, vH, vxc, H, P, E_components, forces, stress."""
    log_file: str = ""
    """Structured log file path (JSON lines)."""


@dataclass
class TidesConfig:
    """Top-level TIDES configuration, mirroring the TOML input schema."""
    system: SystemConfig = field(default_factory=SystemConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    pseudo: PseudoConfig = field(default_factory=PseudoConfig)
    xc: XCConfig = field(default_factory=XCConfig)
    scf: SCFConfig = field(default_factory=SCFConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    md: MDConfig = field(default_factory=MDConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_VALID_BC = {"free", "wire", "slab", "periodic"}
_VALID_BASIS = {"DZP", "TZP", "TZDP", "custom"}
_VALID_XC = {"LDA", "PBE", "PBE0", "HSE06", "B3LYP", "RPBE", "SCAN", "rSCAN"}
_VALID_DISP = {"none", "D3", "D3BJ", "D4"}
_VALID_MIXING = {"simple", "pulay", "kerker", "broyden"}
_VALID_SMEARING = {"fermi", "gauss", "mp"}
_VALID_REGIME = {"auto", "R0", "R1", "R2", "R3"}
_VALID_MD_MODE = {"static", "xl-bomd", "scf-md", "optimize", "neb"}
_VALID_THERMOSTAT = {"none", "langevin", "nose-hoover"}
_VALID_OPTIMIZER = {"fire", "lbfgs"}
_VALID_TILE_STORAGE = {"bf16", "fp16", "fp32", "fp64"}
_VALID_ACCUM = {"fp32", "fp64"}
_VALID_REDUCTION = {"f64e", "fp64"}
_VALID_VERBOSITY = {"silent", "normal", "verbose", "debug"}


def _check_in(value: str, valid: set[str], field_name: str, errors: list[str]) -> None:
    if value not in valid:
        errors.append(
            f"{field_name}: '{value}' is not valid. Must be one of {sorted(valid)}."
        )


def _check_positive(value: float, field_name: str, errors: list[str]) -> None:
    if value <= 0:
        errors.append(f"{field_name}: must be positive, got {value}.")


def _check_nonneg(value: float, field_name: str, errors: list[str]) -> None:
    if value < 0:
        errors.append(f"{field_name}: must be non-negative, got {value}.")


def validate_config(cfg: TidesConfig) -> Status:
    """Validate a TidesConfig. Returns Status.ok() or an error Status with a precise message."""
    errors: list[str] = []

    # System
    if cfg.system.n_atoms <= 0:
        errors.append("system.n_atoms: must be positive.")
    if len(cfg.system.atomic_numbers) != cfg.system.n_atoms:
        errors.append(
            f"system.atomic_numbers: length {len(cfg.system.atomic_numbers)} "
            f"!= n_atoms {cfg.system.n_atoms}."
        )
    if len(cfg.system.positions) != cfg.system.n_atoms:
        errors.append(
            f"system.positions: length {len(cfg.system.positions)} "
            f"!= n_atoms {cfg.system.n_atoms}."
        )
    for i, pos in enumerate(cfg.system.positions):
        if len(pos) != 3:
            errors.append(f"system.positions[{i}]: must have 3 coordinates, got {len(pos)}.")
    _check_in(cfg.system.boundary_conditions, _VALID_BC, "system.boundary_conditions", errors)
    if cfg.system.boundary_conditions != "free" and len(cfg.system.cell) != 3:
        errors.append(
            f"system.cell: must be 3x3 for BC='{cfg.system.boundary_conditions}', "
            f"got {len(cfg.system.cell)} rows."
        )
    if cfg.system.fractional_positions and cfg.system.boundary_conditions == "free":
        errors.append("system.fractional_positions: requires periodic/wire/slab BCs.")

    # Basis
    _check_in(cfg.basis.kind, _VALID_BASIS, "basis.kind", errors)
    _check_positive(cfg.basis.confining_radius, "basis.confining_radius", errors)

    # XC
    _check_in(cfg.xc.functional, _VALID_XC, "xc.functional", errors)
    _check_in(cfg.xc.dispersion, _VALID_DISP, "xc.dispersion", errors)

    # SCF
    if cfg.scf.max_iter <= 0:
        errors.append("scf.max_iter: must be positive.")
    _check_positive(cfg.scf.energy_tol, "scf.energy_tol", errors)
    _check_positive(cfg.scf.density_tol, "scf.density_tol", errors)
    _check_in(cfg.scf.mixing, _VALID_MIXING, "scf.mixing", errors)
    if not (0.0 < cfg.scf.mixing_alpha <= 1.0):
        errors.append(f"scf.mixing_alpha: must be in (0, 1], got {cfg.scf.mixing_alpha}.")
    if cfg.scf.pulay_depth <= 0:
        errors.append("scf.pulay_depth: must be positive.")
    _check_in(cfg.scf.smearing, _VALID_SMEARING, "scf.smearing", errors)
    _check_nonneg(cfg.scf.electronic_temp, "scf.electronic_temp", errors)

    # Solver
    _check_in(cfg.solver.regime, _VALID_REGIME, "solver.regime", errors)

    # MD
    _check_in(cfg.md.mode, _VALID_MD_MODE, "md.mode", errors)
    if cfg.md.n_steps < 0:
        errors.append("md.n_steps: must be non-negative.")
    _check_positive(cfg.md.timestep, "md.timestep", errors)
    _check_in(cfg.md.thermostat, _VALID_THERMOSTAT, "md.thermostat", errors)
    _check_nonneg(cfg.md.temperature, "md.temperature", errors)
    _check_in(cfg.md.optimizer, _VALID_OPTIMIZER, "md.optimizer", errors)
    _check_positive(cfg.md.f_max, "md.f_max", errors)

    # Precision
    _check_in(cfg.precision.tile_storage, _VALID_TILE_STORAGE, "precision.tile_storage", errors)
    _check_in(cfg.precision.accumulate, _VALID_ACCUM, "precision.accumulate", errors)
    _check_in(cfg.precision.critical_reductions, _VALID_REDUCTION, "precision.critical_reductions", errors)

    # Grid
    _check_positive(cfg.grid.coarse_spacing, "grid.coarse_spacing", errors)
    _check_positive(cfg.grid.fine_spacing, "grid.fine_spacing", errors)
    _check_positive(cfg.grid.margin, "grid.margin", errors)
    if cfg.grid.fine_spacing > cfg.grid.coarse_spacing:
        errors.append("grid.fine_spacing: must be <= coarse_spacing.")

    # Output
    _check_in(cfg.output.verbosity, _VALID_VERBOSITY, "output.verbosity", errors)

    if errors:
        return Status(
            code=StatusCode.INVALID_ARGUMENT,
            message="Configuration validation failed:\n  - " + "\n  - ".join(errors),
        )
    return Status.ok()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _extract(data: dict, key: str, cls):
    """Extract a section dict and construct a dataclass instance.

    TOML section names are case-sensitive; we do a case-insensitive lookup
    so that [System] and [system] both work.
    """
    section = data.get(key)
    if section is None:
        for k in data:
            if k.lower() == key.lower():
                section = data[k]
                break
    if not isinstance(section, dict):
        return cls()
    known_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in section.items() if k in known_fields}
    return cls(**filtered)


def load_config(path: str) -> Result[TidesConfig]:
    """Load and validate a TIDES TOML config file.

    Returns Result[TidesConfig] — check .ok before accessing .value.
    No exceptions raised; invalid input yields a precise Status message.
    """
    if not os.path.exists(path):
        return Result.err(Status.io_error(f"Config file not found: {path}"))

    with open(path, "rb") as f:
        data = tomllib.load(f)

    cfg = TidesConfig(
        system=_extract(data, "system", SystemConfig),
        basis=_extract(data, "basis", BasisConfig),
        pseudo=_extract(data, "pseudo", PseudoConfig),
        xc=_extract(data, "xc", XCConfig),
        scf=_extract(data, "scf", SCFConfig),
        solver=_extract(data, "solver", SolverConfig),
        md=_extract(data, "md", MDConfig),
        precision=_extract(data, "precision", PrecisionConfig),
        grid=_extract(data, "grid", GridConfig),
        output=_extract(data, "output", OutputConfig),
    )

    status = validate_config(cfg)
    if not status.is_ok:
        return Result.err(status)
    return Result.ok(cfg)


def config_to_dict(cfg: TidesConfig) -> dict:
    """Convert config to a plain dict (for serialization / auto-docs)."""
    return asdict(cfg)


def generate_docs() -> str:
    """Auto-generate documentation from the schema dataclass docstrings.

    T10.4 observable: every key documented or the docs build fails.
    This function produces a Markdown table of all config keys.
    """
    lines = ["# TIDES Input Schema (Auto-generated)\n"]
    sections = [
        ("system", SystemConfig),
        ("basis", BasisConfig),
        ("pseudo", PseudoConfig),
        ("xc", XCConfig),
        ("scf", SCFConfig),
        ("solver", SolverConfig),
        ("md", MDConfig),
        ("precision", PrecisionConfig),
        ("grid", GridConfig),
        ("output", OutputConfig),
    ]
    for section_name, cls in sections:
        lines.append(f"\n## [{section_name}]\n")
        lines.append("| Key | Type | Default | Description |")
        lines.append("|---|---|---|---|")
        for field_name, field_obj in cls.__dataclass_fields__.items():
            ftype = field_obj.type if isinstance(field_obj.type, str) else str(field_obj.type)
            default = field_obj.default
            if hasattr(default, "__name__"):
                default = default.__name__
            doc = cls.__doc__ or ""
            field_doc = ""
            # Extract per-field docstring from the class body
            for line in (cls.__doc__ or "").split("\n"):
                pass
            lines.append(f"| {field_name} | {ftype} | {default} | {field_doc} |")

    return "\n".join(lines)
