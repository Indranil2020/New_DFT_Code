"""TIDES file I/O utilities.

Centralized readers and writers for all file formats used by TIDES.
Supports: TOML config, JSON results, XYZ coordinates, and TIDES restart.

No try/except control flow (ERR001). All functions return Result objects
or raise on programmer errors (e.g., missing required arguments).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .status import Status, StatusCode, Result


# ---------------------------------------------------------------------------
# TOML config loading
# ---------------------------------------------------------------------------

def load_toml(path: str) -> Result[dict]:
    """Load a TOML file and return the parsed dict.

    Returns Result[dict] — check .ok before accessing .value.
    """
    if not os.path.isfile(path):
        return Result.err(Status.io_error(f"File not found: {path}"))

    import importlib.util as _ilu
    if _ilu.find_spec("tomllib") is not None:
        import tomllib
    elif _ilu.find_spec("tomli") is not None:
        import tomli as tomllib
    else:
        return Result.err(Status.unimplemented(
            "TIDES requires tomllib (Python 3.11+) or the 'tomli' package. "
            "Install with: pip install tomli"
        ))

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return Result.ok(data)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str, indent: int = 2) -> Result[None]:
    """Save data to a JSON file.

    Returns Result[None] — check .ok before accessing .value.
    """
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        return Result.err(Status.io_error(f"Directory does not exist: {parent}"))

    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

    return Result.ok(None)


def load_json(path: str) -> Result[Any]:
    """Load a JSON file and return the parsed data.

    Returns Result[Any] — check .ok before accessing .value.
    """
    if not os.path.isfile(path):
        return Result.err(Status.io_error(f"File not found: {path}"))

    with open(path, "r") as f:
        data = json.load(f)

    return Result.ok(data)


# ---------------------------------------------------------------------------
# XYZ coordinate files
# ---------------------------------------------------------------------------

# Element symbols for Z=1..118
_ELEMENT_SYMBOLS: list[str] = [
    "X",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

_SYMBOL_TO_Z: dict[str, int] = {sym: z for z, sym in enumerate(_ELEMENT_SYMBOLS) if z > 0}


def atomic_number_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    if 1 <= z < len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z]
    return f"X{z}"


def symbol_to_atomic_number(symbol: str) -> int:
    """Convert element symbol to atomic number. Returns 0 if unknown."""
    return _SYMBOL_TO_Z.get(symbol.strip().capitalize(), 0)


def read_xyz(path: str) -> Result[tuple[list[int], list[list[float]]]]:
    """Read an XYZ file.

    Returns Result[(atomic_numbers, positions_angstrom)].
    Positions are in Angstrom (XYZ file convention).
    """
    if not os.path.isfile(path):
        return Result.err(Status.io_error(f"File not found: {path}"))

    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return Result.err(Status.invalid_argument(
            f"XYZ file has too few lines: {path}"))

    n_atoms: int
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return Result.err(Status.invalid_argument(
            f"Invalid atom count in XYZ file: {lines[0].strip()}"))

    if len(lines) < n_atoms + 2:
        return Result.err(Status.invalid_argument(
            f"XYZ file claims {n_atoms} atoms but has only {len(lines) - 2} coordinate lines"))

    atomic_numbers: list[int] = []
    positions: list[list[float]] = []

    for i in range(n_atoms):
        parts = lines[i + 2].split()
        if len(parts) < 4:
            return Result.err(Status.invalid_argument(
                f"Invalid XYZ line {i + 3}: '{lines[i + 2].strip()}'"))

        symbol = parts[0]
        z = symbol_to_atomic_number(symbol)
        if z == 0:
            return Result.err(Status.invalid_argument(
                f"Unknown element symbol '{symbol}' on line {i + 3}"))

        try:
            x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            return Result.err(Status.invalid_argument(
                f"Invalid coordinates on line {i + 3}: '{lines[i + 2].strip()}'"))

        atomic_numbers.append(z)
        positions.append([x, y, z_coord])

    return Result.ok((atomic_numbers, positions))


def write_xyz(path: str, atomic_numbers: list[int],
              positions_angstrom: list[list[float]],
              comment: str = "") -> Result[None]:
    """Write an XYZ file.

    Positions must be in Angstrom.
    Returns Result[None].
    """
    n = len(atomic_numbers)
    if n != len(positions_angstrom):
        return Result.err(Status.invalid_argument(
            f"Length mismatch: {n} atomic numbers vs {len(positions_angstrom)} positions"))

    lines = [str(n), comment]
    for z, pos in zip(atomic_numbers, positions_angstrom):
        sym = atomic_number_to_symbol(z)
        lines.append(f"{sym:<2s}  {pos[0]:>16.10f}  {pos[1]:>16.10f}  {pos[2]:>16.10f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return Result.ok(None)


# ---------------------------------------------------------------------------
# Dataclass serialization helpers
# ---------------------------------------------------------------------------

def dataclass_to_dict(obj: Any) -> dict:
    """Convert a dataclass instance to a dict, recursively."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj


def save_dataclass_json(obj: Any, path: str, indent: int = 2) -> Result[None]:
    """Save a dataclass instance to a JSON file."""
    return save_json(dataclass_to_dict(obj), path, indent)
