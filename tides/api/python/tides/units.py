"""TIDES physical constants and unit conversions.

Single source of truth for all unit conversions in the Python API.
All user-facing config values use SI-friendly units (Angstrom, fs, eV).
The C++ engine uses atomic units (Bohr, a.u., Hartree).

Conversions go one direction: user units → atomic units (for C++).
No hardcoded conversion factors anywhere else in the codebase.

No try/except control flow (ERR001).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Fundamental physical constants (CODATA 2018)
# ---------------------------------------------------------------------------

# Length
BOHR_TO_ANGSTROM: float = 0.529177210903
"""1 Bohr in Angstroms."""
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
"""1 Angstrom in Bohr (= 1.8897261254535)."""

# Energy
HARTREE_TO_EV: float = 27.211386245988
"""1 Hartree in eV."""
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV
"""1 eV in Hartree."""
HARTREE_TO_KJ_MOL: float = 2625.4996394798
"""1 Hartree in kJ/mol."""
KJ_MOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KJ_MOL
"""1 kJ/mol in Hartree."""
HARTREE_TO_KCAL_MOL: float = 627.5094740631
"""1 Hartree in kcal/mol."""
KCAL_MOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL_MOL
"""1 kcal/mol in Hartree."""
HARTREE_TO_UHA: float = 1.0e6
"""1 Hartree in microHartree (µHa)."""
UHA_TO_HARTREE: float = 1.0 / HARTREE_TO_UHA
"""1 µHa in Hartree."""

# Time
FS_TO_AU: float = 41.3413733366
"""1 femtosecond in atomic units of time (ℏ/E_h)."""
AU_TO_FS: float = 1.0 / FS_TO_AU
"""1 a.u. of time in femtoseconds."""
FS_TO_PS: float = 1.0e-3
"""1 femtosecond in picoseconds."""
PS_TO_FS: float = 1.0 / FS_TO_PS
"""1 picosecond in femtoseconds."""

# Mass
AMU_TO_AU: float = 1822.888486209
"""1 atomic mass unit in electron-mass units (a.u.)."""
AU_TO_AMU: float = 1.0 / AMU_TO_AU
"""1 a.u. mass in atomic mass units."""

# Force
HA_BOHR_TO_EV_ANGSTROM: float = HARTREE_TO_EV / BOHR_TO_ANGSTROM
"""1 Ha/Bohr in eV/Angstrom."""
EV_ANGSTROM_TO_HA_BOHR: float = 1.0 / HA_BOHR_TO_EV_ANGSTROM
"""1 eV/Angstrom in Ha/Bohr."""

# Pressure
HA_BOHR3_TO_GPA: float = 294210.2648438959
"""1 Ha/Bohr^3 in GPa."""
GPA_TO_HA_BOHR3: float = 1.0 / HA_BOHR3_TO_GPA
"""1 GPa in Ha/Bohr^3."""


# ---------------------------------------------------------------------------
# Atomic mass table (in atomic mass units)
# Source: IUPAC 2021 standard atomic weights (most abundant isotope)
# ---------------------------------------------------------------------------

ATOMIC_MASS_AMU: dict[int, float] = {
    1: 1.00794,    # H
    2: 4.002602,   # He
    3: 6.941,      # Li
    4: 9.012182,   # Be
    5: 10.811,     # B
    6: 12.0107,    # C
    7: 14.0067,    # N
    8: 15.9994,    # O
    9: 18.9984032, # F
    10: 20.1797,   # Ne
    11: 22.98976928,# Na
    12: 24.3050,   # Mg
    13: 26.9815386,# Al
    14: 28.0855,   # Si
    15: 30.973762, # P
    16: 32.065,    # S
    17: 35.453,    # Cl
    18: 39.948,    # Ar
    19: 39.0983,   # K
    20: 40.078,    # Ca
    21: 44.955912, # Sc
    22: 47.867,    # Ti
    23: 50.9415,   # V
    24: 51.9961,   # Cr
    25: 54.938045, # Mn
    26: 55.845,    # Fe
    27: 58.933195, # Co
    28: 58.6934,   # Ni
    29: 63.546,    # Cu
    30: 65.38,     # Zn
    31: 69.723,    # Ga
    32: 72.64,     # Ge
    33: 74.92160,  # As
    34: 78.96,     # Se
    35: 79.904,    # Br
    36: 83.798,    # Kr
    37: 85.4678,   # Rb
    38: 87.62,     # Sr
    39: 88.90585,  # Y
    40: 91.224,    # Zr
    41: 92.90628,  # Nb
    42: 95.96,     # Mo
    43: 98.0,      # Tc
    44: 101.07,    # Ru
    45: 102.90550, # Rh
    46: 106.42,    # Pd
    47: 107.8682,  # Ag
    48: 112.411,   # Cd
    49: 114.818,   # In
    50: 118.71,    # Sn
    51: 121.760,   # Sb
    52: 127.60,    # Te
    53: 126.90447, # I
    54: 131.293,   # Xe
    55: 132.9054519,# Cs
    56: 137.327,   # Ba
    57: 138.90547, # La
    58: 140.116,   # Ce
    59: 140.90765, # Pr
    60: 144.242,   # Nd
    61: 145.0,     # Pm
    62: 150.36,    # Sm
    63: 151.964,   # Eu
    64: 157.25,    # Gd
    65: 158.92535, # Tb
    66: 162.500,   # Dy
    67: 164.93032, # Ho
    68: 167.259,   # Er
    69: 168.93421, # Tm
    70: 173.054,   # Yb
    71: 174.9668,  # Lu
    72: 178.49,    # Hf
    73: 180.94788, # Ta
    74: 183.84,    # W
    75: 186.207,   # Re
    76: 190.23,    # Os
    77: 192.217,   # Ir
    78: 195.084,   # Pt
    79: 196.966569,# Au
    80: 200.59,    # Hg
    81: 204.3833,  # Tl
    82: 207.2,     # Pb
    83: 208.98040, # Bi
    84: 209.0,     # Po
    85: 210.0,     # At
    86: 222.0,     # Rn
    87: 223.0,     # Fr
    88: 226.0,     # Ra
    89: 227.0,     # Ac
    90: 232.03806, # Th
    91: 231.03588, # Pa
    92: 238.02891, # U
}

DEFAULT_ATOMIC_MASS_AMU: float = 1.0
"""Fallback mass for unknown elements."""


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def angstrom_to_bohr(length_ang: float) -> float:
    """Convert length from Angstrom to Bohr."""
    return length_ang * ANGSTROM_TO_BOHR


def bohr_to_angstrom(length_bohr: float) -> float:
    """Convert length from Bohr to Angstrom."""
    return length_bohr * BOHR_TO_ANGSTROM


def positions_angstrom_to_bohr(positions: list[list[float]]) -> list[float]:
    """Convert a list of 3D positions from Angstrom to Bohr (flat output)."""
    flat = []
    for pos in positions:
        flat.extend([p * ANGSTROM_TO_BOHR for p in pos])
    return flat


def hartree_to_ev(energy_ha: float) -> float:
    """Convert energy from Hartree to eV."""
    return energy_ha * HARTREE_TO_EV


def ev_to_hartree(energy_ev: float) -> float:
    """Convert energy from eV to Hartree."""
    return energy_ev * EV_TO_HARTREE


def hartree_to_uha(energy_ha: float) -> float:
    """Convert energy from Hartree to microHartree."""
    return energy_ha * HARTREE_TO_UHA


def ha_bohr_to_ev_angstrom(force_ha_bohr: float) -> float:
    """Convert force from Ha/Bohr to eV/Angstrom."""
    return force_ha_bohr * HA_BOHR_TO_EV_ANGSTROM


def fs_to_au(time_fs: float) -> float:
    """Convert time from femtoseconds to atomic units."""
    return time_fs * FS_TO_AU


def fs_to_ps(time_fs: float) -> float:
    """Convert time from femtoseconds to picoseconds."""
    return time_fs * FS_TO_PS


def amu_to_au(mass_amu: float) -> float:
    """Convert mass from atomic mass units to electron-mass units (a.u.)."""
    return mass_amu * AMU_TO_AU


def atomic_mass_au(atomic_number: int) -> float:
    """Get atomic mass in a.u. (electron-mass units) for a given Z."""
    return amu_to_au(ATOMIC_MASS_AMU.get(atomic_number, DEFAULT_ATOMIC_MASS_AMU))


def atomic_masses_au(atomic_numbers: list[int]) -> list[float]:
    """Get atomic masses in a.u. for a list of atomic numbers."""
    return [atomic_mass_au(z) for z in atomic_numbers]


# ---------------------------------------------------------------------------
# Grid config conversion helper
# ---------------------------------------------------------------------------

def grid_config_to_bohr(coarse_spacing_ang: float, margin_ang: float) -> tuple[float, float]:
    """Convert grid parameters from Angstrom to Bohr.

    Args:
        coarse_spacing_ang: coarse grid spacing in Angstrom
        margin_ang: grid margin around molecule in Angstrom

    Returns:
        (grid_h_bohr, grid_margin_bohr)
    """
    return angstrom_to_bohr(coarse_spacing_ang), angstrom_to_bohr(margin_ang)
