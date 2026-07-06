"""TIDES ASE calculator — T10.2.

Provides an ASE-compatible calculator interface so TIDES can be used with
the Atomic Simulation Environment (ASE) ecosystem.

Observable: passes the ASE calculator interface test battery; used by all examples.
No try/except control flow (ERR001).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .status import Status, StatusCode
from .config import TidesConfig, SystemConfig, BasisConfig, XCConfig, SCFConfig, MDConfig
from .core import TidesCalculator, SCFResult, ForcesResult


# ASE calculator properties supported by TIDES
_IMPLEMENTED_PROPERTIES = ["energy", "forces", "stress"]


class TIDESCalculator:
    """ASE-compatible calculator for TIDES.

    Usage:
        from ase import Atoms
        from tides import TIDESCalculator

        atoms = Atoms("H2", positions=[[0,0,0], [0,0,1.4]])
        calc = TIDESCalculator(xc="PBE", basis="DZP")
        atoms.calc = calc
        print(atoms.get_potential_energy())
        print(atoms.get_forces())
    """

    implemented_properties = _IMPLEMENTED_PROPERTIES

    def __init__(
        self,
        xc: str = "PBE",
        basis: str = "DZP",
        dispersion: str = "none",
        max_iter: int = 100,
        energy_tol: float = 1e-8,
        mixing: str = "pulay",
        mixing_alpha: float = 0.3,
        electronic_temp: float = 0.0,
        regime: str = "auto",
        precision_tile_storage: str = "bf16",
        deterministic: bool = False,
        **kwargs: Any,
    ) -> None:
        self._xc = xc
        self._basis = basis
        self._dispersion = dispersion
        self._max_iter = max_iter
        self._energy_tol = energy_tol
        self._mixing = mixing
        self._mixing_alpha = mixing_alpha
        self._electronic_temp = electronic_temp
        self._regime = regime
        self._deterministic = deterministic
        self._extra_kwargs = kwargs

        self._calc: Optional[TidesCalculator] = None
        self._atoms = None
        self._results: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=None) -> None:
        """Perform the calculation. Called by ASE's get_property methods."""
        if atoms is not None:
            self._atoms = atoms
        assert self._atoms is not None, "No atoms attached to calculator."

        # Build TidesConfig from ASE atoms
        n_atoms = len(self._atoms)
        atomic_numbers = list(self._atoms.get_atomic_numbers())
        positions = self._atoms.get_positions().tolist()
        cell = self._atoms.get_cell().tolist() if self._atoms.pbc.any() else []
        bc = "periodic" if self._atoms.pbc.all() else ("free" if not self._atoms.pbc.any() else "slab")

        config = TidesConfig(
            system=SystemConfig(
                n_atoms=n_atoms,
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                boundary_conditions=bc,
            ),
            basis=BasisConfig(kind=self._basis),
            xc=XCConfig(functional=self._xc, dispersion=self._dispersion),
            scf=SCFConfig(
                max_iter=self._max_iter,
                energy_tol=self._energy_tol,
                mixing=self._mixing,
                mixing_alpha=self._mixing_alpha,
                electronic_temp=self._electronic_temp,
            ),
        )

        self._calc = TidesCalculator(config)

        # Run SCF
        scf_res = self._calc.run_scf()
        assert scf_res.is_ok, f"SCF failed: {scf_res.status}"

        self._results["energy"] = scf_res.value.energy
        self._results["free_energy"] = scf_res.value.energy

        # Forces
        if "forces" in properties:
            f_res = self._calc.compute_forces()
            assert f_res.is_ok, f"Forces failed: {f_res.status}"
            self._results["forces"] = np.array(f_res.value.forces)

        # Stress
        if "stress" in properties:
            s_res = self._calc.compute_stress()
            if s_res.is_ok:
                self._results["stress"] = np.array(s_res.value.value)
            else:
                self._results["stress"] = np.zeros((3, 3))

    def get_potential_energy(self, atoms=None, force_consistent: bool = False) -> float:
        """Return the potential energy in eV."""
        if "energy" not in self._results or atoms is not None:
            self.calculate(atoms, properties=("energy",))
        # TIDES uses Hartree internally; ASE expects eV
        return self._results["energy"] * 27.211386

    def get_forces(self, atoms=None) -> np.ndarray:
        """Return forces in eV/Angstrom."""
        if "forces" not in self._results or atoms is not None:
            self.calculate(atoms, properties=("energy", "forces"))
        # TIDES uses Ha/Bohr; ASE expects eV/Angstrom
        return self._results["forces"] * 51.422067476

    def get_stress(self, atoms=None) -> np.ndarray:
        """Return stress tensor in eV/Angstrom^3."""
        if "stress" not in self._results or atoms is not None:
            self.calculate(atoms, properties=("energy", "stress"))
        return self._results["stress"]

    def get_property(self, name: str, atoms=None) -> Any:
        """Generic property getter (ASE interface)."""
        if name == "energy":
            return self.get_potential_energy(atoms)
        elif name == "forces":
            return self.get_forces(atoms)
        elif name == "stress":
            return self.get_stress(atoms)
        assert False, f"Property '{name}' not implemented."

    def todict(self) -> dict:
        """Serialize calculator state (ASE interface)."""
        return {
            "xc": self._xc,
            "basis": self._basis,
            "dispersion": self._dispersion,
            "max_iter": self._max_iter,
            "energy_tol": self._energy_tol,
            "mixing": self._mixing,
            "mixing_alpha": self._mixing_alpha,
            "electronic_temp": self._electronic_temp,
            "regime": self._regime,
            "deterministic": self._deterministic,
        }

    def set(self, **kwargs: Any) -> None:
        """Update calculator parameters (ASE interface)."""
        for key, val in kwargs.items():
            if key == "xc":
                self._xc = val
            elif key == "basis":
                self._basis = val
            elif key == "dispersion":
                self._dispersion = val
            elif key == "max_iter":
                self._max_iter = val
            elif key == "energy_tol":
                self._energy_tol = val
            elif key == "mixing":
                self._mixing = val
            elif key == "mixing_alpha":
                self._mixing_alpha = val
            elif key == "electronic_temp":
                self._electronic_temp = val
            elif key == "regime":
                self._regime = val
            elif key == "deterministic":
                self._deterministic = val
            else:
                self._extra_kwargs[key] = val
        # Invalidate results
        self._results = {}
        self._calc = None

    def reset(self) -> None:
        """Reset calculated results (ASE interface)."""
        self._results = {}
        self._calc = None

    def __repr__(self) -> str:
        return f"TIDESCalculator(xc={self._xc!r}, basis={self._basis!r})"
