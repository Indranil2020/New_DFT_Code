"""TIDES — TIle-based Democratic Electronic-Structure suite.

Python API for the TIDES Kohn-Sham DFT engine. All public functions return
Status/result objects; no try/except control flow (coding standard ERR001).
"""
from __future__ import annotations

from .status import Status, StatusCode, Result
from .config import (
    TidesConfig, load_config, validate_config,
    SystemConfig, BasisConfig, PseudoConfig, XCConfig, SCFConfig,
    SolverConfig, MDConfig, PrecisionConfig, GridConfig, OutputConfig,
)
from .core import TidesCalculator, SCFResult, EnergyResult, ForcesResult, MDResult

# ASE calculator is optional (requires numpy + ase).
# No try/except (ERR001): use importlib to check availability.
import importlib.util

_ASE_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("ase") is not None
)
if _ASE_AVAILABLE:
    from .ase_calculator import TIDESCalculator as ASECalculator
else:
    ASECalculator = None  # type: ignore[assignment,misc]

__version__ = "0.1.0-alpha"
__all__ = [
    "Status",
    "StatusCode",
    "Result",
    "TidesConfig",
    "load_config",
    "validate_config",
    "SystemConfig",
    "BasisConfig",
    "PseudoConfig",
    "XCConfig",
    "SCFConfig",
    "SolverConfig",
    "MDConfig",
    "PrecisionConfig",
    "GridConfig",
    "OutputConfig",
    "TidesCalculator",
    "SCFResult",
    "EnergyResult",
    "ForcesResult",
    "MDResult",
    "ASECalculator",
    "__version__",
]
