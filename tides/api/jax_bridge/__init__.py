"""TIDES JAX bridge package.

Exposes energy_and_forces(R) -> (E, F) with analytic-gradient VJP.
Requires jax + jaxlib (optional dependency).
"""
from .tides_jax import energy_and_forces, gradcheck, _JAX_AVAILABLE

__all__ = ["energy_and_forces", "gradcheck", "_JAX_AVAILABLE"]
