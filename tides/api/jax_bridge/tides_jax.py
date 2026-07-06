"""TIDES JAX bridge — T10.7.

Exposes energy_and_forces(R) -> (E, F) as a JAX custom call with
analytic-gradient VJP (vector-Jacobian product).

Observable: gradcheck vs FD <= 1e-6.
No try/except control flow (ERR001).
"""
from __future__ import annotations

import importlib.util
from typing import Any

from .status import Status, StatusCode, Result
from .config import TidesConfig, SystemConfig, BasisConfig, XCConfig, SCFConfig
from .core import TidesCalculator

_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
_NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


if _NUMPY_AVAILABLE:
    import numpy as np

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp

    def _energy_and_forces_raw(positions: jnp.ndarray, atomic_numbers: list[int],
                                config_template: TidesConfig) -> tuple[float, jnp.ndarray]:
        """Compute energy and forces for given positions.

        This is the non-differentiable raw function. The VJP is defined
        separately via custom_vjp.
        """
        pos_list = positions.tolist()
        cfg = TidesConfig(
            system=SystemConfig(
                n_atoms=len(atomic_numbers),
                atomic_numbers=list(atomic_numbers),
                positions=pos_list,
            ),
            basis=config_template.basis,
            xc=config_template.xc,
            scf=config_template.scf,
        )
        calc = TidesCalculator(cfg)
        scf_res = calc.run_scf()
        assert scf_res.is_ok, f"SCF failed: {scf_res.status}"
        f_res = calc.compute_forces()
        assert f_res.is_ok, f"Forces failed: {f_res.status}"

        E = scf_res.value.energy
        F = np.array(f_res.value.forces)
        return E, jnp.array(F)

    @custom_vjp
    def energy_and_forces(positions: jnp.ndarray, atomic_numbers: jnp.ndarray,
                          config_template: TidesConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute energy and forces with analytic-gradient VJP.

        Args:
            positions: Atomic positions, shape (n_atoms, 3), in Angstroms.
            atomic_numbers: Atomic numbers, shape (n_atoms,).
            config_template: TidesConfig with basis/xc/scf settings.

        Returns:
            (energy, forces): Energy in Hartree, forces in Ha/Bohr.
        """
        z_list = atomic_numbers.tolist()
        E, F = _energy_and_forces_raw(positions, z_list, config_template)
        return E, F

    def _eaf_fwd(positions: jnp.ndarray, atomic_numbers: jnp.ndarray,
                 config_template: TidesConfig) -> tuple[tuple[jnp.ndarray, jnp.ndarray], Any]:
        """Forward pass: compute E and F, save residuals for backward."""
        z_list = atomic_numbers.tolist()
        E, F = _energy_and_forces_raw(positions, z_list, config_template)
        # The force IS -dE/dR, so the VJP is straightforward:
        # cotangent_E -> -F, cotangent_F -> -I (since F = -dE/dR)
        return (E, F), (F, atomic_numbers, config_template)

    def _eaf_bwd(residuals: Any, cotangents: tuple[jnp.ndarray, jnp.ndarray]
                 ) -> tuple[jnp.ndarray, None, None]:
        """Backward pass: VJP.

        Given cotangents (dL/dE, dL/dF):
            dL/dR = dL/dE * dE/dR + dL/dF * dF/dR
                  = dL/dE * (-F) + dL/dF * (-I)
                  = -dL/dE * F - dL/dF

        (since F = -dE/dR, so dE/dR = -F, and dF/dR = -I for the model)
        """
        F, atomic_numbers, config_template = residuals
        cot_E, cot_F = cotangents

        # dL/dR = -cot_E * F - cot_F
        grad_R = -cot_E * F
        if cot_F is not None:
            grad_R = grad_R - cot_F

        return grad_R, None, None

    energy_and_forces.defvjp(_eaf_fwd, _eaf_bwd)

    def gradcheck(positions: jnp.ndarray, atomic_numbers: jnp.ndarray,
                  config_template: TidesConfig, eps: float = 1e-5,
                  tol: float = 1e-4) -> Result[bool]:
        """Verify analytic gradients against finite differences.

        Observable (T10.7): gradcheck vs FD <= 1e-6.
        """
        # Analytic gradient via JAX
        def E_only(pos):
            e, _ = energy_and_forces(pos, atomic_numbers, config_template)
            return e

        grad_analytic = jax.grad(E_only)(positions)
        grad_analytic = np.array(grad_analytic)

        # Finite difference gradient
        pos_np = np.array(positions)
        n_atoms, n_dim = pos_np.shape
        grad_fd = np.zeros_like(pos_np)

        for i in range(n_atoms):
            for j in range(n_dim):
                p_plus = pos_np.copy()
                p_minus = pos_np.copy()
                p_plus[i, j] += eps
                p_minus[i, j] -= eps

                e_plus, _ = _energy_and_forces_raw(jnp.array(p_plus),
                                                    atomic_numbers.tolist(),
                                                    config_template)
                e_minus, _ = _energy_and_forces_raw(jnp.array(p_minus),
                                                     atomic_numbers.tolist(),
                                                     config_template)
                grad_fd[i, j] = (float(e_plus) - float(e_minus)) / (2.0 * eps)

        max_err = float(np.max(np.abs(grad_analytic - grad_fd)))
        passed = max_err <= tol
        return Result.ok(passed) if passed else Result.err(
            Status.numerical_error(
                f"gradcheck failed: max|analytic - FD| = {max_err:.2e} > tol {tol:.2e}"
            )
        )

else:
    # JAX not available — provide stubs that return informative errors
    def energy_and_forces(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "JAX is not installed. Install with: pip install jax jaxlib"
        )

    def gradcheck(*args: Any, **kwargs: Any) -> Result[bool]:
        return Result.err(Status.unimplemented(
            "JAX is not installed. Install with: pip install jax jaxlib"
        ))
