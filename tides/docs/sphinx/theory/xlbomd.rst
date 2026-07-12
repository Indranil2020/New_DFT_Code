XL-BOMD Extended Lagrangian MD
=============================

Extended Lagrangian Born-Oppenheimer Molecular Dynamics (XL-BOMD) achieves
time-reversible, energy-conserving MD with approximately one density-matrix
solve per step.

Equations of Motion
-------------------

The auxiliary electronic DOF (density matrix :math:`P`) evolves harmonically:

.. math::

   P_{\text{aux}}(t+\Delta t) = 2 P_{\text{aux}}(t) - P_{\text{aux}}(t-\Delta t) + \Delta t^2 \, K \, (P_0 - P_{\text{aux}}(t))

Nuclei move on the shadow potential:

.. math::

   \mathbf{R}(t+\Delta t) = 2\mathbf{R}(t) - \mathbf{R}(t-\Delta t) + \Delta t^2 \frac{\mathbf{F}(\mathbf{R}, P_{\text{aux}})}{M}

KSA Kernel
----------

The kernel :math:`K` is a low-rank approximation of the inverse Jacobian.
Three levels:

- **Order 0:** :math:`K = I` (identity, no curvature correction)
- **Order 1:** Diagonal scaling :math:`K_{ii} = \kappa / (\kappa + |P_i| \cdot \delta)`
- **Order 2:** Low-rank correction using top eigenvectors of the curvature

Thermostats
-----------

**Nose-Hoover Chain (NHC):** Integrated via the Suzuki-Yoshida scheme with
configurable chain length (default 4) and sub-timesteps (default 3).

**Langevin:** Stochastic friction + random force.

NVE Drift
--------

The observable (GB2 gate): :math:`\Delta E \leq 30` µHa/atom/ps at ~1
solve/step.
