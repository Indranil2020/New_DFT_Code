XL-BOMD Extended Lagrangian Molecular Dynamics
==============================================

XL-BOMD (Extended Lagrangian Born-Oppenheimer Molecular Dynamics)
:cite:`niklasson2008` introduces auxiliary electronic degrees of freedom that
evolve harmonically around the ground state, enabling time-reversible,
energy-conserving MD with approximately one density-matrix solve per step.

Extended Lagrangian
-------------------

The extended Lagrangian is:

.. math::

   \mathcal{L} = \mathcal{L}_\text{nuc}(R, \dot{R})
   + \frac{\mu}{2}\dot{n}^2
   - \frac{\kappa}{2}\bigl(n - n_0(R)\bigr)^2,

where :math:`\mu` is a fictitious electronic mass, :math:`\kappa` is a spring
constant, and :math:`n_0(R)` is the ground-state density at nuclear
configuration :math:`R`.

Equations of Motion
-------------------

The Euler–Lagrange equations give the Verlet-type update:

.. math::

   n(t+\Delta t) &= 2n(t) - n(t-\Delta t) + \Delta t^2\, \kappa\,
   \bigl(n_0(R) - n(t)\bigr),

   R(t+\Delta t) &= 2R(t) - R(t-\Delta t) + \frac{\Delta t^2}{M}\,
   F\!\bigl(R, n(t)\bigr).

The key advantage: :math:`n_0(R)` needs to be computed only once per step
(the "1 solve/step" design), not iterated to convergence.

KSA Kernel
----------

The Krylov-subspace-approximated (KSA) kernel replaces :math:`\kappa` with an
approximate inverse Jacobian :math:`K \approx J^{-1}`, where
:math:`J = \partial n_0 / \partial n`. This stabilizes the dynamics for
small-gap systems where the Jacobian is ill-conditioned.

In the full implementation, :math:`K` is a low-rank approximation:

.. math::

   K \approx I - V \Lambda V^T,

where :math:`V` are the top eigenvectors of the curvature. The kernel order
controls the rank:

* **Order 0**: :math:`K = I` (identity — the simplest approximation).
* **Order 1**: diagonal scaling :math:`K = \text{diag}(\kappa_i)`.
* **Order 2**: full low-rank with top-:math:`k` eigenvectors.

Nose-Hoover Chain Thermostat
-----------------------------

For NVT dynamics, a Nose-Hoover chain (NHC) of length :math:`M` is integrated
via the Suzuki–Yoshida scheme. The chain variables
:math:`\{\xi_i, p_{\xi_i}\}_{i=1}^M` are coupled to the nuclear velocities:

.. math::

   \dot{p}_{\xi_1} = \frac{1}{Q_1}\!\bigl(p^2/M - N_f k_B T\bigr)
   - \frac{p_{\xi_2}}{Q_2}\, p_{\xi_1},

   \dot{p}_{\xi_i} = \frac{p_{\xi_{i-1}}^2}{Q_{i-1}} - \frac{p_{\xi_{i+1}}}{Q_{i+1}}\, p_{\xi_i}.

The Suzuki–Yoshida integration uses a multi-timestep decomposition with
:math:`n_\text{sy} = 3` or :math:`7` substeps for energy conservation.

Shadow Dynamics
---------------

True XL-BOMD shadow dynamics propagate the auxiliary density :math:`P_\text{aux}`
harmonically without recomputing the ground state each step:

.. math::

   P_\text{aux}^{(k+1)} = 2 P_\text{aux}^{(k)} - P_\text{aux}^{(k-1)}
   + \Delta t^2\, K\, \bigl(P_0 - P_\text{aux}^{(k)}\bigr).

:math:`P_0` is computed once at initialization (or refreshed every
:math:`N_\text{refresh}` steps). The nuclear forces use the shadow potential
from :math:`P_\text{aux}`, not a fresh SCF.

Energy Conservation (NVE)
--------------------------

For NVE dynamics, the shadow energy:

.. math::

   E_\text{shadow} = E_\text{nuc}(R, \dot{R})
   + \frac{\kappa}{2}\bigl(n - n_0\bigr)^2

is conserved. The drift budget is
:math:`\leq 30\;\mu\text{Ha}/\text{atom}/\text{ps}` (GB2 gate).
