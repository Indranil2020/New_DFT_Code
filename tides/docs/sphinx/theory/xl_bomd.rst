XL-BOMD Shadow Dynamics
=======================

Extended Lagrangian
-------------------

XL-BOMD (Niklasson, PRL 2008) introduces auxiliary electronic degrees of
freedom :math:`n(t)` that evolve harmonically around the ground state
:math:`n_0(R)`:

.. math::
   \mathcal{L} = \mathcal{L}_{\text{nuc}}(R, \dot{R})
       + \frac{\mu}{2}\dot{n}^2
       - \frac{\kappa}{2}(n - n_0(R))^2

where :math:`\mu` is a fictitious electronic mass and :math:`\kappa` is a
spring constant.

Equations of Motion
-------------------

The Euler-Lagrange equations give:

.. math::
   n(t+\Delta t) = 2n(t) - n(t-\Delta t)
       + \Delta t^2 \kappa (n_0(R) - n(t))

.. math::
   R(t+\Delta t) = 2R(t) - R(t-\Delta t)
       + \Delta t^2 F(R, n(t)) / M

KSA Kernel
----------

The full Krylov-subspace-approximated (KSA) kernel replaces :math:`\kappa` with
an approximate inverse Jacobian :math:`K \approx J^{-1}`, where
:math:`J = \partial n_0 / \partial n`. This stabilizes the dynamics for
small-gap systems where the Jacobian is ill-conditioned.

In the simplified CPU reference, :math:`K = I` (identity), and
:math:`n_0(R)` is computed fresh each step (the "1 solve/step" design).

Energy Conservation (NVE)
-------------------------

For NVE dynamics, the shadow energy:

.. math::
   E_{\text{shadow}} = E_{\text{nuc}}(R, \dot{R})
       + \frac{\kappa}{2}(n - n_0)^2

is conserved. The drift budget is
:math:`\leq 30\;\mu\text{Ha}/\text{atom}/\text{ps}` (GB2 gate).
