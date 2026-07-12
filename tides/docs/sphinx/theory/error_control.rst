A-Posteriori Error Control
==========================

TIDES implements DFTK-style certified error bounds — the scientific
differentiator (§3.2.6).

SCF Residual
------------

The SCF residual measures the commutator of the Hamiltonian with the density
matrix:

.. math::

   R = \| [H[P], P] \|_F = \| HP - PH \|_F

For a converged SCF, :math:`[H, P] = 0` since :math:`H` and :math:`P` share
the same eigenbasis.

Energy Error Bound
------------------

The energy error is bounded by (Bauer-Fike theorem adapted for DFT):

.. math::

   \Delta E \leq C \cdot \| [H, P] \|_F \cdot \| P \|_F

where :math:`C \approx 1` for well-conditioned problems.

Eigenvalue Error Bounds
-----------------------

For each eigenpair :math:`(\varepsilon_k, \mathbf{x}_k)`:

.. math::

   \Delta \varepsilon_k \leq \| (H - \varepsilon_k S) \mathbf{x}_k \|_2

Force Error Bound
-----------------

Force errors scale as the square root of the SCF residual:

.. math::

   \Delta F \leq C \cdot \sqrt{ \| [H, P] \|_F }

This reflects the Hellmann-Feynman theorem: forces are first derivatives of
the energy, so their error is :math:`O(\sqrt{\Delta E})`.

Iteration Recommendation
-----------------------

Given the residual decay rate :math:`r`, the number of additional iterations
needed to reach a target :math:`\epsilon`:

.. math::

   n_{\text{extra}} = \frac{\log(\epsilon / R_{\text{current}})}{\log(r)}
