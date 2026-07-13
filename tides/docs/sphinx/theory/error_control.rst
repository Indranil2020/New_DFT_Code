A-Posteriori Error Control
==========================

A-posteriori error control is TIDES' scientific differentiator (§3.2.6):
after SCF convergence, residual-based bounds certify the accuracy of the
computed energy, forces, and eigenvalues :cite:`tides2026`.

SCF Residual
------------

The primary residual is the commutator of the Hamiltonian with the density
matrix:

.. math::

   R = [H, P] = HP - PH.

For a converged SCF solution, :math:`[H, P] = 0` (the density matrix
commutes with the Hamiltonian in the eigenbasis). The Frobenius norm
:math:`\|R\|_F` measures the degree of non-commutativity.

Energy Error Bound
------------------

Using the Bauer–Fike theorem adapted for DFT, the energy error is bounded by:

.. math::

   |E_\text{approx} - E_\text{exact}|
   \leq C \cdot \|R\|_F \cdot \|P\|_F,

where :math:`C` is a problem-dependent constant related to the spectral gap.
For gapped systems, :math:`C \sim 1/\Delta E_\text{gap}`.

Eigenvalue Error Bound
----------------------

For each eigenpair :math:`(\epsilon_k, \mathbf{x}_k)`, the residual is:

.. math::

   r_k = \|(H - \epsilon_k S)\,\mathbf{x}_k\|,

and the eigenvalue error is bounded by:

.. math::

   |\epsilon_k^\text{approx} - \epsilon_k^\text{exact}| \leq \|r_k\|.

Force Error Bound
-----------------

The force error scales with the square root of the SCF residual:

.. math::

   \|\Delta \mathbf{F}\| \leq C \cdot \|R\|_F^{1/2}.

This square-root scaling means that forces converge faster than the energy
with respect to SCF iterations, consistent with the variational principle.

Convergence Assessment
----------------------

The :class:`APosterioriErrorControl` module provides:

1. **ErrorBounds**: all residual norms and error bounds after SCF.
2. **EnergyConverged**: check whether :math:`\Delta E \leq` target.
3. **RecommendIterations**: from the residual decay history, predict the
   number of additional SCF iterations needed to reach the target accuracy.

The density residual :math:`\|P_\text{new} - P_\text{old}\|_F` provides a
secondary convergence metric independent of the energy.
