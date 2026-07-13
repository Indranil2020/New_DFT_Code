.. _scf-driver:

SCF Driver and Convergence
==========================

The self-consistent field (SCF) driver implements the Kohn-Sham iterative
cycle :cite:`tides2026`.

SCF Loop
--------

Given an initial density matrix :math:`P`, the SCF cycle proceeds:

1. **Build** the effective Hamiltonian :math:`H[P] = T + V_\text{ext} + V_H[P] + V_\text{xc}[P]`.
2. **Orthogonalize** the overlap :math:`S` via Löwdin
   :math:`S^{-1/2}` with eigenvalue filtering (discarding eigenvalues below a
   relative threshold to handle near-linear dependence).
3. **Solve** the generalized eigenproblem :math:`H\mathbf{x} = \epsilon S \mathbf{x}`
   via the solver broker (R0 dense eig, R1 ChFSI, R2 SP2, or R3 FOE).
4. **Occupy** the :math:`N_\text{occ}` lowest orbitals to build the new density
   :math:`P_\text{new} = C_\text{occ} C_\text{occ}^T`.
5. **Mix** :math:`P_\text{next} = \alpha P_\text{new} + (1 - \alpha) P_\text{old}`
   (simple) or Pulay/DIIS.
6. **Check** convergence: :math:`|E^{(k)} - E^{(k-1)}| < \epsilon`.

Pulay / DIIS Mixing
-------------------

The Pulay (DIIS) mixing scheme minimizes the residual
:math:`\mathbf{R}^{(k)} = P_\text{new}^{(k)} - P^{(k)}` in a least-squares
sense over the history of previous iterations:

.. math::

   \mathbf{c}^* = \arg\min_{\sum c_j = 1}
   \Bigl\|\sum_j c_j \mathbf{R}^{(j)}\Bigr\|^2,

leading to an augmented linear system:

.. math::

   \begin{pmatrix}
   \langle \mathbf{R}_0, \mathbf{R}_0 \rangle & \cdots & 1 \\
   \vdots & \ddots & \vdots \\
   1 & \cdots & 0
   \end{pmatrix}
   \begin{pmatrix} c_0 \\ \vdots \\ \lambda \end{pmatrix}
   =
   \begin{pmatrix} 0 \\ \vdots \\ 1 \end{pmatrix}.

The mixed density is :math:`P_\text{next} = \sum_j c_j^* P_\text{new}^{(j)}`.

If the DIIS coefficients diverge (:math:`|c_j| > 10` or non-finite), the mixer
falls back to Kerker-damped simple mixing with an adaptive damping parameter:

.. math::

   \alpha_\text{eff} = \min\!\Bigl(\max\!\bigl(\frac{\alpha}{1 + \|\mathbf{R}\|},\, 0.05\bigr),\, 0.8\Bigr).

Energy Assembly
---------------

The Kohn-Sham total energy:

.. math::

   E_\text{tot} = E_\text{kin} + E_\text{ne} + E_H + E_\text{xc} + E_\text{ion},

where:

* :math:`E_\text{kin} = \sum_k f_k \epsilon_k - \text{Tr}\!\bigl(P (V_\text{ext} + V_H + V_\text{xc})\bigr)`
* :math:`E_\text{ne} = \text{Tr}(P V_\text{ext})`
* :math:`E_H = \tfrac{1}{2}\,\text{Tr}(P V_H)`
* :math:`E_\text{xc} = \text{Tr}(P \epsilon_\text{xc})` (energy density, not potential)
* :math:`E_\text{ion} = \tfrac{1}{2}\sum_{I \neq J} Z_I Z_J / |R_I - R_J|` (Ewald for periodic)

All traces use FP64-emulated reductions (f64e) on the production path.
