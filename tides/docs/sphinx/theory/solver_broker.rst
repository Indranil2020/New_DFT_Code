Solver Broker and Regime Dispatch
=================================

The solver broker is the central dispatch mechanism that routes each DFT
calculation to the optimal solver regime based on system size, electronic gap,
temperature, and available GPU memory :cite:`tides2026`.

Regimes
-------

+-------+------------------+------------------------------------------+
| Regime| System Size      | Method                                   |
+=======+==================+==========================================+
| R0    | :math:`N \lesssim 200` basis | Batched dense eigensolver (LAPACK) |
| R1    | :math:`N \lesssim 2000` basis | Chebyshev-filtered subspace iter. |
| R2    | :math:`N \gtrsim 2000`, gapped| SP2 density-matrix purification   |
| R3    | :math:`N \gtrsim 2000`, metal | Fermi-operator expansion (FOE)    |
+-------+------------------+------------------------------------------+

Dispatch Logic
--------------

Given a :class:`BrokerInput` with system parameters, the broker selects:

1. **Metallic** (gap :math:`< 0.1` eV or :math:`T_e > 0`): R3 FOE for large
   :math:`N`, R0 for small :math:`N`.
2. **Small molecular** (:math:`N_\text{basis} \leq 200`): R0 batched dense
   (fastest and most accurate for Phase A molecular systems).
3. **Mid-range gapped** (:math:`N_\text{basis} \leq 2000`): R1 ChFSI.
4. **Large gapped** (:math:`N_\text{basis} > 2000`): R2 SP2 submatrix.
5. **Fallback**: calibration table lookup by atom count range and VRAM.

Calibration Table
-----------------

The ``tides tune`` command generates a per-device calibration table by timing
each regime on representative system sizes. The table caches:

* Atom count range (:math:`N_\text{lo}`, :math:`N_\text{hi}`)
* Time per step (ms)
* VRAM requirement (MB)
* Whether the regime is available on this hardware

The broker guarantees that the chosen regime is within 10% of the best
available regime for the given system size (T4.6 observable).

BrokerRunner: Dispatch and Solve
--------------------------------

The :class:`BrokerRunner` provides a single entry point that:

1. Calls :meth:`SolverBroker::Dispatch` to select the regime.
2. Routes to the corresponding solver:

   * R0 → :meth:`BatchedDenseEig::SolveGeneralized`
   * R1 → :meth:`ChFSI::Solve`
   * R2 → :meth:`SP2Purification::Purify`
   * R3 → :meth:`FermiOperatorExpansion::Compute`

3. Returns the density matrix :math:`P`, eigenvalues (for R0/R1), convergence
   flag, and the regime used.

ChFSI subspace reuse and locking/deflation are supported (see
:ref:`tile-substrate`).
