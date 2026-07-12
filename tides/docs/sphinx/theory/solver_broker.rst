Solver Broker Regime Dispatch
=============================

The solver broker dispatches each problem to the optimal solver regime
based on system size, gap, temperature, and available VRAM.

Regimes
-------

============ =============== ==================================== ====================
Regime       Solver          Use Case                             Complexity
============ =============== ==================================== ====================
R0           Batched dense   Small molecules (< 200 basis fn)    :math:`O(N^3)`
R1           ChFSI           Mid-range gapped (200–2000 basis)   :math:`O(N^2 \cdot k)`
R2           SP2             Large gapped (> 2000 basis)         :math:`O(N)`
R3           FOE/SQ          Metallic / finite-T                  :math:`O(N)`
============ =============== ==================================== ====================

Dispatch Logic
--------------

.. math::

   \text{regime} = \begin{cases}
   R_0 & \text{if } N_{\text{basis}} \leq 200 \text{ and } E_g > 0.1 \text{ eV} \\
   R_3 & \text{if metallic } (E_g < 0.1 \text{ eV or } T_e > 0) \text{ and } N > 200 \\
   R_1 & \text{if } N_{\text{basis}} \leq 2000 \\
   R_2 & \text{otherwise (gapped, large)}
   \end{cases}

Calibration
-----------

The ``tides tune`` command measures crossover points on the actual hardware
and caches a per-device calibration table. The broker checks that its chosen
regime is within 10% of the fastest available option (T4.6 observable).
