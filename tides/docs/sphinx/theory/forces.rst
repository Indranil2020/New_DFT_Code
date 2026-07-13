Forces: Hellmann–Feynman + Pulay + Grid + Dispersion
====================================================

Total Force Expression
----------------------

The force on atom :math:`I` is :math:`F_I = -\partial E / \partial R_I`. The
total energy :math:`E[\rho, \{R_I\}]` depends on positions both explicitly
(ion-ion, electron-ion) and implicitly (through the basis functions
:math:`\phi_\mu(\mathbf{r}; \mathbf{R}_I)` which move with the atoms in the NAO
representation). This gives four contributions:

.. math::
   F_I = F_I^{\text{HF}} + F_I^{\text{Pulay}} + F_I^{\text{grid}}
       + F_I^{\text{disp}} + F_I^{\text{ion}}

Hellmann–Feynman Force
----------------------

For a fixed density matrix :math:`P`, the Hellmann–Feynman force is:

.. math::
   F_I^{\text{HF}} = -\text{Tr}\left(P \frac{\partial H}{\partial R_I}\right)

In the NAO basis, :math:`H = T + V_{\text{ext}} + V_H[\rho] + V_{xc}[\rho]`,
where the kinetic :math:`T` and external :math:`V_{\text{ext}}`
(nucleus-electron) have explicit position dependence through the two-center
integrals (T2.6 derivative streams).

For the two-center terms (:math:`T`, :math:`V_{nl}` in Kleinman–Bylander form),
the derivative :math:`\partial H_{\mu\nu} / \partial R_I` is computed
analytically from the splined radial tables and spherical-harmonic rotation
matrices (WP2 T2.6).

Pulay Force
-----------

Because the NAO basis functions move with the atoms, the density matrix
:math:`P` has implicit position dependence. The Pulay force is:

.. math::
   F_I^{\text{Pulay}} = -\text{Tr}\left(\frac{\partial P}{\partial R_I} H\right)
       + \text{Tr}\left(P \frac{\partial S}{\partial R_I} \epsilon\right)

where :math:`\epsilon` is the energy-weighted density matrix:
:math:`\epsilon_{\mu\nu} = \sum_k f_k \epsilon_k C_{\mu k} C_{\nu k}`.

Using the idempotency relation :math:`P = P S P` (for the projector onto
occupied space), the Pulay force can be written as:

.. math::
   F_I^{\text{Pulay}} = -2\text{Tr}\left(\frac{\partial S}{\partial R_I}
       \, \epsilon \, P\right) + \text{Tr}\left(\frac{\partial S}{\partial R_I}
       \, P \, H \, P\right)

This requires :math:`\partial S / \partial R_I` (from T2.6) and the
energy-weighted density :math:`\epsilon`.

Grid Force
----------

The grid contribution comes from the XC energy and the grid-based part of the
Hartree energy:

.. math::
   F_I^{\text{grid}} = -\frac{\partial}{\partial R_I}\int \left[
       \epsilon_{xc}(\mathbf{r}) + \frac{1}{2}v_H(\mathbf{r})\right]
       n(\mathbf{r}) \, d\mathbf{r}

In TIDES, this is computed via the adjoint of the :math:`\rho`-build map
(WP3 T3.6): the grid-based potential
:math:`v(\mathbf{r}) \to H_{\mu\nu}` adjoint gives
:math:`\partial E / \partial R_I` from the grid terms.

Dispersion Force
----------------

For D3/D4, the dispersion energy is a sum of :math:`C_6 R^{-6}` terms:

.. math::
   E_{\text{disp}} = -\sum_{I<J}
       \frac{C_6^{IJ}(R_I, R_J)}{R_{IJ}^6} f_{\text{damp}}(R_{IJ})

The force is the analytic gradient:

.. math::
   F_I^{\text{disp}} = -\frac{\partial E_{\text{disp}}}{\partial R_I}

which is computed by the ``simple-dftd3`` / ``dftd4`` libraries (WP7 T7.1).

Finite-Difference Validation
-----------------------------

The 5-point central difference formula for the force:

.. math::
   F_I = -\frac{E(R_I + 2h) - 8E(R_I + h) + 8E(R_I - h)
       - E(R_I - 2h)}{12h} + O(h^4)

This is used as the validation oracle (GA1 gate): analytic forces must match
FD to :math:`\leq 10^{-6}` Ha/Bohr on the FP64 path.
