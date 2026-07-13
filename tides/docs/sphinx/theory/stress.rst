Stress Tensor
=============

The stress tensor for periodic systems:

.. math::
   \sigma_{\alpha\beta} = -\frac{1}{V}
       \frac{\partial E}{\partial \epsilon_{\alpha\beta}}

where :math:`\epsilon` is the strain tensor and :math:`V` is the cell volume.
Components:

* **Kinetic:** from the strain derivative of the basis functions
* **Hartree + XC:** from the grid deformation
* **Ion-ion:** from the Ewald strain derivative

Computed via finite differences in the CPU reference (T6.4), analytically in
the production path.
