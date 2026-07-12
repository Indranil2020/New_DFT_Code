Grid Poisson and XC
====================

Poisson Solver
--------------

The Hartree potential :math:`V_H` is computed by solving Poisson's equation
on a real-space grid:

.. math::

   \nabla^2 V_H(\mathbf{r}) = -4\pi \rho(\mathbf{r})

Four boundary conditions are supported:

- **Free** (molecular): direct summation with self-term regularization
- **Wire** (1D periodic): FFT along periodic direction
- **Slab** (2D periodic): FFT in periodic plane
- **Periodic** (3D): full 3D FFT via cuFFT

XC Engine
---------

The XC functional is evaluated on the real-space grid:

.. math::

   E_{xc} = \int \rho(\mathbf{r}) \, \varepsilon_{xc}[\rho](\mathbf{r}) \, d^3r

Supported functionals: LDA (PW92), PBE, and hybrids (PBE0, HSE06) via libxc.
The Tier-0 GPU path evaluates XC pointwise on the GPU with FP64 accuracy
and FP32 mixed-precision A/B testing.
