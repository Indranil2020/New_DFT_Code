.. TIDES Theory Manual and API Documentation

TIDES — Tensor-Core DFT Engine
==============================

TIDES (TIdes DFT Engine for Science) is a GPU-native, mixed-precision DFT
engine built on a tile substrate with solver-broker dispatch, extended-Lagrangian
Born-Oppenheimer molecular dynamics (XL-BOMD), and certified a-posteriori
error control.

This documentation set provides the mathematical derivations for every
physics and math module, the solver broker regime dispatch logic, and the
auto-generated API configuration reference.

.. toctree::
   :maxdepth: 2
   :caption: Theory Manual

   theory/nao_basis
   theory/scf
   theory/solver_broker
   theory/xlbomd
   theory/tile_substrate
   theory/poisson_xc
   theory/error_control

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/config_options

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
