Configuration Options
=====================

This page is auto-generated from the TidesConfig dataclass schema.
To regenerate, run: ``python3 tools/auto_docs_generator.py``

.. contents::
   :local:
   :depth: 2

System
------

.. list-table:: SystemConfig
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - n_atoms
     - int
     - 0
     - Number of atoms.
   * - atomic_numbers
     - list[int]
     - []
     - Atomic numbers (Z) for each atom.
   * - positions
     - list[list[float]]
     - []
     - Atomic positions in Angstroms, shape [n_atoms][3].
   * - cell
     - list[list[float]]
     - []
     - Unit cell vectors in Angstroms (3x3). Empty for free BCs.
   * - boundary_conditions
     - str
     - free
     - Boundary conditions: free, wire, slab, or periodic.
   * - fractional_positions
     - bool
     - False
     - If True, positions are fractional (periodic only).

Basis
-----

.. list-table:: BasisConfig
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - kind
     - str
     - DZP
     - Basis quality: DZP, TZP, TZDP, or custom.
   * - confining_radius
     - float
     - 5.0
     - Confining radius in Bohr for on-the-fly generation.
   * - diffuse
     - bool
     - False
     - Add diffuse functions (for anions/surfaces).

XC
--

.. list-table:: XCConfig
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - functional
     - str
     - PBE
     - XC functional: LDA, PBE, PBE0, HSE06, B3LYP, etc.
   * - dispersion
     - str
     - none
     - Dispersion correction: none, D3, D3BJ, D4.

Example TOML
------------

.. code-block:: toml

   [system]
   n_atoms = 2
   atomic_numbers = [1, 1]
   positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
   boundary_conditions = "free"

   [basis]
   kind = "DZP"

   [xc]
   functional = "PBE"
   dispersion = "D3BJ"

   [scf]
   max_iter = 100
   energy_tol = 1e-8
   mixing = "pulay"
   mixing_alpha = 0.3
