TIDES Configuration Options (Auto-Generated)
================================================

This page is auto-generated from the TIDES dataclass schema. To regenerate, run ``python3 tides/tools/auto_docs_generator.py``.

[basis]
-------

Basis set configuration (NAO).

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - kind
     - str
     - 'DZP'
     - Basis quality: DZP, TZP, TZDP, or custom.
   * - confining_radius
     - float
     - 5.0
     - Confining radius in Bohr for on-the-fly generation.
   * - diffuse
     - bool
     - False
     - Add diffuse functions (for anions/surfaces).

Example TOML::

   [basis]
   kind = 'DZP'
   confining_radius = 5.0
   diffuse = False


[grid]
------

Real-space grid configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - coarse_spacing
     - float
     - 0.2
     - Coarse grid spacing in Angstroms (orbital grid).
   * - fine_spacing
     - float
     - 0.15
     - Fine grid spacing in Angstroms (density grid).

Example TOML::

   [grid]
   coarse_spacing = 0.2
   fine_spacing = 0.15


[md]
----

Molecular dynamics parameters.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - mode
     - str
     - 'static'
     - MD mode: static, xl-bomd, scf-md, optimize, neb.
   * - n_steps
     - int
     - 0
     - Number of MD/optimization steps.
   * - timestep
     - float
     - 0.5
     - Timestep in femtoseconds.
   * - thermostat
     - str
     - 'none'
     - Thermostat: none (NVE), langevin, nose-hoover.
   * - temperature
     - float
     - 0.0
     - Target temperature in Kelvin (thermostatted runs).
   * - optimizer
     - str
     - 'fire'
     - Geometry optimizer: fire, lbfgs.
   * - f_max
     - float
     - 0.0001
     - Force convergence criterion for optimization (Ha/Bohr).

Example TOML::

   [md]
   mode = 'static'
   n_steps = 0
   timestep = 0.5
   thermostat = 'none'
   temperature = 0.0
   optimizer = 'fire'
   f_max = 0.0001


[output]
--------

Output control.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - verbosity
     - str
     - 'normal'
     - Verbosity: silent, normal, verbose, debug.
   * - restart_file
     - str
     - ''
     - HDF5 restart file path.
   * - dump_stages
     - list[str]
     - []
     - Pipeline stages to dump (bisect-the-physics): geometry, S, H0, rho, vH, vxc, H, P, E_components, forces, stress.
   * - log_file
     - str
     - ''
     - Structured log file path (JSON lines).

Example TOML::

   [output]
   verbosity = 'normal'
   restart_file = ''
   log_file = ''


[precision]
-----------

Precision policy configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - tile_storage
     - str
     - 'bf16'
     - Tile storage dtype: bf16, fp16, fp32, fp64.
   * - accumulate
     - str
     - 'fp32'
     - Accumulation dtype: fp32, fp64.
   * - critical_reductions
     - str
     - 'f64e'
     - Critical reduction method: f64e (Ozaki), fp64.
   * - deterministic
     - bool
     - False
     - If True, use ordered reductions for reproducibility.

Example TOML::

   [precision]
   tile_storage = 'bf16'
   accumulate = 'fp32'
   critical_reductions = 'f64e'
   deterministic = False


[pseudo]
--------

Pseudopotential configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - kind
     - str
     - 'ONCV'
     - Pseudopotential type: ONCV (PseudoDojo default).
   * - library
     - str
     - 'PseudoDojo'
     - Pseudopotential library source.

Example TOML::

   [pseudo]
   kind = 'ONCV'
   library = 'PseudoDojo'


[scf]
-----

SCF convergence parameters.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - max_iter
     - int
     - 100
     - Maximum SCF iterations.
   * - energy_tol
     - float
     - 1e-08
     - Energy convergence tolerance in Hartree.
   * - density_tol
     - float
     - 1e-06
     - Density convergence tolerance.
   * - mixing
     - str
     - 'pulay'
     - Mixing method: simple, pulay, kerker, broyden.
   * - mixing_alpha
     - float
     - 0.3
     - Mixing parameter (0 < alpha <= 1).
   * - pulay_depth
     - int
     - 6
     - Pulay history depth.
   * - smearing
     - str
     - 'fermi'
     - Smearing method: fermi, gauss, mp (Methfessel-Paxton).
   * - electronic_temp
     - float
     - 0.0
     - Electronic temperature in Kelvin (0 = T=0).

Example TOML::

   [scf]
   max_iter = 100
   energy_tol = 1e-08
   density_tol = 1e-06
   mixing = 'pulay'
   mixing_alpha = 0.3
   pulay_depth = 6
   smearing = 'fermi'
   electronic_temp = 0.0


[solver]
--------

Solver regime configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - regime
     - str
     - 'auto'
     - Solver regime: auto, R0, R1, R2, R3 (auto = broker dispatch).
   * - broker_calib_file
     - str
     - ''
     - Path to broker calibration table from `tides tune`.

Example TOML::

   [solver]
   regime = 'auto'
   broker_calib_file = ''


[system]
--------

System definition: atoms, positions, cell, boundary conditions.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
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
     - 'free'
     - Boundary conditions: free, wire, slab, or periodic.
   * - fractional_positions
     - bool
     - False
     - If True, positions are fractional (periodic only).

Example TOML::

   [system]
   n_atoms = 0
   boundary_conditions = 'free'
   fractional_positions = False


[tides]
-------

Top-level TIDES configuration, mirroring the TOML input schema.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - system
     - SystemConfig
     - SystemConfig()
     - 
   * - basis
     - BasisConfig
     - BasisConfig()
     - 
   * - pseudo
     - PseudoConfig
     - PseudoConfig()
     - 
   * - xc
     - XCConfig
     - XCConfig()
     - 
   * - scf
     - SCFConfig
     - SCFConfig()
     - 
   * - solver
     - SolverConfig
     - SolverConfig()
     - 
   * - md
     - MDConfig
     - MDConfig()
     - 
   * - precision
     - PrecisionConfig
     - PrecisionConfig()
     - 
   * - grid
     - GridConfig
     - GridConfig()
     - 
   * - output
     - OutputConfig
     - OutputConfig()
     - 

Example TOML::

   [tides]
   system = SystemConfig()
   basis = BasisConfig()
   pseudo = PseudoConfig()
   xc = XCConfig()
   scf = SCFConfig()
   solver = SolverConfig()
   md = MDConfig()
   precision = PrecisionConfig()
   grid = GridConfig()
   output = OutputConfig()


[xc]
----

Exchange-correlation functional configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Key
     - Type
     - Default
     - Description
   * - functional
     - str
     - 'PBE'
     - XC functional: LDA, PBE, PBE0, HSE06, B3LYP, etc.
   * - dispersion
     - str
     - 'none'
     - Dispersion correction: none, D3, D3BJ, D4.

Example TOML::

   [xc]
   functional = 'PBE'
   dispersion = 'none'

