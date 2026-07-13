API Reference
=============

The TIDES C++ API is documented via Doxygen and integrated into Sphinx using
the Breathe extension.

.. note::

   Doxygen XML output must be generated before building the API docs.
   Run ``make doxygen`` in the ``docs/sphinx`` directory, or build the
   project with CMake (which generates Doxygen XML as a build step).

Core Modules
------------

.. doxygennamespace:: tides
   :project: tides

Subsystems
----------

.. doxygennamespace:: tides::scf
   :project: tides

.. doxygennamespace:: tides::solvers
   :project: tides

.. doxygennamespace:: tides::tile
   :project: tides

.. doxygennamespace:: tides::grid
   :project: tides
