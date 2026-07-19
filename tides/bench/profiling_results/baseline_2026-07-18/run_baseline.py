#!/usr/bin/env python3
"""Day-0 baseline runner for ROADMAP-2026-07-18 (wraps benchmark_fair_comparison).

Runs the fair-comparison ladder through C10H22 (32 atoms) only. The full-ladder
run attempted on the morning of 2026-07-18 (tides/bench/benchmark_fair_output.log)
stalled inside C14H30 (44 atoms) and never reached the final summary/JSON dump,
so the 44- and 62-atom rungs are excluded here and that stall is itself recorded
as a baseline fact (see README.md in this directory).

No source files are modified: benchmark_fair_comparison.py is imported as-is and
its MOLECULES ladder is truncated in this process only. The JSON lands at the
script's hard-coded path (profiling_results/fair_benchmark_2026-07-17.json) and
is copied into this directory afterwards by the collection step.
"""
import sys

sys.path.insert(0, '/home/indranil/git/New_DFT_Code/tides/bench')

import benchmark_fair_comparison as bench

assert bench.MOLECULES[7][0] == 'C10H22', bench.MOLECULES[7]
bench.MOLECULES = bench.MOLECULES[:8]  # CH4 ... C10H22 (32 atoms)

bench.main()
