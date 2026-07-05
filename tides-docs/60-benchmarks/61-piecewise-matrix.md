# Piecewise competitor matrix (all baselines open source; reverify licenses at kickoff)
| # | TIDES module | Baseline (license) | Payload | Metric | Excellence bar |
|---|---|---|---|---|---|
| 1 | NAO H/S build | ABACUS (LGPL-3), SIESTA (GPL-3), CP2K GPW (GPL-2) | 10^3-atom Si; 512-H2O | atoms/s @ matched sparsity | >=3x ABACUS-GPU |
| 2 | rho/V grid ops | GPAW (GPL-3), SPARC (GPL-3) | same | % HBM roofline | >=60% sustained |
| 3 | Poisson all BCs | BigDFT PSolver (GPL); cuFFT ref | 128^3–512^3 free/slab/bulk | time @1e-10 | >= parity with FFT path |
| 4 | Dense eig bridge | ELPA (LGPL-3), cuSOLVERMp | 10k–40k dense | time | tracked (external) |
| 5 | ChFSI | DFT-FE (LGPL-2.1), SPARC | 5k-atom Mo | time/SCF @<=1 meV/atom | <=1.5x DFT-FE |
| 6 | SP2-submatrix | CP2K NOLSM/DBCSR (GPL-2), NTPoly (MIT) | 10^4–10^6 a-Si:H | time/purif @<=0.5 meV/atom | >=2x DBCSR-GPU |
| 7 | FOE/SQ metals | SPARC-SQ (GPL-3), CheSS (GPL) | 10^4 Al, Te 1–10 kK | time/step | parity, GPU-native |
| 8 | Hybrids ISDF+ACE | CP2K ADMM, QE ACE (GPL) | 500-atom TiO2 slab HSE06 | time/SCF @matched acc | <=4x own PBE |
| 9 | MD engine | CP2K, GPAW; anchors DFTB+ (LGPL-3), tblite (LGPL) | 64/512/4096 H2O NVE | steps/s + drift | within 5–20x DFTB+ |
| 10 | Small-molecule e2e | GPU4PySCF (Apache-2), PySCF (Apache-2); xQC if open | W4-11 & S22 subsets | batched mol/h; cold latency | >=10x throughput; <=2x cold |
| 11 | Accuracy e2e | published all-electron refs (ACWF etc.) | ACWF subset, S22, surfaces | meV/atom; kcal/mol | convergence documented |
| 12 | Scaling | CONQUEST (MIT), CP2K | 10^5->10^6 a-Si, water | weak/strong eff. | >=80% weak to 8 GPUs (C) |
Phase note: rows 1–5,9,10 start in Phases A/B on RTX/A40; rows 6–8 validated on workstations at
proxy sizes; row 12 is Phase C.
