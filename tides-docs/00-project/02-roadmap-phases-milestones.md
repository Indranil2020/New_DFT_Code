# Roadmap — four phases (60 months), workstation-first
## Phase A — Molecules on workstations (M1–M12)
Scope: WP1 substrate, WP2 integrals, WP3 grids/Poisson/XC, WP4 R0+R1, WP6 SCF+forces, WP9/WP10 foundations.
- M6 gate GA1: end-to-end LDA/GGA molecular SCF; forces pass the FD ladder (50-verification/50).
- M12 gate GA2: alpha release; R0 batching >=5x10^3 single-points/hour on RTX (record H100 when possible);
  piecewise benchmark report #1 (rows 1–4 of 60-benchmarks/61).

## Phase B — Extended systems + linear scaling on workstations (M13–M30)
Scope: PBC/stress, WP5 R2 to 10^4 atoms on ONE card, XL-BOMD production, WP7 hybrids, 2-GPU node, surfaces.
- M18 gate GB1 (make-or-break): 2,000-atom a-Si:H — R2 energy within 0.5 meV/atom of R1 dense, on A40.
  Miss => fallback OMM/FOE for R2; +2 quarters.
- M24 gate GB2: XL-BOMD NVE 64-H2O, 100 ps, drift <=30 uHa/atom/ps, ~1 solve/step, on RTX.
- M30 gate GB3: 10^4-atom single-card run <=1 meV/atom; HSE06 slab <=4x own-PBE; QTT gate R-1
  (continue iff >=2x speed or >=4x memory at equal accuracy vs FFT path).

## Phase C — Scale-out + metals at scale (M31–M48; cluster begins)
- M36 GC1: weak scaling >=80% to 8 GPUs (10^5 a-Si).  - M42 GC2: 10^6-atom demo; benchmark paper #2.
- M48: QTT gate R-2 (merge to default or archive with tech report). PAW go/no-go M36 (input T7.6).

## Phase D — Hardening + v1.0 (M49–M60)
API freeze M51; external beta >=5 groups M54; final campaign + reproducibility archive (DOI) M57;
v1.0 + flagship paper + governance handoff M60.

## Standing rule
Analytical model first (each WP states its model); measurement at every gate; deviation >2x forces a
written model revision. This is the anti-(multiplicative-speedup-fallacy) discipline.
