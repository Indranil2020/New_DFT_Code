# Vision, scope, honest claims
## Vision
One open-source Kohn–Sham DFT engine, fastest-in-class across 10 -> 10^6 atoms, GPU-native and
mixed-precision-native, no machine learning in the core, runnable and verifiable on consumer GPUs.

## The middle way (why one code can cover the spectrum)
No single solver is optimal at both ends; one SUBSTRATE can be. Everything reduces to batched dense
operations on small matrix tiles (block-sparse NAO matrices) on tensor cores. A solver broker
dispatches each problem to the asymptotically right algorithm (R0 batch / R1 dense-ChFSI /
R2 purification / R3 Fermi-operator), all sharing one Hamiltonian builder, one force expression,
one input format, one test harness.

## Honest claims (contract with ourselves; every number at fixed, verified accuracy)
- 10–200 atoms, cold single point: within ~2x of the fastest GPU Gaussian codes. Not claimed: beating them.
- 10–200 atoms, throughput: >=10x per-GPU molecules/hour via native batching (R0).
- MD: within 5–20x of DFTB+ step rate at DZP-PBE quality via XL-BOMD (today's gap: 100–1000x).
- 10^3–10^4 atoms: fastest open-source time-to-solution at <=1 meV/atom on ONE GPU
  (10^4-atom validation fits a 24 GB RTX in mixed precision: ~0.5 MB/atom).
- 10^5 atoms: single-GPU capability headline (80 GB-class memory).
- 10^6 atoms: 8–32 GPUs at >=80% weak-scaling. Not claimed: 10^6 on one GPU (memory-impossible in 2026).
- Metals at scale: exact Mermin finite-Te DFT (R3). Not claimed: T=0 metallic linear scaling.
- No ML in core; hooks exist so others may attach ML XC externally.

## Non-goals (change only by RFC + explicit descope of something else)
TDDFT, GW/RPA, spin-orbit (until v1.0+), ML potentials, plane-wave mode.
