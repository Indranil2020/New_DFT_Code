# Hardware strategy — workstation-first (REVISED per PI)
## Ground truth
Primary development and Phase A/B science: local workstations with RTX-class 24 GB cards; A40 48 GB
available; H100 occasional. Cluster access assumed only from Phase C.

## Design law (consequences, not preferences)
1. RTX/A40 native FP64 is ~1/64 of FP32. The mixed-precision + Ozaki-emulated-FP64 path (20-math/21)
   is therefore EXISTENTIAL: FP64-critical reductions run as tensor-core emulation from day one.
   CI contract: the full verification suite passes nightly on one RTX card.
2. DZP mixed-precision footprint ~0.5 MB/atom => 10^4 atoms ~ 5–10 GB: ALL R2/R3 algorithms are
   developed and validated on owned workstations. Only scaling beyond ~3x10^4 atoms waits for Phase C.
3. Every performance observable names its device (RTX primary; A40 memory tests; H100 ceiling).
   No Definition-of-Done may require hardware the team does not own before Phase C.
4. Multi-GPU starts as single-node (2 cards, NCCL). MPI/NVSHMEM multi-node is Phase C.

## Device table (fill exact SKUs at kickoff; run `tides tune` per machine)
| Device | VRAM | native FP64 | tensor cores | role |
|---|---|---|---|---|
| RTX 4090-class | 24 GB | ~1/64 FP32 | ~2 orders above FP64 | primary CI + dev |
| A40 | 48 GB | ~1/64 FP32 | high | memory-heavy tests |
| H100 (occasional) | 80 GB | strong | very high | ceiling reference |
