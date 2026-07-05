# Parallelism & I/O by phase
Phase A: single GPU; CUDA graphs; host<->device transfers only at stage boundaries.
Phase B: single-node multi-GPU (2 cards realistic on team workstations): NCCL collectives;
tile-graph partitioning (METIS) with imbalance <=10%; halo exchange overlapped with compute
(exposed comm <=15% of step time on 4096-H2O proxy).
Phase C: MPI + NVSHMEM one-sided halos; hierarchical Poisson; restart-safe long campaigns.
I/O: HDF5 for stage dumps/restarts (31); JSON-lines structured logs consumed directly by the
benchmark dashboard (no log scraping); NVML/rocm-smi energy metering wired into every run record.
