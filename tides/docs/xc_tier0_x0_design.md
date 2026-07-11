# XC Tier-0 X0 context and kernel design

## Context brief

The July 2026 Tier-0 execution plan is the governing XC architecture for this
work.  Its first deliverable is X0: a device-resident XC contract, SCF-scope
arena, analytic PW92 derivative shared with the CPU oracle, and a libxc-backed
rung-0 test.  The existing `core/grid/xc.cu` wrappers are host-vector demos:
they allocate, copy, synchronize, and return host arrays on every call.  They
remain legacy compatibility code until X1.6 can replace its callers.

The current `rho_build` does not produce analytic gradients from a density
matrix, and `vmat_build` has no GGA adjoint input.  Consequently X0 supports
only unpolarized LDA-PW92.  X1 must add PBE only after the rho/gradient and
vmat dependencies are ready; it must not route PBE through CPU libxc.

The X1 implementation now supplies borrowed device-view APIs for the missing
product/adjoint boundary: `BuildRhoGradientDevice` writes analytic rho and
three gradient planes from `P`, `phi`, and `grad_phi`; `BuildGgaVmatDevice`
consumes `w*v_rho` and `2*w*v_sigma*grad(rho)` directly.  These are
device-resident correctness kernels with no allocation or synchronization.
They deliberately do not adapt the old host-vector convenience wrappers.
The future TileMat/GEMM backend may replace their O(nbasis²) reference loops,
but must preserve this data contract and no-copy behavior.

## X0 acceptance criteria

1. `XcGridIn`/`XcGridOut` express 256-byte-aligned device SoA inputs and the
   exact weighted outputs consumed by `vmat_build`.
2. `XcArena::Reserve` allocates the padded SCF-scope buffers with
   `cudaMallocAsync`; `XcEval` performs neither allocation nor host
   synchronization.
3. The shared PW92 implementation has no finite-difference derivative and is
   used by the CPU atomgen reference and device LDA functor.
4. Host and device LDA-PW92 results compare against CPU libxc over the rung-0
   density lattice at the tolerance configured from `verification/tolerances.yaml`.

## Module boundaries

- `grid/xc/functionals/*.cuh`: host/device scalar arithmetic; no libxc or CUDA
  runtime dependency.
- `grid/xc/xc_engine.hpp`: frozen host API and non-owning device views.
- `grid/xc/xc_arena.hpp`: SCF-scope owner of the engine buffers.
- `grid/xc/xc_engine.cu`: launch-time dispatch and the one-thread-per-point
  LDA kernel.
- `grid/xc/kernels/xc_gga_kernel.cu`: fused PBE kernel: computes sigma from
  analytic gradient planes, writes `w*v_rho` and `2*w*v_sigma*grad(rho)`, and
  accumulates `w*rho*eps` without materializing eps or v arrays.
- `grid/rho_build_gpu.hpp` / `grid/vmat_build_gpu.hpp`: borrowed device-view
  rho/gradient and GGA-adjoint contracts; their direct kernels are correctness
  implementations pending TileMat/GEMM replacement.
- `grid/xc/tests/*`: external libxc oracle.  The tolerance is injected by
  CMake from the YAML source of truth rather than written into a test.

## LDA kernel traffic model

For each grid point the kernel loads `rho` and `w`, evaluates exchange and
correlation in registers, stores `w*v_rho`, and contributes `w*rho*eps` to one
FP64 atomic per warp.  It materializes neither `eps` nor an unweighted
potential.  This is the intended Tier-0 output shape; the legacy wrappers are
not part of the residency claim.

## Proof plan

Run syntax/configure, host rung-0 libxc oracle, CUDA device rung-0 oracle,
existing LDA regression, compute-sanitizer memcheck/racecheck, and a clean
worktree diff review.  X0 does not claim molecular energies, GGA adjointness,
or roofline performance; those are X1/X1.4/X1.5 gates.

## PBE-C numerical-oracle boundary

The PBE-C implementation uses an algebraically equivalent `log1p`/`expm1`
form rather than libxc's generated `log(1+x)` saturation expression.  This is
necessary for FP64 accuracy: on the required high-gradient lattice the
generated worker subtracts nearly equal terms in `v_sigma`, and its LDA seed
also loses significant digits near the density/sigma clamps.  The resulting
CPU-libxc value can differ from the mathematical PBE-C value by far more than
the rung-0 relative tolerance even though the in-tree functor is correct.

The PBE rung-0 test therefore retains CPU libxc as the raw oracle wherever it
agrees with an analytic extended-precision PBE-C reference at the YAML
tolerance.  At points where libxc itself fails that check, the functor is
required to meet the same `5e-14` tolerance against the extended-precision
reference and the test reports the divergence count.  Exchange remains a
strict raw-libxc comparison, and density/sigma threshold boundary cases remain
explicitly compared to libxc semantics.  This is a documented accuracy
override of a numerically unstable oracle value, not a relaxed acceptance bar.

## Validation log (2026-07-10)

- CPU-only configure/build: PASS.  This compiles the public contract without
  enabling CUDA and runs the host rung-0 and atomgen LDA tests.
- Host rung-0 against libxc 7.0.0: PASS.  Maximum relative errors were
  `6.04e-16` for `eps` and `3.69e-16` for `v_rho`, below `5e-14`.
- CUDA configure/build: PASS for `tides_cuda_grid`, the Tier-0 device oracle,
  and the legacy CUDA-XC regression executable.
- Device execution and compute-sanitizer: BLOCKED by the local runtime.  The
  CUDA runtime probe reports `cudaGetDeviceCount: no CUDA-capable device is
  detected` despite `nvidia-smi` visibility; compute-sanitizer also cannot
  load `libsanitizer-collection.so`.  These are infrastructure failures, not
  passing GPU evidence, and must be rerun on a usable CUDA worker.
- PBE host rung-0: PASS on 3,081 points (64 densities × 48 reduced gradients
  plus nine threshold cases).  The stable PBE-C functor is within `8.57e-15`
  for `v_rho` and `3.16e-15` for `v_sigma` versus the extended-precision
  analytic reference; raw libxc precision divergences are reported by the
  test rather than silently counted as functor failures.
- PBE engine/adjoint: CUDA compilation PASS for the fused PBE kernel, the
  deterministic fixed-order energy mode, and the borrowed-device
  rho/gradient→PBE→GGA-vmat pipeline test.  The legacy `XCEvalPbeCuda`
  CPU-libxc/GPU-reduction path has been removed.  Host rung-1 PBE adjoint
  PASS: 100 random symmetric density-matrix directions, maximum central-FD
  error `8.60e-11` against the YAML `1e-9` criterion.
- Device PBE and device rho→PBE→vmat tests: COMPILE PASS, EXECUTION BLOCKED
  by the same CUDA runtime visibility failure described above.  They must be
  run, with compute-sanitizer, on a usable CUDA worker before an X1 GPU gate
  can be claimed.
