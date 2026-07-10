# Comprehensive GPU XC Implementation Plan for TIDES
## Complete Survey, Literature, and Execution Roadmap (July 2026)

**Objective:** Port the entire libxc functional catalog to native GPU CUDA `__device__` kernels for the TIDES DFT code, eliminating all CPU↔GPU round-trips in the SCF XC pipeline.

**Chosen Path:** Libxc Maple Source → CUDA C++ Auto-Generation Pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem: Why This Matters](#2-the-problem-why-this-matters)
3. [Historical Evolution of GPU XC (2008–2026)](#3-historical-evolution-of-gpu-xc-20082026)
4. [Complete Survey of All GPU XC Libraries & Projects](#4-complete-survey-of-all-gpu-xc-libraries--projects)
5. [Deep Dive: Libxc Architecture & Maple Generation Pipeline](#5-deep-dive-libxc-architecture--maple-generation-pipeline)
6. [The Chosen Path: Maple → CUDA C++](#6-the-chosen-path-maple--cuda-c)
7. [Alternative Paths Analyzed and Rejected](#7-alternative-paths-analyzed-and-rejected)
8. [Required Reading & Key Papers](#8-required-reading--key-papers)
9. [All GitHub / GitLab / Source Links](#9-all-github--gitlab--source-links)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Risk Assessment & Mitigation](#11-risk-assessment--mitigation)
12. [Appendix: Functional Coverage Matrices](#12-appendix-functional-coverage-matrices)

---

## 1. Executive Summary

As of July 2026, **no complete, production-ready, native-CUDA implementation of all 700+ libxc functionals exists.** The landscape is fragmented:

- **Libxc** has an experimental CUDA/HIP backend (as of May 2026), but coverage of all functionals is unverified and it requires autotools (`--enable-cuda`), with no CMake support.
- **ExchCXX** provides mature device-side CUDA/HIP kernels but only covers ~10 common functionals (LDA, PBE, B88, LYP, etc.).
- **JAX-XC** covers ~95% of libxc functionals but runs through JAX/XLA (Python), not native CUDA, and cannot be embedded in a C++/CUDA codebase.
- **gpu4pyscf** achieved significant speedups by building custom GPU XC wrappers around libxc, but still relies on CPU libxc for many functionals.
- **TeraChem** has proprietary GPU XC kernels but is closed-source and limited to specific functionals.

**The only scalable path to "all libxc functionals on GPU native" is to adapt libxc's existing Maple-to-C code generation pipeline to emit CUDA C++ `__device__` functions.** This document provides the complete survey, history, literature, and execution plan.

---

## 2. The Problem: Why This Matters

### 2.1 The gpu4pyscf Bottleneck (Your Experience)

In GPU4PySCF and similar codes, the XC evaluation follows this anti-pattern:

```
GPU: rho_build → [D2H transfer] → CPU: libxc EvalPBEOnGrid → [H2D transfer] → GPU: energy reduction
```

This CPU↔GPU round-trip is the bottleneck. Your observation of **20× speedup** on RTX by moving the entire LDA XC pipeline to GPU confirms the literature: the data transfer cost dominates for small-to-medium systems.

### 2.2 The Ideal Fused Pipeline (What TIDES Needs)

```
GPU: rho_build → GPU: XC_eval (device kernel) → GPU: vmat_build → GPU: H_xc
         ↑___________________________________________________________↓
                    (one scalar energy back to CPU per SCF iteration)
```

This is the architecture used by GauXC + ExchCXX and is the established best practice for GPU DFT.

---

## 3. Historical Evolution of GPU XC (2008–2026)

### 3.1 Early Era (2008–2012): Proof of Concept

- **2008–2009:** Ufimtsev & Martínez (Stanford) publish foundational GPU quantum chemistry papers. TeraChem development begins. First GPU ERI and SCF implementations.
- **2010:** Yasuda publishes GPU DFT XC implementation — one of the first to evaluate XC on GPU. TeraChem's dissertation (Chapter 4) references this work for the three-part GPU XC kernel design:
  1. Evaluate density and gradient at grid points
  2. Evaluate DFT functional at each grid point
  3. Sum functional values to produce matrix elements

### 3.2 Libxc Era (2012–2017): CPU-Centric Standardization

- **2012:** Marques, Oliveira & Burnus publish libxc v1.0 paper (*Comput. Phys. Commun.* 183, 2272). All functionals hand-written in C.
- **2017:** Libxc v4.0 introduces **Maple-based automatic code generation**. Lehtola et al. rewrite almost all functionals in Maple. A driver script (`maple2c.py`) generates C source. This is the critical inflection point — the mathematical definitions are now separated from the CPU implementation pattern.
- **2018:** Lehtola, Steigemann, Oliveira & Marques publish the modern libxc reference (*SoftwareX* 7, 1). Library now contains 48 LDAs, 261 GGAs, 92 mGGAs.

### 3.3 GPU DFT Maturation (2019–2023)

- **2019–2020:** Williams-Young (LBNL) develops **ExchCXX** and **GauXC** under the DOE Exascale Computing Project (NWChemEx). ExchCXX provides the first mature, open-source, device-native CUDA XC kernels for a subset of functionals. GauXC integrates these into a full Gaussian-basis GPU DFT library.
- **2020:** Williams-Young et al. publish "On the Efficient Evaluation of the Exchange Correlation Potential on Graphics Processing Unit Clusters" (*Frontiers in Chemistry* 8, 581058). Introduces batched kernel invocations, batched GEMM/SYR2K for XC matrix formation, and three-level GPU parallelism. Demonstrates linear strong scaling on V100.
- **2021:** Williams-Young et al. publish performance portability paper (*Parallel Computing* 108, 102829) — extends GauXC/ExchCXX to HIP (AMD) and SYCL (Intel).
- **2021–2023:** GPU4PySCF development accelerates. Caltech team (Chan, Sun, Li) builds GPU-accelerated PySCF plugin. Initial version focuses on 4-center ERIs and direct SCF. XC still relies on libxc CPU with custom wrappers.
- **2023:** JAX-XC published at ICLR 2023 Workshop (Sea AI Lab, Singapore). Machine-translates libxc Maple definitions to JAX. Covers most but not all functionals. Runs on GPU via XLA, not native CUDA.

### 3.4 Recent Era (2024–2026): Convergence and Gaps

- **2024:** GPU4PySCF v1.0 released (ByteDance Research). 30× speedup over 32-core CPU. Still uses libxc CPU for XC with custom GPU wrappers. Paper: Wu et al., *J. Phys. Chem. A* (2024) and Li et al., arXiv:2407.09700.
- **2024:** GPU4PySCF v1.6.0 refactors libxc interface, removes rarely used functionals. Third-order XC derivatives can be evaluated on GPU (requires gpu4pyscf-libxc 0.7).
- **2025:** Stocks & Barca publish "Efficient Algorithms for GPU Accelerated Evaluation of the DFT Exchange-Correlation Functional" (*J. Chem. Theory Comput.* 21, 20, 10263–10280). Comparative study of 4 GPU algorithms. Batched XC matrix formation from density matrix yields 1.4–5.2× speedup over leading GPU KS-DFT codes.
- **May 2026:** Libxc README updated to state: "Libxc supports GPU execution on both NVIDIA (CUDA) and AMD (HIP) hardware." Enabled via `--enable-cuda` with autotools. **CMake is NOT supported for GPU builds.** When compiled with CUDA, libxc always expects GPU pointers. Coverage of all 700+ functionals is unverified.
- **June 2026:** GPU4PySCF v1.7.0 released. Adds cuEST integration, Stratmann-improved Becke grid, TDDFT-FSSH dynamics.

---

## 4. Complete Survey of All GPU XC Libraries & Projects

### 4.1 Libxc (The Source of Truth)

| Attribute | Details |
|-----------|---------|
| **Official Repo** | https://gitlab.com/libxc/libxc.git |
| **Website** | https://libxc.gitlab.io/ |
| **GitHub Mirror** | https://github.com/molpro/libxc |
| **License** | MPL 2.0 |
| **Functionals** | 700+ (LDA, GGA, mGGA, Hybrid, RSH) |
| **GPU Support** | Experimental CUDA/HIP (May 2026) |
| **GPU Build** | `--enable-cuda` with autotools ONLY. CMake NOT supported. |
| **GPU Pointer Behavior** | Always expects GPU pointers when compiled with CUDA. No runtime CPU/GPU dispatch. |
| **Generation** | Maple → C (since v4, 2017) |
| **Script** | `python3 scripts/maple2c.py --functional=NAME --maxorder=4` |
| **Status** | Experimental CUDA. Coverage matrix unpublished. |

**Key Insight:** Since libxc functionals are auto-generated from Maple, the CUDA port likely compiles the same C code through `nvcc -x cu`. Most functionals *should* work, but performance and robustness on the full catalog are unverified.

### 4.2 ExchCXX (The Gold Standard for Device-Native XC)

| Attribute | Details |
|-----------|---------|
| **Repo** | https://github.com/wavefunction91/ExchCXX |
| **Developer** | David B. Williams-Young (LBNL) |
| **License** | Modified 3-Clause BSD |
| **GPU Backends** | CUDA (NVIDIA), HIP (AMD), SYCL (Intel, experimental) |
| **Functionals** | ~10 (LDA: Slater, VWN3/5, PW91, PZ86; GGA: PBE X/C, revPBE, B88, LYP) |
| **API** | `eval_exc_vxc_device()` — pure device-side, zero CPU round-trip |
| **Used By** | GauXC, NWChemEx, MPQC, ExaChem |
| **Status** | Mature, actively maintained |

**Device-Supported Functionals:**

| Name | Libxc ID | ExchCXX ID | Device? |
|------|----------|------------|---------|
| Slater Exchange | `XC_LDA_X` | `Kernel::SlaterExchange` | Y |
| VWN III | `XC_LDA_C_VWN_3` | `Kernel::VWN3` | Y |
| VWN V | `XC_LDA_C_VWN_RPA` | `Kernel::VWN5` | Y |
| PW91 (LDA) | `XC_LDA_C_PW` | `Kernel::PW91_LDA` | Y |
| PW91 Mod | `XC_LDA_C_PW_MOD` | `Kernel::PW91_LDA_MOD` | Y |
| PW91 RPA | `XC_LDA_C_PW_RPA` | `Kernel::PW91_LDA_RPA` | Y |
| PZ86 | `XC_LDA_C_PZ` | `Kernel::PZ86_LDA` | Y |
| PZ86 Mod | `XC_LDA_C_PZ_MOD` | `Kernel::PZ86_LDA_MOD` | Y |
| PBE Exchange | `XC_GGA_X_PBE` | `Kernel::PBE_X` | Y |
| PBE Correlation | `XC_GGA_C_PBE` | `Kernel::PBE_C` | Y |
| revPBE Exchange | `XC_GGA_X_PBE_R` | `Kernel::revPBE_X` | Y |
| Becke 88 | `XC_GGA_X_B88` | `Kernel::B88` | Y |
| Lee-Yang-Parr | `XC_GGA_C_LYP` | `Kernel::LYP` | Y |

**Critical Limitation:** Only covers the "common 10" that drive 95% of production DFT. SCAN, r²SCAN, TPSS, Minnesota functionals, etc. are absent.

### 4.3 GauXC (The Full GPU DFT Driver)

| Attribute | Details |
|-----------|---------|
| **Repo** | https://github.com/wavefunction91/GauXC (implied from ExchCXX page) |
| **Developer** | David B. Williams-Young (LBNL) |
| **License** | BSD (similar to ExchCXX) |
| **Purpose** | Modern C++ library for massively parallel, GPU-accelerated Gaussian-basis DFT |
| **XC Backend** | ExchCXX |
| **Integrals** | LibintX (for DF-J engine) |
| **Used By** | NWChemEx, MPQC, ExaChem |
| **Status** | Mature, primary DFT driver for NWChemEx |

**Key Papers:**
- Williams-Young et al., "On the Efficient Evaluation of the Exchange Correlation Potential on Graphics Processing Unit Clusters," *Frontiers in Chemistry* 8, 581058 (2020). DOI: 10.3389/fchem.2020.581058
- Williams-Young et al., "Achieving performance portability in Gaussian basis set density functional theory on accelerator based architectures in NWChemEx," *Parallel Computing* 108, 102829 (2021). DOI: 10.1016/j.parco.2021.102829
- Williams-Young et al., "Distributed memory, GPU accelerated Fock construction for..." *J. Chem. Phys.* (2023). DOI: 10.1063/5.0151070

### 4.4 JAX-XC (The Differentiable Alternative)

| Attribute | Details |
|-----------|---------|
| **Repo** | https://github.com/sail-sg/jax_xc |
| **Developer** | Sea AI Lab (Singapore) |
| **License** | Apache 2.0 |
| **Base** | libxc 6.0.0 Maple definitions |
| **Coverage** | ~95% of libxc (~24 functionals missing) |
| **GPU Stack** | JAX/XLA (NOT native CUDA) |
| **Differentiable** | Yes (end-to-end autodiff) |
| **Last Update** | December 2022 (no updates since) |
| **Embeddable in C++/CUDA** | **NO** — requires Python/JAX runtime |

**Missing Functionals in JAX-XC (24 total):**

| Reason | Functionals |
|--------|-------------|
| **Becke-Roussel — no closed-form** (requires numerical C solver) | `gga_x_fd_lb94`, `gga_x_fd_revlb94`, `gga_x_gg99`, `gga_x_kgg99`, `hyb_gga_xc_case21`, `hyb_mgga_xc_b94_hyb`, `hyb_mgga_xc_br3p86`, `mgga_c_b94`, `mgga_x_b00`, `mgga_x_bj06`, `mgga_x_br89`, `mgga_x_br89_1`, `mgga_x_mbr`, `mgga_x_mbrxc_bg`, `mgga_x_mbrxh_bg`, `mgga_x_mggac`, `mgga_x_rpp09`, `mgga_x_tb09` |
| **Requires explicit 1D integration** | `lda_x_1d_exponential`, `lda_x_1d_soft` |
| **JIT too slow for `exp1` / `E1_scaled`** | `gga_x_wpbeh`, `gga_c_ft97` |
| **VXC functional — not comparable** | `lda_xc_tih`, `gga_c_pbe_jrgx`, `gga_x_lb` |

**Why JAX-XC is NOT suitable for TIDES:**
1. Requires full JAX/XLA runtime — cannot link as a `.so` into C++
2. JIT compilation overhead on every new grid shape
3. No control over memory layout, register usage, or kernel fusion
4. The 24 missing functionals would still need a native fallback
5. Not maintained since 2022

### 4.5 gpu4pyscf (The Industrial Python Solution)

| Attribute | Details |
|-----------|---------|
| **Repo** | https://github.com/pyscf/gpu4pyscf |
| **Developers** | ByteDance Research, Caltech (Chan, Sun), Ohio State (Herbert) |
| **License** | Apache 2.0 |
| **GPU Stack** | CUDA + CuPy + custom CUDA kernels |
| **XC Approach** | Custom Python API wrapping libxc CPU; some GPU-native XC (gpu4pyscf-libxc) |
| **Speedup** | 30× over 32-core CPU (v1.0) |
| **Latest** | v1.7.0 (June 2026) |

**Key Papers:**
- Li, Sun, Zhang, Chan, "Introducing GPU-acceleration into the Python-based Simulations of Chemistry Framework," arXiv:2407.09700 (2024).
- Wu, Sun, Pu, Zheng, Ma, Yan, Xia, Wu, Huo, Li, Ren, Gong, Zhang, Gao, "Enhancing GPU-acceleration in the Python-based Simulations of Chemistry Framework," arXiv:2404.09452 (2024).
- Also published in *J. Phys. Chem. A* (2025): DOI: 10.1021/acs.jpca.4c05876

**Key Release Notes:**
- v1.6.0 (March 2025): "Refactored the libxc interface and removed rarely used XC functionals. Third-order XC derivatives can be evaluated on GPU (requires gpu4pyscf-libxc 0.7)."
- v1.7.0 (June 2026): cuEST integration, TDDFT-FSSH, Stratmann-improved Becke grid.

**Critical Quote from GPU4PySCF Paper (2024):**
> "CUDA support is still experimental in the latest version of libXC (v6.2)."

This confirms that even the most advanced industrial GPU DFT code still treats libxc CUDA as experimental.

### 4.6 NWChemEx (The Exascale Framework)

| Attribute | Details |
|-----------|---------|
| **Repo** | https://github.com/NWChemEx-Project |
| **Funding** | DOE Exascale Computing Project (17-SC-20-SC) |
| **License** | Apache 2.0 |
| **GPU Model** | SYCL/DPC++ (Intel), CUDA (NVIDIA), HIP (AMD) |
| **DFT Driver** | GauXC + ExchCXX |
| **Status** | Active development, targeting exascale |

**Key Papers:**
- "From NWChem to NWChemEx" — eScholarship (2020s). Documents rewrite from Fortran to modern C++ with GPU support planned from inception.
- ALCF Case Study: https://www.alcf.anl.gov/science/case-studies/high-performance-computational-chemistry-quantum-level-nwchemex

**Related: ExaChem**
- Repo: https://github.com/ExaChem/exachem
- Uses TAMM tensor library + GauXC for DFT
- Apache 2.0 license

**Related: PWDFT (Plane-Wave for NWChemEx)**
- Repo: https://github.com/ebylaska/PWDFT
- SYCL-based plane-wave DFT for Intel GPUs (Aurora)

### 4.7 TeraChem (The Proprietary Pioneer)

| Attribute | Details |
|-----------|---------|
| **Developer** | Martínez Group (Stanford) / PetaChem, LLC |
| **License** | Proprietary / Commercial |
| **GPU Stack** | CUDA (NVIDIA-only) |
| **First Release** | 2008–2009 (Ufimtsev & Martínez papers) |
| **Methods** | HF, DFT, TDDFT, CIS, CASSCF, MP2, CCSD, xTB |
| **GPU XC** | Native CUDA kernels (proprietary) |

**Key Papers:**
- Ufimtsev & Martínez, "Quantum chemistry on GPUs," *J. Chem. Theory Comput.* (2008–2009).
- "Extending GPU-Accelerated Gaussian Integrals in the TeraChem Software Package to f Type Orbitals," *J. Chem. Phys.* 161, 174118 (2024). DOI: 10.1063/5.0219997
- Comprehensive review: *J. Chem. Theory Comput.* (2020).

**Key Insight from TeraChem Dissertation (Stanford, Chapter 4):**
TeraChem's GPU XC implementation follows Yasuda's three-part design:
1. Evaluate electronic density and gradient at each grid point from density matrix
2. Evaluate DFT functional at each grid point
3. Sum functional values to produce final matrix elements

Kernels are segmented by angular momentum pairs (ss, sp, sd, pp, pd, dd for s,p,d basis). Each thread handles one primitive shell pair, looping through all grid points.

### 4.8 ABACUS (The Chinese Open-Source Contender)

| Attribute | Details |
|-----------|---------|
| **Repo** | https://github.com/deepmodeling/abacus-develop |
| **Community** | DeepModeling |
| **License** | GNU GPLv3 |
| **Basis** | Plane wave + Numerical Atomic Orbitals (NAO) |
| **GPU Support** | Yes (CUDA) |
| **XC** | Internal + Libxc (all LDA, GGA, mGGA, hybrid) |
| **Status** | Active, v3.9 released December 2024 |

**Documentation:** https://abacus.deepmodeling.com/en/latest/

### 4.9 Quantum ESPRESSO GPU (The Plane-Wave Legacy)

| Attribute | Details |
|-----------|---------|
| **GPU Plugin** | https://github.com/fspiga/qe-gpu-plugin |
| **Original Author** | Filippo Spiga |
| **GPU Stack** | CUDA (PWscf, NEB) |
| **Supported GPUs** | Fermi, Kepler, Pascal (legacy — M2070 to P100) |
| **QE Version** | 5.4 (plugin era) |
| **Modern QE GPU** | NVIDIA HPC SDK container (QE 7.2, CUDA 12.3) |

**Container:** https://github.com/anj1/quantum-espresso-container

**Note:** Quantum ESPRESSO's GPU path is primarily for plane-wave FFT and linear algebra, not for native XC kernel evaluation. XC in QE is typically evaluated on CPU or via simple GPU offloading.

### 4.10 Other Notable Mentions

| Project | Link | Notes |
|---------|------|-------|
| **Jrystal** | https://github.com/sail-sg/jrystal | JAX-based differentiable DFT for solids (plane wave). Uses JAX-XC. |
| **BrianQC** | Q-Chem backend | GPU-accelerated integrals, proprietary. Compared against TeraChem in 2024 paper. |
| **BigChem** | https://github.com/mtzgroup/bigchem | Distributed QM with TeraChem GPU support. |
| **Inq** | Tuckerman et al. (NYU) | Plane-wave DFT with GPU support. Emphasizes CPU↔GPU data transfer minimization. |
| **RMG** | https://github.com/ReactionMechanismGenerator/RMG-Py | Microkinetic mechanisms, not DFT XC. |

---

## 5. Deep Dive: Libxc Architecture & Maple Generation Pipeline

### 5.1 The Maple Revolution (2017)

Libxc switched to automatic code generation with Maple in version 4 (2017). Previous versions (≤3) employed hand-written C implementations.

**Reference:**
> "Libxc switched to automatical code generation with Maple in version 4 in 2017, while previous versions employed hand-written C implementations." — libxc.gitlab.io

**Why Maple?**
- Computer algebra systems (CAS) can derive and implement analytical derivatives of complex functionals automatically.
- Maple 2015 introduced optimized C output, eliminating overhead of CAS-generated code.
- The autogeneration approach speeds up introduction of new functionals and increases reliability.
- Hand-written derivatives are tedious, prone to bugs, and difficult to maintain for 700+ functionals.

**Exception:** A few functionals could not be reimplemented in Maple due to technical reasons, e.g., the **Becke-Roussel exchange functional** which does not have a closed analytic form (relies on a transcendental equation requiring numerical C solver).

### 5.2 Directory Structure

```
libxc/
├── maple/                  # Maple source files for all functionals
│   ├── lda/
│   ├── gga/
│   ├── mgga/
│   └── ...
├── scripts/
│   ├── maple2c.py          # MAIN SCRIPT: Maple → C generator
│   └── get_functional_info.py
├── src/
│   ├── maple2c/            # Auto-generated C kernels from Maple
│   ├── work_lda.c          # Generic LDA interface (work functions)
│   ├── work_gga_x.c        # Generic GGA exchange interface
│   ├── work_gga_c.c        # Generic GGA correlation interface
│   ├── work_mgga_x.c       # Generic mGGA exchange interface
│   ├── work_mgga_c.c       # Generic mGGA correlation interface
│   └── funcs_*.c           # Generated functional registration tables
└── configure.ac            # Autotools config (CUDA: --enable-cuda)
```

### 5.3 The Generation Pipeline

**Step 1: Write functional in Maple**
```maple
# Example: maple/gga_x_pbe.mpl
# Defines the PBE exchange energy density as a function of s (reduced density gradient)
```

**Step 2: Generate C source**
```bash
python3 scripts/maple2c.py --functional=gga_x_pbe --maxorder=4
```
This produces:
- `src/maple2c/gga_x_pbe.c` — the auto-generated C kernel
- Contains functions like `xc_gga_x_pbe`, `xc_gga_x_pbe_first_order`, etc.
- Derivatives up to 4th order (for Hessian, TDDFT, etc.)

**Step 3: Register functional in libxc**
Write a definition file in `src/` containing:
1. `#define` macro with numerical functional identifier
2. External parameter declarations (if any)
3. `xc_func_info_type` constructor (type, references, flags, thresholds, worker functions)

**Step 4: Regenerate functional tables**
```bash
cd src
make funcs
# OR
python3 ../scripts/get_functional_info.py --srcdir=..
```
This generates:
- `funcs_lda.c`, `funcs_gga.c`, `funcs_mgga.c`
- `xc_funcs.h`, `xc_funcs_worker.h`, `libxc_inc.f90`

### 5.4 The "Work Functions" Pattern

Libxc uses generic **work functions** that dispatch to the specific functional kernels:

- `work_lda.c` — loops over grid points, calls LDA kernel for each point
- `work_gga_x.c` — loops over grid points, computes reduced gradient `s`, calls GGA exchange kernel
- `work_gga_c.c` — similar for GGA correlation
- `work_mgga_x.c`, `work_mgga_c.c` — for meta-GGA

**The CPU-centric pattern:**
```c
// work_lda.c (simplified)
for (int ip = 0; ip < np; ip++) {
    double rho = rho_in[ip];
    double eps, vrho;
    xc_lda_kernel(&rho, &eps, &vrho, 1);  // call generated kernel
    exc[ip] = eps;
    vxc[ip] = vrho;
}
```

**The GPU opportunity:** The inner kernel call (`xc_lda_kernel`) is pure arithmetic. The outer loop is the grid iteration pattern. If we replace the outer loop with a CUDA kernel launch (one thread per grid point), and mark the inner kernel as `__device__`, we get native GPU evaluation.

### 5.5 Libxc Experimental CUDA Backend (May 2026)

**Build instructions (from search results):**
```bash
git clone https://gitlab.com/libxc/libxc.git
cd libxc
autoreconf -i
./configure --enable-cuda
make
sudo make install
```

**Critical limitations:**
1. **Autotools only** — CMake is NOT supported for GPU builds.
2. **Always GPU pointers** — When compiled with `--enable-cuda`, libxc ALWAYS expects GPU device pointers. There is no runtime dispatch to handle both CPU and GPU arrays.
3. **Coverage unknown** — No published matrix of which of the 700+ functionals work on GPU.
4. **Experimental** — GPU4PySCF paper (2024) explicitly calls it "experimental in libXC v6.2."

**Hypothesis on how it works:**
Since the generated C code is standard C99 with no CPU-specific intrinsics, compiling it with `nvcc -x cu` (treating C as CUDA) likely makes most functionals work. The `static inline` functions become `__device__` functions when compiled with nvcc. The work functions would need to be wrapped in `__global__` kernels.

---

## 6. The Chosen Path: Maple → CUDA C++

### 6.1 Philosophy

Instead of trying to extract PTX from JAX-XC (impossible), or waiting for libxc CUDA to mature (uncertain), or limiting ourselves to ExchCXX's 10 functionals (insufficient), we will **adapt libxc's existing Maple-to-C pipeline to emit CUDA C++ `__device__` functions** and write our own `__global__` grid evaluation kernels.

### 6.2 Technical Strategy

**Phase 1: Adapt the Generation Scripts**

1. **Fork libxc** from https://gitlab.com/libxc/libxc.git
2. **Modify `scripts/maple2c.py`** to:
   - Add `__device__` qualifier to generated functions (instead of `static inline`)
   - Add `__host__ __device__` for functions that need to run on both CPU and GPU
   - Replace `double` with templated types if mixed precision is desired later
   - Ensure all generated code is CUDA C++ compatible (no VLAs, no variable-length arrays, no C99 features unsupported by nvcc)

3. **Regenerate all functional kernels:**
   ```bash
   for func in $(cat functional_list.txt); do
       python3 scripts/maple2c.py --functional=$func --maxorder=2
   done
   ```
   (Start with maxorder=2 for SCF; add 3rd/4th order later for Hessian/TDDFT)

**Phase 2: Write CUDA Grid Evaluation Kernels**

Replace libxc's `work_*.c` CPU loops with CUDA `__global__` kernels:

```cuda
// Example: LDA XC evaluation kernel
__global__ void xc_lda_eval_kernel(
    int npts,
    const double* rho,
    double* exc,
    double* vrho,
    int func_id)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= npts) return;

    double r = rho[ip];
    double e, v;

    // Dispatch to device functional kernel
    xc_lda_kernel_device(r, &e, &v, func_id);

    exc[ip] = e;
    vrho[ip] = v;
}
```

For GGA:
```cuda
__global__ void xc_gga_eval_kernel(
    int npts,
    const double* rho,
    const double* sigma,  // |∇ρ|²
    double* exc,
    double* vrho,
    double* vsigma,
    int func_id)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= npts) return;

    double r = rho[ip];
    double s = sigma[ip];
    double e, vr, vs;

    xc_gga_kernel_device(r, s, &e, &vr, &vs, func_id);

    exc[ip] = e;
    vrho[ip] = vr;
    vsigma[ip] = vs;
}
```

**Phase 3: Handle the Numerical Functionals**

For the ~24 functionals that cannot be expressed in closed-form Maple (Becke-Roussel type, 1D integration, etc.):

1. **Port numerical C solvers to CUDA:**
   - Implement Newton-Raphson iteration on device
   - Use lookup tables + interpolation for transcendental equations
   - Test convergence and stability across the full density range

2. **Fallback strategy:**
   - Evaluate these on CPU in batches
   - Copy results to GPU
   - Accept the transfer cost for exotic functionals (they are rarely used in production)

**Phase 4: Integrate into TIDES**

1. **Fused pipeline:**
   ```
   rho_build (GPU) → XC_eval (GPU device kernel) → vmat_build (GPU kernel) → H_xc (GPU)
   ```

2. **vmat_build GPU kernel:**
   - Each thread block computes one tile of H_ij
   - Threads accumulate `v[g] * phi_i[g] * phi_j[g]` over grid points
   - Use shared memory for phi buffers
   - AtomicAdd or reduction tree for final H_ij

3. **Memory management:**
   - Keep `rho`, `sigma`, `tau`, `lapl`, `exc`, `vrho`, `vsigma`, `vtau`, `vlapl` all on device
   - Only transfer final energy scalar back to CPU

### 6.3 Expected Challenges

| Challenge | Mitigation |
|-----------|------------|
| Maple-generated code uses C99 features unsupported by nvcc | Audit and patch `maple2c.py` output; replace VLAs with fixed-size arrays or dynamic allocation |
| Register pressure in complex GGA/mGGA kernels | Profile with `ncu`; split complex kernels; use `__launch_bounds__` |
| Numerical stability of device-side Newton-Raphson | Extensive testing; compare against CPU libxc reference |
| Compilation time for 700+ functionals | Compile functionals into separate object files; link selectively |
| Hybrid/RSH functionals need EXX | Keep local DFT part on GPU; EXX is a separate problem (GauXC's sn-K or LibintX) |

---

## 7. Alternative Paths Analyzed and Rejected

### 7.1 JAX-XC → Native CUDA

**Verdict:** ❌ **Not feasible.**

- JAX-XC is Python functions using `jax.numpy`. GPU code is generated by XLA compiler, not hand-written CUDA.
- AOT serialization produces binaries that still require the full XLA runtime (`libjaxlib`).
- Extracting PTX is possible but requires reverse-engineering XLA's memory layout, buffer allocation, and calling conventions. Extremely fragile.
- JAX allows C++ → JAX (custom ops), but NOT the reverse (JAX → C++).

### 7.2 Libxc Experimental CUDA Backend

**Verdict:** ⚠️ **Viable fallback, but not primary path.**

- Claims GPU support but coverage of all 700+ functionals is unverified.
- Autotools-only build; no CMake support.
- Always expects GPU pointers — no runtime CPU/GPU dispatch.
- Can be used as a reference/validation tool, but relying on it for production is risky until a coverage matrix is published.

### 7.3 ExchCXX Expansion

**Verdict:** ⚠️ **Good for common functionals, insufficient for complete coverage.**

- Only ~10 functionals supported.
- Adding new functionals to ExchCXX is done by hand (as stated in their README: "generating these interfaces must currently be done by hand").
- For TIDES to be "the fastest DFT code," we need the full catalog, not just the common 10.

### 7.4 Hand-Write All Kernels

**Verdict:** ⚠️ **What you did for LDA; not scalable to 700+ functionals.**

- Writing CUDA kernels for PBE, BLYP, etc. by hand is feasible (you've done it).
- Writing kernels for 261 GGAs and 92 mGGAs by hand is not maintainable.
- The Maple generation pipeline exists precisely to avoid this.

### 7.5 OpenCL

**Verdict:** ❌ **Wrong tool for this job.**

- OpenCL on NVIDIA is a second-class citizen; significant performance penalty vs. CUDA.
- No mature OpenCL DFT infrastructure (no OpenCL BLAS equivalent to cuBLAS/MAGMA).
- Ecosystem is declining in HPC.

---

## 8. Required Reading & Key Papers

### 8.1 Libxc & Maple Generation

1. **Lehtola, Steigemann, Oliveira, Marques**, "Recent developments in Libxc — A comprehensive library of functionals for density functional theory," *SoftwareX* **7**, 1 (2018). DOI: 10.1016/j.softx.2017.11.002
   - The modern libxc reference. Describes Maple auto-generation.

2. **Marques, Oliveira, Burnus**, "Libxc: a library of exchange and correlation functionals for density functional theory," *Comput. Phys. Commun.* **183**, 2272 (2012). DOI: 10.1016/j.cpc.2012.05.007
   - Early libxc reference (versions ≤3, hand-written C).

3. **Lehtola & Marques**, "Reproducibility of density functional approximations: how new functionals should be reported," *J. Chem. Phys.* **159**, 114116 (2023). DOI: 10.1063/5.0167763
   - Why libxc documentation and version tracking matter.

### 8.2 GPU DFT & XC Evaluation

4. **Williams-Young, de Jong, van Dam, Yang**, "On the Efficient Evaluation of the Exchange Correlation Potential on Graphics Processing Unit Clusters," *Frontiers in Chemistry* **8**, 581058 (2020). DOI: 10.3389/fchem.2020.581058
   - Foundational paper on batched GPU XC evaluation. Three-level parallelism. Batched GEMM/SYR2K.

5. **Williams-Young, Bagusetty, de Jong, Doerfler, van Dam, Vázquez-Mayagoitia, Windus, Yang**, "Achieving performance portability in Gaussian basis set density functional theory on accelerator based architectures in NWChemEx," *Parallel Computing* **108**, 102829 (2021). DOI: 10.1016/j.parco.2021.102829
   - HIP and SYCL ports of ExchCXX/GauXC.

6. **Williams-Young et al.**, "Distributed memory, GPU accelerated Fock construction for..." *J. Chem. Phys.* (2023). DOI: 10.1063/5.0151070
   - GauXC + LibintX integration. Multi-GPU scaling on Perlmutter (A100).

7. **Stocks & Barca**, "Efficient Algorithms for GPU Accelerated Evaluation of the DFT Exchange-Correlation Functional," *J. Chem. Theory Comput.* **21**, 20, 10263–10280 (2025). DOI: 10.1021/acs.jctc.5c01229
   - Comparative study of 4 GPU XC algorithms. Batched density-matrix-based XC formation wins for large systems. 1.4–5.2× speedup over leading GPU codes.

### 8.3 GPU4PySCF & Industrial GPU DFT

8. **Li, Sun, Zhang, Chan**, "Introducing GPU-acceleration into the Python-based Simulations of Chemistry Framework," arXiv:2407.09700 (2024).
   - Initial GPU4PySCF paper. 4-center ERI focus.

9. **Wu, Sun, Pu, Zheng, Ma, Yan, Xia, Wu, Huo, Li, Ren, Gong, Zhang, Gao**, "Enhancing GPU-acceleration in the Python-based Simulations of Chemistry Framework," arXiv:2404.09452 (2024).
   - Full GPU4PySCF v1.0. DFT, gradients, Hessians, solvent models. 30× speedup.

10. **Qiita Article (Japanese)**, "gpu4pyscf を RTX 5000 シリーズで動かす" (2026).
    - Practical build instructions for Blackwell (RTX 5080). CUDA 12.8. `-DCUDA_ARCHITECTURES="86-real;89-real;120-real"`.

### 8.4 JAX-XC & Differentiable DFT

11. **Sea AI Lab**, "JAX-XC: Exchange Correlation Functionals Translated from libxc to JAX," ICLR 2023 Workshop on Machine Learning for Materials.
    - Machine translation of libxc Maple to JAX. ~95% coverage. 24 missing functionals documented.

12. **Jrystal Documentation**, https://sail-sg.github.io/jrystal/
    - JAX-based differentiable DFT for solids. Uses JAX-XC.

### 8.5 TeraChem & Historical GPU QC

13. **Ufimtsev & Martínez**, "Quantum chemistry on GPUs," *J. Chem. Theory Comput.* (2008–2009).
    - Foundational GPU quantum chemistry papers.

14. **TeraChem Dissertation (Stanford)**, Chapter 4: "Implementation of density functional theory exchange-correlation potentials on GPUs."
    - https://stacks.stanford.edu/file/druid:hb803mt5913/FinalDissertation-augmented.pdf
    - Three-part GPU XC kernel design. Angular-momentum-segmented kernels.

15. **J. Chem. Phys. 161, 174118 (2024)** — TeraChem f-orbital extension.
    - DOI: 10.1063/5.0219997. Kernel splitting for register pressure. Dynamic precision.

### 8.6 NWChemEx & Exascale

16. **"From NWChem to NWChemEx"**, eScholarship (2020s).
    - Documents rewrite from Fortran to C++, GitHub hosting, CI/CD, GPU planning from inception.

17. **ALCF Case Study**, "High-Performance Computational Chemistry at the Quantum Level: NWChemEx."
    - https://www.alcf.anl.gov/science/case-studies/high-performance-computational-chemistry-quantum-level-nwchemex
    - SYCL/DPC++ for Intel Aurora. CUDA-to-SYCL translation.

---

## 9. All GitHub / GitLab / Source Links

### Core Libraries

| Library | URL | Type |
|---------|-----|------|
| **Libxc (Official)** | https://gitlab.com/libxc/libxc.git | GitLab |
| **Libxc (GitHub Mirror)** | https://github.com/molpro/libxc | GitHub |
| **Libxc Website** | https://libxc.gitlab.io/ | Web |
| **ExchCXX** | https://github.com/wavefunction91/ExchCXX | GitHub |
| **GauXC** | https://github.com/wavefunction91/GauXC | GitHub |
| **JAX-XC** | https://github.com/sail-sg/jax_xc | GitHub |
| **Jrystal** | https://github.com/sail-sg/jrystal | GitHub |

### GPU DFT Codes

| Code | URL | Type |
|------|-----|------|
| **GPU4PySCF** | https://github.com/pyscf/gpu4pyscf | GitHub |
| **PySCF** | https://github.com/pyscf/pyscf | GitHub |
| **NWChemEx** | https://github.com/NWChemEx-Project | GitHub Org |
| **ExaChem** | https://github.com/ExaChem/exachem | GitHub |
| **PWDFT** | https://github.com/ebylaska/PWDFT | GitHub |
| **ABACUS** | https://github.com/deepmodeling/abacus-develop | GitHub |
| **Quantum ESPRESSO GPU Plugin** | https://github.com/fspiga/qe-gpu-plugin | GitHub |
| **QE GPU Container** | https://github.com/anj1/quantum-espresso-container | GitHub |

### Related Tools

| Tool | URL | Type |
|------|-----|------|
| **BigChem** | https://github.com/mtzgroup/bigchem | GitHub |
| **TiledArray** | https://github.com/ValeevGroup/tiledarray | GitHub |
| **scalapackpp** | https://github.com/wavefunction91/scalapackpp | GitHub |
| **blacspp** | https://github.com/wavefunction91/blacspp | GitHub |
| **GenELPA** | https://github.com/pplab/GenELPA | GitHub |
| **NVIDIA Eigensolver GPU** | https://github.com/NVIDIA/Eigensolver_gpu | GitHub |

### Developer Pages

| Developer | URL |
|-----------|-----|
| **David Williams-Young (LBNL)** | http://wavefunction91.github.io/software/ |
| **Susi Lehtola (libxc maintainer)** | https://libxc.gitlab.io/ |
| **Sea AI Lab (JAX-XC)** | https://sail-sg.github.io/ |
| **DeepModeling (ABACUS)** | https://deepmodeling.com/ |

---

## 10. Implementation Roadmap

### Milestone 0: Foundation (Week 1–2)
- [ ] Fork libxc from GitLab
- [ ] Audit `scripts/maple2c.py` and `maple/` directory structure
- [ ] Build libxc with `--enable-cuda` and test experimental backend on RTX
- [ ] Document which functionals work/don't work with libxc CUDA
- [ ] Profile current TIDES XC pipeline to quantify exact bottleneck

### Milestone 1: Maple → CUDA Proof of Concept (Week 3–6)
- [ ] Modify `maple2c.py` to emit `__device__` qualifiers
- [ ] Generate CUDA C++ for LDA functionals (Slater, VWN, PW91, PZ86)
- [ ] Write `__global__` LDA XC evaluation kernel
- [ ] Integrate into TIDES: replace CPU libxc LDA path with GPU-native LDA
- [ ] Validate against CPU libxc reference (energy, gradient, numerical thresholds)
- [ ] Benchmark: measure speedup vs. current TIDES LDA path

### Milestone 2: GGA Expansion (Week 7–12)
- [ ] Generate CUDA C++ for PBE, BLYP, B88, LYP GGA kernels
- [ ] Write `__global__` GGA XC evaluation kernel (handles rho + sigma)
- [ ] Handle GGA vmat_build on GPU (the adjoint map: V_xc → H_ij)
- [ ] Integrate into TIDES SCF loop: rho_build → XC_eval → vmat_build all on GPU
- [ ] Validate and benchmark against CPU reference
- [ ] Target: reproduce your 20× speedup for GGA on RTX

### Milestone 3: mGGA & Numerical Functionals (Week 13–18)
- [ ] Generate CUDA C++ for meta-GGA functionals (TPSS, SCAN, r²SCAN)
- [ ] Port numerical C solvers to CUDA (Becke-Roussel type)
- [ ] Implement fallback batch-CPU evaluation for problematic functionals
- [ ] Validate mGGA energies and gradients

### Milestone 4: Full Catalog & Optimization (Week 19–26)
- [ ] Batch-generate CUDA C++ for ALL 700+ libxc functionals
- [ ] Compile functionals into modular static libraries (selective linking)
- [ ] Profile with Nsight Compute; optimize register usage and occupancy
- [ ] Implement mixed-precision evaluation (FP32 for XC, FP64 for accumulation) if stable
- [ ] Add 3rd/4th order derivatives (for Hessian, TDDFT, NMR)

### Milestone 5: Integration & Production (Week 27–32)
- [ ] Full SCF loop: zero D2H/H2D transfers between rho_build → XC → vmat_build
- [ ] Multi-GPU support (domain decomposition or batch splitting)
- [ ] Documentation and regression test suite
- [ ] Publish benchmark results and contribute back to libxc community

---

## 11. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Libxc Maple code contains C99 features incompatible with nvcc | High | High | Audit and patch `maple2c.py`; maintain patch set; upstream fixes |
| Numerical solvers (Becke-Roussel) unstable on GPU | Medium | Medium | Extensive testing; fallback to CPU batch evaluation |
| Compilation time excessive for 700+ functionals | Medium | Medium | Modular compilation; selective linking; precompiled headers |
| Register pressure crashes complex kernels | Medium | High | Kernel splitting (TeraChem strategy); `__launch_bounds__`; profiling |
| libxc upstream changes break our patches | Medium | Medium | Pin to stable version; rebase periodically; maintain CI |
| Hybrid functionals need EXX on GPU | Medium | High | Use GauXC sn-K or LibintX for EXX; keep local XC on GPU |
| Accuracy loss in mixed-precision | Low | High | Keep FP64 for accumulation; test against FP64 reference |

---

## 12. Appendix: Functional Coverage Matrices

### 12.1 Libxc Functional Counts (as of 2018 paper)

| Type | Count | Notes |
|------|-------|-------|
| LDA | 48 | Includes 1D, 2D, spin-polarized variants |
| GGA | 261 | Exchange, correlation, combined |
| mGGA | 92 | Including kinetic energy functionals |
| Hybrid | Many | Global and range-separated |
| **Total** | **700+** | Largest functional library in existence |

### 12.2 GPU XC Library Coverage Comparison

| Library | LDA | GGA | mGGA | Hybrid | Total | Native GPU | Embeddable in C++ |
|---------|-----|-----|------|--------|-------|------------|-------------------|
| **Libxc (CPU)** | ✅ All | ✅ All | ✅ All | ✅ All | 700+ | ❌ | ✅ |
| **Libxc (Exp. CUDA)** | ? | ? | ? | ? | ? | ⚠️ | ✅ |
| **ExchCXX** | 8 | 4 | ❌ | ❌ | ~12 | ✅ | ✅ |
| **JAX-XC** | ~95% | ~95% | ~95% | ~95% | ~676 | ❌ (XLA) | ❌ |
| **TeraChem** | ✅ | ✅ (subset) | ❌ | ✅ | ~20 | ✅ | ❌ (proprietary) |
| **GPU4PySCF** | ✅ | ✅ | ✅ | ✅ | 700+ | ⚠️ (wrappers) | ❌ (Python) |
| **TIDES (Target)** | ✅ All | ✅ All | ✅ All | ✅ All | 700+ | ✅ | ✅ |

### 12.3 ExchCXX Detailed Coverage

| Family | Functional | Libxc ID | ExchCXX Kernel | Device? |
|--------|-----------|----------|----------------|---------|
| LDA | Slater Exchange | `XC_LDA_X` | `Kernel::SlaterExchange` | ✅ |
| LDA | VWN III | `XC_LDA_C_VWN_3` | `Kernel::VWN3` | ✅ |
| LDA | VWN V | `XC_LDA_C_VWN_RPA` | `Kernel::VWN5` | ✅ |
| LDA | PW91 | `XC_LDA_C_PW` | `Kernel::PW91_LDA` | ✅ |
| LDA | PW91 Mod | `XC_LDA_C_PW_MOD` | `Kernel::PW91_LDA_MOD` | ✅ |
| LDA | PW91 RPA | `XC_LDA_C_PW_RPA` | `Kernel::PW91_LDA_RPA` | ✅ |
| LDA | PZ86 | `XC_LDA_C_PZ` | `Kernel::PZ86_LDA` | ✅ |
| LDA | PZ86 Mod | `XC_LDA_C_PZ_MOD` | `Kernel::PZ86_LDA_MOD` | ✅ |
| GGA | PBE Exchange | `XC_GGA_X_PBE` | `Kernel::PBE_X` | ✅ |
| GGA | PBE Correlation | `XC_GGA_C_PBE` | `Kernel::PBE_C` | ✅ |
| GGA | revPBE Exchange | `XC_GGA_X_PBE_R` | `Kernel::revPBE_X` | ✅ |
| GGA | Becke 88 | `XC_GGA_X_B88` | `Kernel::B88` | ✅ |
| GGA | LYP | `XC_GGA_C_LYP` | `Kernel::LYP` | ✅ |

---

## Document Metadata

- **Author:** Compiled for TIDES DFT Code Project
- **Date:** July 2026
- **Search Coverage:** All searches conducted up to July 2026
- **Scope:** Complete survey of GPU XC evaluation for DFT, with focus on native CUDA implementation of full libxc catalog
- **Chosen Path:** Libxc Maple Source → CUDA C++ Auto-Generation
- **Status:** Planning Phase — Ready for Execution

---

*"The only scalable path to all libxc functionals on GPU native is to adapt libxc's existing Maple-to-C code generation pipeline to emit CUDA C++ `__device__` functions."*
