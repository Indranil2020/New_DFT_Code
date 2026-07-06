# TIDES Theory Manual — Derivations

This manual provides the mathematical derivations for every physics and math
module in TIDES. Per coding standard 35-coding-standards.md, a PR adding an
equation to code without its manual derivation is rejected.

## Table of Contents

1. [Forces: Hellmann–Feynman + Pulay + Grid + Dispersion](#1-forces)
2. [XL-BOMD Shadow Dynamics](#2-xl-bomd-shadow-dynamics)
3. [Density-Matrix Purification (SP2)](#3-density-matrix-purification-sp2)
4. [Fermi-Operator Expansion (FOE)](#4-fermi-operator-expansion-foe)
5. [Mixed-Precision Ozaki Scheme](#5-mixed-precision-ozaki-scheme)
6. [Energy Assembly](#6-energy-assembly)
7. [Stress Tensor](#7-stress-tensor)

---

## 1. Forces

### 1.1 Total Force Expression

The force on atom $I$ is $F_I = -\partial E / \partial R_I$. The total energy
$E[\rho, \{R_I\}]$ depends on positions both explicitly (ion-ion, electron-ion)
and implicitly (through the basis functions $\phi_\mu(\mathbf{r}; \mathbf{R}_I)$
which move with the atoms in the NAO representation). This gives four
contributions:

$$F_I = F_I^{\text{HF}} + F_I^{\text{Pulay}} + F_I^{\text{grid}} + F_I^{\text{disp}} + F_I^{\text{ion}}$$

### 1.2 Hellmann–Feynman Force

For a fixed density matrix $P$, the Hellmann–Feynman force is:

$$F_I^{\text{HF}} = -\text{Tr}\left(P \frac{\partial H}{\partial R_I}\right)$$

In the NAO basis, $H = T + V_{\text{ext}} + V_H[\rho] + V_{xc}[\rho]$, where the
kinetic $T$ and external $V_{\text{ext}}$ (nucleus-electron) have explicit
position dependence through the two-center integrals (T2.6 derivative streams).

For the two-center terms ($T$, $V_{nl}$ in Kleinman–Bylander form), the
derivative $\partial H_{\mu\nu} / \partial R_I$ is computed analytically from
the splined radial tables and spherical-harmonic rotation matrices (WP2 T2.6).

### 1.3 Pulay Force

Because the NAO basis functions move with the atoms, the density matrix $P$
has implicit position dependence. The Pulay force is:

$$F_I^{\text{Pulay}} = -\text{Tr}\left(\frac{\partial P}{\partial R_I} H\right) + \text{Tr}\left(P \frac{\partial S}{\partial R_I} \epsilon\right)$$

where $\epsilon$ is the energy-weighted density matrix:
$\epsilon_{\mu\nu} = \sum_k f_k \epsilon_k C_{\mu k} C_{\nu k}$.

Using the idempotency relation $P = P S P$ (for the projector onto occupied
space), the Pulay force can be written as:

$$F_I^{\text{Pulay}} = -2\text{Tr}\left(\frac{\partial S}{\partial R_I} \, \epsilon \, P\right) + \text{Tr}\left(\frac{\partial S}{\partial R_I} \, P \, H \, P\right)$$

This requires $\partial S / \partial R_I$ (from T2.6) and the energy-weighted
density $\epsilon$.

### 1.4 Grid Force

The grid contribution comes from the XC energy and the grid-based part of the
Hartree energy:

$$F_I^{\text{grid}} = -\frac{\partial}{\partial R_I}\int \left[\epsilon_{xc}(\mathbf{r}) + \frac{1}{2}v_H(\mathbf{r})\right] n(\mathbf{r}) \, d\mathbf{r}$$

In TIDES, this is computed via the adjoint of the $\rho$-build map (WP3 T3.6):
the grid-based potential $v(\mathbf{r}) \to H_{\mu\nu}$ adjoint gives
$\partial E / \partial R_I$ from the grid terms.

### 1.5 Dispersion Force

For D3/D4, the dispersion energy is a sum of $C_6 R^{-6}$ terms:

$$E_{\text{disp}} = -\sum_{I<J} \frac{C_6^{IJ}(R_I, R_J)}{R_{IJ}^6} f_{\text{damp}}(R_{IJ})$$

The force is the analytic gradient:

$$F_I^{\text{disp}} = -\frac{\partial E_{\text{disp}}}{\partial R_I}$$

which is computed by the `simple-dftd3` / `dftd4` libraries (WP7 T7.1).

### 1.6 Finite-Difference Validation

The 5-point central difference formula for the force:

$$F_I = -\frac{E(R_I + 2h) - 8E(R_I + h) + 8E(R_I - h) - E(R_I - 2h)}{12h} + O(h^4)$$

This is used as the validation oracle (GA1 gate): analytic forces must match
FD to $\leq 10^{-6}$ Ha/Bohr on the FP64 path.

---

## 2. XL-BOMD Shadow Dynamics

### 2.1 Extended Lagrangian

XL-BOMD (Niklasson, PRL 2008) introduces auxiliary electronic degrees of
freedom $n(t)$ that evolve harmonically around the ground state $n_0(R)$:

$$\mathcal{L} = \mathcal{L}_{\text{nuc}}(R, \dot{R}) + \frac{\mu}{2}\dot{n}^2 - \frac{\kappa}{2}(n - n_0(R))^2$$

where $\mu$ is a fictitious electronic mass and $\kappa$ is a spring constant.

### 2.2 Equations of Motion

The Euler-Lagrange equations give:

$$n(t+\Delta t) = 2n(t) - n(t-\Delta t) + \Delta t^2 \kappa (n_0(R) - n(t))$$
$$R(t+\Delta t) = 2R(t) - R(t-\Delta t) + \Delta t^2 F(R, n(t)) / M$$

### 2.3 KSA Kernel

The full Krylov-subspace-approximated (KSA) kernel replaces $\kappa$ with an
approximate inverse Jacobian $K \approx J^{-1}$, where
$J = \partial n_0 / \partial n$. This stabilizes the dynamics for small-gap
systems where the Jacobian is ill-conditioned.

In the simplified CPU reference, $K = I$ (identity), and $n_0(R)$ is computed
fresh each step (the "1 solve/step" design).

### 2.4 Energy Conservation (NVE)

For NVE dynamics, the shadow energy:

$$E_{\text{shadow}} = E_{\text{nuc}}(R, \dot{R}) + \frac{\kappa}{2}(n - n_0)^2$$

is conserved. The drift budget is $\leq 30\;\mu\text{Ha}/\text{atom}/\text{ps}$
(GB2 gate).

---

## 3. Density-Matrix Purification (SP2)

### 3.1 SP2 Iteration

The SP2 (second-order spectral projection) purification computes the density
matrix $P = \theta(\mu I - H)$ without diagonalization, via the recursion:

$$P_{k+1} = \begin{cases} 2P_k - P_k^2 & \text{if } \text{Tr}(P_k) < N_e/2 \\ P_k^2 & \text{otherwise} \end{cases}$$

starting from $P_0 = (c_{\min} I - H) / (c_{\min} - c_{\max})$ where
$c_{\min}, c_{\max}$ bound the spectrum of $H$.

### 3.2 Submatrix Method

For sparse $H$ (gapped systems), the submatrix method (NOLSM, Schäffer et al.)
solves the SP2 iteration as many small dense problems, one per atom
neighborhood. For each atom $I$, define the local Hamiltonian $H_I$ as the
submatrix of $H$ restricted to the basis functions within the truncation
radius of atom $I$. The local density matrix $P_I$ is computed by SP2 on
$H_I$, and the global $P$ is assembled from the diagonal blocks of all $P_I$.

This maps the sparse problem to batched dense GEMM — the tile substrate's
sweet spot.

---

## 4. Fermi-Operator Expansion (FOE)

### 4.1 Chebyshev Expansion of the Fermi Function

At finite electronic temperature $T_e$, the density matrix is:

$$P = f(H, \mu, T_e) = \frac{1}{1 + e^{\beta(H - \mu)}}$$

where $\beta = 1/(k_B T_e)$. This is approximated by a Chebyshev polynomial
expansion of order $n_p \sim \beta \cdot \Delta H$:

$$f(H) \approx \sum_{k=0}^{n_p} c_k T_k(\tilde{H})$$

where $\tilde{H} = (H - c_{\min}) / (c_{\max} - c_{\min}) \in [-1, 1]$ and
$T_k$ are Chebyshev polynomials.

### 4.2 Spectral Quadrature (SQ)

The SPARC spectral quadrature method computes $\text{Tr}(f(H))$ via Gauss
quadrature, avoiding explicit matrix functions:

$$\text{Tr}(f(H)) = \sum_{j} w_j \, e_j^T f(H) e_j$$

where $\{e_j, w_j\}$ are quadrature nodes. Each $e_j^T f(H) e_j$ is computed
via Chebyshev recurrence — a sequence of GEMMs on the tile substrate.

---

## 5. Mixed-Precision Ozaki Scheme

### 5.1 Error-Free Transformations

The Ozaki scheme decomposes an FP64 matrix $A$ into a sum of lower-precision
matrices:

$$A = A^{(0)} + A^{(1)} + \cdots + A^{(n_s)}$$

where $A^{(0)}$ is representable in FP16/BF16 and the residual terms
$A^{(k)}$ capture the rounding error. Each term is multiplied using
tensor-core GEMM (FP16 input, FP32 accumulate), and the results are summed
in FP32.

### 5.2 Accuracy Guarantee

For $n_s$ slices, the error in $C = A \times B$ is bounded by:

$$\|C_{\text{Ozaki}} - C_{\text{FP64}}\| \leq n_s \cdot \epsilon_{\text{FP16}} \cdot \|A\| \cdot \|B\|$$

For typical DFT Hamiltonians, $n_s = 2$–$3$ slices suffice for FP64-equivalent
accuracy ($\leq 10^{-13}$ relative on traces).

---

## 6. Energy Assembly

The Kohn-Sham total energy:

$$E_{\text{tot}} = E_{\text{kin}} + E_{\text{ne}} + E_H + E_{xc} + E_{\text{ion}}$$

where:
- $E_{\text{kin}} = \sum_k f_k \epsilon_k - \text{Tr}(P (V_{\text{ext}} + V_H + V_{xc}))$
- $E_{\text{ne}} = \text{Tr}(P V_{\text{ext}})$
- $E_H = \frac{1}{2}\text{Tr}(P V_H)$
- $E_{xc} = \text{Tr}(P \epsilon_{xc})$ (energy density, not potential)
- $E_{\text{ion}} = \frac{1}{2}\sum_{I \neq J} Z_I Z_J / |R_I - R_J|$ (Ewald for periodic)

All traces use FP64-emulated reductions (f64e) on the production path.

---

## 7. Stress Tensor

The stress tensor for periodic systems:

$$\sigma_{\alpha\beta} = -\frac{1}{V} \frac{\partial E}{\partial \epsilon_{\alpha\beta}}$$

where $\epsilon$ is the strain tensor and $V$ is the cell volume. Components:
- Kinetic: from the strain derivative of the basis functions
- Hartree + XC: from the grid deformation
- Ion-ion: from the Ewald strain derivative

Computed via finite differences in the CPU reference (T6.4), analytically in
the production path.
