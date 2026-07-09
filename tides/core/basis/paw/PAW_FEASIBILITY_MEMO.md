# T7.6: PAW (Projector Augmented-Wave) Feasibility Memo

## Status: FEASIBLE — recommended for Phase B integration

## 1. Motivation

The TIDES engine currently uses norm-conserving pseudopotentials (NCPP,
PseudoDojo ONCV format). While NCPP works well for light-to-medium elements,
it has limitations for:

- **Heavy elements** (Z > 54): large core radii needed for smooth pseudowaves
  reduce transferability. PAW's partial-wave expansion handles this better.
- **Magnetic systems**: PAW preserves the full all-electron valence density,
  enabling accurate spin densities and hyperfine parameters.
- **Core-level spectroscopy**: PAW provides all-electron observables
  (XPS, XAS) without dedicated all-electron codes.
- **Pressure/strain**: PAW's harder/softer potential sets give better
  transferability across compression regimes.

## 2. PAW Method Overview

PAW (Blöchl, PRB 50, 17953, 1994) maps smooth pseudo-quantities to
all-electron quantities via a linear transformation:

```
|ψ⟩ = |ψ̃⟩ + Σ_i (|φ_i⟩ - |φ̃_i⟩) ⟨p_i | ψ̃⟩
```

where:
- `|ψ⟩` = all-electron wavefunction
- `|ψ̃⟩` = smooth pseudo-wavefunction (what the code works with)
- `|φ_i⟩` = all-electron partial wave (atomic, within augmentation region)
- `|φ̃_i⟩` = pseudo partial wave (smooth, matches φ_i outside r_c)
- `⟨p_i|` = projector function (localized within augmentation sphere)

The density is reconstructed as:
```
n(r) = ñ(r) + Σ_{ij} D_{ij} (φ_i* φ_j - φ̃_i* φ̃_j)
```

where `D_{ij} = Σ_n f_n ⟨ψ̃_n|p_i⟩⟨p_j|ψ̃_n⟩` is the occupancy matrix.

## 3. Integration with TIDES Architecture

### 3.1 Basis Compatibility
TIDES uses Numeric Atom-Centered Orbitals (NAO): `φ_nlm(r) = R_nl(|r-R_a|) Y_lm`.
PAW partial waves and projectors are also atom-centered radial functions ×
spherical harmonics — **structurally identical to NAO**.

**Implementation**: PAW partial waves `|φ̃_i⟩` can be represented as additional
NAO basis functions within the augmentation sphere. The projectors `⟨p_i|` are
auxiliary functions used only for inner products.

### 3.2 Tile Framework
The PAW augmentation is a **per-atom, local** operation:
- Each atom has an augmentation sphere of radius `r_c^a`.
- The compensation charge `n̂(r) = Σ_{ij} D_{ij} Q_{ij}(r)` is localized.
- In the tile framework, augmentation adds a **diagonal block correction**
  to the Hamiltonian and density — no cross-tile coupling.

**Tile impact**: PAW adds a per-atom correction of size `(n_proj × n_proj)` to
the diagonal tiles. This is a small dense operation per atom, not a global
sparse operation. The tile GEMM pipeline is unaffected.

### 3.3 SCF Driver
The SCF driver needs:
1. **PAW Hamiltonian**: `H = H̃ + Σ_a Σ_{ij} h_{ij}^a |p_i⟩⟨p_j|` where
   `h_{ij}^a` are the PAW kinetic + potential matrix elements.
2. **PAW density**: `n = ñ + Σ_a Σ_{ij} D_{ij}^a (φ_i*φ_j - φ̃_i*φ̃_j)`.
3. **PAW energy**: additional terms for core-valence, double-counting.

The existing `SCFDriver` with DIIS/Pulay mixing works unchanged — PAW only
modifies the Hamiltonian and density assembly, not the mixing strategy.

### 3.4 Forces
PAW forces have an additional on-site term:
```
F_a = F_a^pseudo + Σ_{ij} D_{ij}^a ∂h_{ij}^a/∂R_a
```
This is a per-atom derivative of the PAW matrix elements — a small dense
gradient (n_proj × n_proj × 3), computed analytically.

### 3.5 GPU Acceleration
PAW operations are:
- **Inner products** `⟨p_i|ψ̃_n⟩`: dense GEMV per atom, batched across atoms.
- **Augmentation**: dense GEMM per atom (n_proj × n_proj), batched.
- **Compensation charge**: radial function evaluation on the fine grid.

All are embarrassingly parallel across atoms and fit the batched GEMM
substrate naturally. Expected GPU speedup: same as SCF (50-100×).

## 4. Implementation Plan

### Phase 1: Data Structures (2 weeks)
- PAW dataset reader (PAWXML format, compatible with GPAW/ABINIT)
- `PAWAtom` struct: projectors, partial waves, augmentation operators
- Integration with `BasisSet` (PAW projectors as auxiliary NAO-like functions)

### Phase 2: Hamiltonian & Density (3 weeks)
- PAW on-site correction to H: `H_PAW = H̃ + Σ_a V^a_{ij} |p_i⟩⟨p_j|`
- PAW density reconstruction: `n = ñ + Δn^a` per atom
- Compensation charge on the fine grid (for Poisson)
- Energy assembly: PAW-specific terms (kinetic, core-valence, double-counting)

### Phase 3: Forces & Stress (2 weeks)
- Analytic PAW force contribution: `F^a = ∂h_{ij}^a/∂R_a × D_{ij}^a`
- PAW stress contribution
- FD5 validation against existing force test framework

### Phase 4: Validation (2 weeks)
- All-electron comparison for atoms (H, C, O, Si, Fe)
- Equilibrium lattice constants for Si, GaAs, Fe vs experiment
- Spin-polarized Fe atom: magnetic moment vs all-electron reference
- Band structure of Si vs existing NCPP path

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| PAW dataset availability | Low | PseudoDojo and JTH (ABINIT) PAW datasets cover all elements |
| Augmentation sphere overlap | Medium | Standard truncation at r_c; overlaps handled by partition-of-unity |
| Core density singularity | Medium | Use frozen-core approximation (standard in PAW) |
| Mixed-precision stability | Medium | PAW on-site terms are small dense operations; keep FP64 for D_{ij} |
| Performance regression | Low | PAW adds <5% to SCF time (small per-atom corrections) |

## 6. Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| PAW | All-electron accuracy, heavy elements | Implementation complexity | **Recommended** |
| Ultrasoft PP | Simpler than PAW | Less accurate, no all-electron | Not recommended |
| NCPP only | Already implemented | Limited transferability for heavy Z | Keep as default |
| All-electron (full) | No approximation | Too expensive for 10^6 atoms | Not for TIDES |

## 7. Conclusion

PAW is feasible within the TIDES architecture with moderate effort (~9 weeks).
The key insight is that PAW's per-atom augmentation is structurally compatible
with the NAO basis and tile framework — it adds small dense corrections to
diagonal tiles, not a global sparse operation. The GPU acceleration path is
natural (batched GEMM across atoms).

**Recommendation**: Implement PAW in Phase B as an optional basis path alongside
NCPP. Default remains NCPP for production; PAW for heavy elements and
all-electron observables.

## References

- Blöchl, PRB 50, 17953 (1994) — original PAW method
- Kresse & Joubert, PRB 59, 1758 (1999) — PAW for plane-wave codes
- GPAW documentation — PAW implementation in real-space grid code
- ABINIT PAW documentation — JTH PAW dataset format
- PseudoDojo PAW datasets — ONCV-compatible PAW for DFT codes
