# Fermi-operator expansion / Spectral Quadrature (regime R3, metallic, finite Te)
## Idea
P = f_FD((H - mu S)/kTe) approximated by Chebyshev expansion (FOE) or Gauss quadrature of the resolvent
(SQ, Suryanarayana lineage — the method behind million-atom SPARC results). Polynomial order
p ~ beta * (spectral width); spectral bounds via a few Lanczos steps. All work = repeated
spgemm_filtered on the same tiles => same substrate, no diagonalization.
## Decisions
FOE default (simplest on tiles); SQ variant evaluated for high-Te; mu found by f64e-safe bracketed
Newton on tr(P S) = N_e. Chemical-potential derivative available for XL-BOMD response terms.
## Observables of understanding
Order-vs-Te curve on 500-atom Al matching theory; free-energy match <=1 meV/atom vs dense control.
