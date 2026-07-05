# Forces and stress — one code path for all regimes
## Decisions
Analytic forces: Hellmann–Feynman + Pulay (NAOs move with atoms) + grid/XC terms + dispersion + Ewald,
expressed through the density matrix P (works identically for integer/fractional occupations and for
R0–R3). Stress by strain derivatives of every energy term. Autodiff is a verification oracle only
(JAX bridge), never the production path.
## Observables of understanding
Written derivation (theory manual chapter) of each force term with its FD isolation test; the 5-point
FD ladder definition in 50-verification/50 recited from memory by every WP6 contributor.
