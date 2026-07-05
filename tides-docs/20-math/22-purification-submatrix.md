# Density-matrix purification via the submatrix method (regime R2, gapped)
## Idea
P = theta(mu*S - H) (spectral projector). SP2 (Niklasson) builds it by a fixed polynomial recursion of
X^2 terms. Submatrix method (Lass–Schade–Kuehne–Plessl lineage; the 100M-atom NOLSM demonstrations):
for each atom, extract the principal submatrix over its sparsity neighborhood, purify DENSE on the GPU
(batched), write back the block column => sparse global problem becomes many small dense problems —
exactly the tensor-core shape. Validity: gapped systems; radius set by density-matrix decay.
## Upgrades we add (beyond tight-binding-grade demos)
Error-compensated truncation (track dropped-norm ledger, correct trace/energy); f64e trace/Fermi
consistency; overlap handling via congruence with approximate inverse factor (Niklasson AINV-style),
computed once per geometry and reused across XL-BOMD steps.
## Kill criterion (gate GB1)
2,000-atom a-Si:H within 0.5 meV/atom of R1 dense on A40, or fall back to OMM/FOE for R2.
