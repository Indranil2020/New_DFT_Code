# Error control and certified accuracy
Every published number carries a budget: basis (DZP->TZP delta), grid (h-convergence), eigensolver /
purification residual, precision (mixed-vs-FP64 A/B), and truncation ledger (from spgemm_filtered and
submatrix radii). A-posteriori residual-based estimates (DFTK-inspired) adapted to NAO give a per-run
certified bound; requirement: reported bound >= measured error on the gauntlet in >=95% of cases.
The benchmark report generator refuses to emit a speed number without its accuracy line attached.
