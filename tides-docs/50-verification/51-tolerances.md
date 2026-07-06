# Tolerances (mirror of verification/tolerances.yaml — single source of truth)
Rung1 kernel: |err| <= budget per kernel (default 8 ulp FP32-equivalent; f64e: <=4x FP64 ulp bound).
Rung2 operator: overlap/kinetic vs PySCF matched basis <=1e-8 Ha; rotation invariance <=1e-12;
  Poisson analytic <=1e-10 Ha; adjointness <=1e-12 (FP64 path).
Rung3 energy: component match vs PySCF/ABACUS <=1e-6 Ha/atom; A/B mixed-vs-FP64 <=0.5 meV/atom.
Rung4 force: FD <=1e-6 Ha/Bohr (FP64 path) / <=1e-4 (production mixed); stress FD <=1e-6 Ha.
Rung5 dynamics: NVE drift <=30 uHa/atom/ps; XL-BOMD RDF vs SCF-MD KS-test p>0.05.
Rung6 physics: ACWF subset <= few meV/atom vs published all-electron refs (basis-labelled);
  S22 (TZP+D4, counterpoise) MAD <=0.35 kcal/mol; charged UKS charge/spin preserved exactly and
  energy ordering stable under DZP->TZP->TZP+diffuse refinement, with density-tail and spin-density
  diagnostics archived for every charged/open-shell case.
Changing any number here requires S9 sign-off and a written justification in this file's history.
