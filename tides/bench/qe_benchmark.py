#!/usr/bin/env python3
"""QE comparison axis for the unified benchmark.

Runs Quantum Espresso (pw.x) on the same molecule ladder as unified_benchmark.py.
QE uses plane-wave basis + PBE pseudopotentials (CPU-only on this machine).

Comparison axis:
  - Same molecules, same geometries
  - QE: PBE + PAW/NC pseudopotentials, plane-wave basis
  - TIDES: LDA-PW92 + NAO-DZP + PseudoDojo ONCV PP
  - PySCF/gpu4pyscf: LDA-PW92 + def2-svp all-electron GTO

  Energy differences reflect basis + XC + PP differences (documented, not bugs).
  Timing comparison is apples-to-apples: wall time to SCF convergence.

Usage:
    python3 bench/qe_benchmark.py [--max-atoms N] [--repeats R]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BOHR_PER_ANG = 1.8897259886

# Reuse molecule definitions from unified_benchmark
sys.path.insert(0, str(Path(__file__).parent))
from unified_benchmark import MOLECULES, atoms_to_pyscf_string

# QE pseudopotential directory (BURAI PBE collection)
QE_PP_DIR = "/home/indranil/Downloads/BURAI1.3.1/pseudopot"
QE_BIN = "/home/indranil/miniconda3/envs/qe_env/bin/pw.x"

# Map atomic number to QE pseudopotential filename
Z_TO_PP = {
    1: "H.pbe-rrkjus_psl.1.0.0.UPF",
    6: "C.pbe-n-rrkjus_psl.1.0.0.UPF",
    7: "N.pbe-n-rrkjus_psl.1.0.0.UPF",
    8: "O.pbe-n-rrkjus_psl.1.0.0.UPF",
}


def generate_qe_input(mol_entry, ecut=50.0, conv_thr=1e-8, prefix="tides_bench"):
    """Generate QE pw.x input file for a molecule."""
    atoms = mol_entry["atoms"]
    n_atoms = len(atoms)
    
    # Convert positions to Angstrom (QE uses Angstrom by default with celldm)
    # Use a large vacuum box (molecule in vacuum)
    # Estimate box size from max coordinate + 10 Angstrom margin
    max_coord = 0.0
    for _, pos in atoms:
        for c in pos:
            max_coord = max(max_coord, abs(c))
    box_size = max(max_coord + 10.0, 20.0)  # At least 20 Angstrom

    # Build atomic species section
    species_lines = []
    for z, _ in atoms:
        sym_map = {1: "H", 6: "C", 7: "N", 8: "O"}
        sym = sym_map.get(z, f"Z{z}")
        pp_file = Z_TO_PP.get(z)
        if pp_file is None:
            return None, f"No PP found for Z={z}"
        species_lines.append(f"  {sym}  {z}  '{pp_file}'")

    # Build atomic positions
    pos_lines = []
    sym_map = {1: "H", 6: "C", 7: "N", 8: "O"}
    for z, pos in atoms:
        sym = sym_map.get(z, f"Z{z}")
        pos_lines.append(f"  {sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")

    input_text = f"""&CONTROL
  calculation = 'scf'
  prefix = '{prefix}'
  pseudo_dir = '{QE_PP_DIR}'
  outdir = './tmp_{prefix}'
  verbosity = 'low'
  etot_conv_thr = {conv_thr:.1e}
/

&SYSTEM
  ibrav = 0
  nat = {n_atoms}
  ntyp = {len(set(z for z, _ in atoms))}
  ecutwfc = {ecut}
  input_dft = 'PBE'
/

&ELECTRONS
  conv_thr = {conv_thr:.1e}
  mixing_beta = 0.3
  maxiter = 100
/

CELL_PARAMETERS angstrom
  {box_size:.6f}  0.0  0.0
  0.0  {box_size:.6f}  0.0
  0.0  0.0  {box_size:.6f}

ATOMIC_SPECIES
{chr(10).join(species_lines)}

ATOMIC_POSITIONS angstrom
{chr(10).join(pos_lines)}

K_POINTS automatic
  1 1 1  0 0 0
"""
    return input_text, None


def run_qe(mol_entry, ecut=50.0, repeats=3, work_dir="/tmp/qe_bench"):
    """Run QE pw.x SCF with timing."""
    os.makedirs(work_dir, exist_ok=True)
    
    results = []
    for rep in range(repeats):
        prefix = f"bench_{mol_entry['name']}_r{rep}"
        input_text, err = generate_qe_input(mol_entry, ecut=ecut, prefix=prefix)
        if err:
            results.append({"rep": rep, "error": err, "energy_ha": None, "wall_time_ms": None})
            continue

        input_file = os.path.join(work_dir, f"{prefix}.in")
        with open(input_file, "w") as f:
            f.write(input_text)

        t0 = time.perf_counter()
        proc = subprocess.run(
            [QE_BIN, "-in", input_file],
            capture_output=True, text=True, timeout=1800
        )
        t1 = time.perf_counter()

        # Parse energy from output
        energy_ha = None
        converged = False
        n_iterations = 0
        for line in proc.stdout.split("\n"):
            if "!" in line and "total energy" in line:
                # Line: !    total energy              =     -1.1234567890 Ry
                parts = line.split("=")
                if len(parts) >= 2:
                    ry_val = float(parts[-1].strip().split()[0])
                    energy_ha = ry_val / 2.0  # Ry to Ha
            if "convergence has been achieved" in line.lower():
                converged = True
            if "SCF correction" in line:
                n_iterations += 1
            if "iteration #" in line.lower():
                n_iterations += 1

        # Clean up temp files
        tmp_dir = os.path.join(work_dir, f"tmp_{prefix}")
        if os.path.exists(tmp_dir):
            subprocess.run(["rm", "-rf", tmp_dir], capture_output=True)

        entry = {
            "rep": rep,
            "energy_ha": energy_ha,
            "converged": converged,
            "n_iterations": n_iterations,
            "wall_time_ms": (t1 - t0) * 1000,
            "n_atoms": len(mol_entry["atoms"]),
            "ecut": ecut,
        }
        if proc.returncode != 0:
            entry["error"] = proc.stderr[:500] if proc.stderr else "Unknown QE error"
        results.append(entry)

    return results


def summarize_repeats(results):
    """Take min wall time, median energy across repeats."""
    valid = [r for r in results if r.get("energy_ha") is not None]
    if not valid:
        return {"energy_ha": None, "wall_time_ms": None, "converged": False, "n_iterations": 0, "n_repeats": len(results)}
    times = [r["wall_time_ms"] for r in valid]
    energies = [r["energy_ha"] for r in valid]
    iters = [r["n_iterations"] for r in valid]
    return {
        "energy_ha": float(__import__("numpy").median(energies)),
        "wall_time_ms": float(min(times)),
        "wall_time_mean_ms": float(sum(times) / len(times)),
        "converged": all(r["converged"] for r in valid),
        "n_iterations": int(__import__("numpy").median(iters)),
        "n_repeats": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="QE benchmark comparison")
    parser.add_argument("--max-atoms", type=int, default=8, help="Max atoms")
    parser.add_argument("--ecut", type=float, default=50.0, help="Plane-wave cutoff (Ry)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per system")
    parser.add_argument("--output", default="bench/profiling_results/qe_benchmark.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 100)
    print(f"  QE BENCHMARK — PBE, ecut={args.ecut} Ry, CPU-only")
    print(f"  QE binary: {QE_BIN}")
    print(f"  PP dir: {QE_PP_DIR}")
    print(f"  Repeats: {args.repeats}, Max atoms: {args.max_atoms}")
    print("=" * 100)

    all_results = []

    for mol in MOLECULES:
        n_atoms = len(mol["atoms"])
        if n_atoms > args.max_atoms:
            print(f"\n--- {mol['name']} ({n_atoms} atoms) — skipping ---")
            continue

        # Check if we have PPs for all atoms
        missing_pp = [z for z, _ in mol["atoms"] if z not in Z_TO_PP]
        if missing_pp:
            print(f"\n--- {mol['name']} ({n_atoms} atoms) — no PP for Z={missing_pp} ---")
            continue

        print(f"\n--- {mol['name']} ({n_atoms} atoms), ecut={args.ecut} Ry ---", flush=True)

        print(f"  QE pw.x...  ", end="", flush=True)
        qe_results = run_qe(mol, ecut=args.ecut, repeats=args.repeats)
        qe_summary = summarize_repeats(qe_results)

        if qe_summary["energy_ha"] is not None:
            print(f"E={qe_summary['energy_ha']:.10f} Ha, {qe_summary['wall_time_ms']:.1f} ms, "
                  f"{qe_summary['n_iterations']} iters, conv={qe_summary['converged']}", flush=True)
        else:
            err = qe_results[0].get("error", "unknown")
            print(f"FAILED: {err[:80]}", flush=True)

        entry = {
            "molecule": mol["name"],
            "n_atoms": n_atoms,
            "qe": qe_summary,
            "ecut_ry": args.ecut,
        }
        all_results.append(entry)

    # Print summary table
    print("\n" + "=" * 80)
    print("  QE BENCHMARK SUMMARY (PBE, plane-wave, CPU)")
    print("=" * 80)
    print(f"  {'Molecule':<14} {'N_atoms':>7} {'Ecut(Ry)':>10} {'QE(ms)':>12} {'QE E(Ha)':>18} {'Iters':>6} {'Conv':>6}")
    print("  " + "-" * 76)
    for entry in all_results:
        qe = entry.get("qe", {})
        e_str = f"{qe.get('energy_ha', 0):.10f}" if qe.get("energy_ha") else "—"
        t_str = f"{qe.get('wall_time_ms', 0):.1f}" if qe.get("wall_time_ms") else "—"
        print(f"  {entry['molecule']:<14} {entry['n_atoms']:>7} {entry['ecut_ry']:>10.1f} "
              f"{t_str:>12} {e_str:>18} {qe.get('n_iterations', 0):>6} {str(qe.get('converged', False)):>6}")
    print("=" * 80)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
