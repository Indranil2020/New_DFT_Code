#!/usr/bin/env python3
"""p2-vext-cpu: Correctness + performance verification for merged-GEMM V_ext.

Runs CH4 PP and H2O PP with TIDES_DUMP_HMAT_DIR set, captures V_ext.txt,
and compares elementwise against a reference dump. Also captures the
'[NaoDriver] V_ext assembly' log timing (second run = warm caches).

Usage:
  LD_PRELOAD=<mkl preload> python3 tides/bench/p2_vext_verify.py \
      --mode <before|after|compare> [--ref-dir <dir>] [--out-dir <dir>]

Modes:
  before  — run with current code, save V_ext.txt to --out-dir
  after   — run with current code, save V_ext.txt to --out-dir
  compare — compare V_ext.txt in --ref-dir vs --out-dir
"""
import sys, os, argparse, time, re, subprocess, math

BOHR = 1.8897261254535

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD_DIR = os.path.join(WT, 'build')
TIDES_SRC = os.path.join(WT, 'tides')
PP_DIR = os.path.join(TIDES_SRC, 'external/pseudopotentials/pseudodojo-pbe-sr')


def make_molecule(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    Z, pos = [], []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        Z.append(atom.GetAtomicNum())
        pos.extend([p.x * BOHR, p.y * BOHR, p.z * BOHR])
    return Z, pos


def run_and_capture(label, Z, pos, dump_dir, max_iter=1):
    """Run NaoDriver.run twice in the same process (warm caches on 2nd run).
    Returns V_ext.txt path and the warm V_ext assembly timing (seconds).

    Captures C++ stdout via a pipe to a temp file (std::cout is not
    redirectable from Python)."""
    os.environ['TIDES_DUMP_HMAT_DIR'] = dump_dir
    os.environ['TIDES_DISABLE_GPU'] = '1'  # CPU-only path

    # Save real stdout, redirect to a pipe
    import tempfile
    old_stdout_fd = os.dup(1)

    # First run (cold caches)
    import tides._native as native
    r1 = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos,
        grid_h=0.3, grid_margin=6.0,
        max_iter=max_iter, tol=1e-7,
        use_dual_grid=False, xc_functional='pbe',
        pp_dir=PP_DIR,
        allow_grid_refine=False,
    )

    # Second run (warm caches) — capture stdout via temp file
    tmpf = tempfile.TemporaryFile(mode='w+')
    os.dup2(tmpf.fileno(), 1)
    r2 = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos,
        grid_h=0.3, grid_margin=6.0,
        max_iter=max_iter, tol=1e-7,
        use_dual_grid=False, xc_functional='pbe',
        pp_dir=PP_DIR,
        allow_grid_refine=False,
    )
    # Flush and restore
    sys.stdout.flush()
    os.dup2(old_stdout_fd, 1)
    os.close(old_stdout_fd)

    tmpf.seek(0)
    warm_log = tmpf.read()
    tmpf.close()

    # Parse V_ext assembly timing from warm log
    vext_time_s = None
    for line in warm_log.split('\n'):
        if 'V_ext assembly' in line and '[NaoDriver]' in line:
            m = re.search(r'step=([\d.]+)\s*ms', line)
            if m:
                vext_time_s = float(m.group(1)) / 1000.0
            break

    vext_path = os.path.join(dump_dir, 'V_ext.txt')
    return vext_path, vext_time_s


def compare_vext(ref_path, out_path, tol=1e-11):
    """Compare two V_ext.txt files elementwise."""
    with open(ref_path) as f:
        ref_vals = [float(x) for x in f.read().split()]
    with open(out_path) as f:
        out_vals = [float(x) for x in f.read().split()]

    assert len(ref_vals) == len(out_vals), \
        f"Length mismatch: ref={len(ref_vals)} vs out={len(out_vals)}"

    max_diff = 0.0
    max_idx = -1
    for i, (a, b) in enumerate(zip(ref_vals, out_vals)):
        d = abs(a - b)
        if d > max_diff:
            max_diff = d
            max_idx = i

    print(f"  V_ext comparison: {len(ref_vals)} elements")
    print(f"  Max |diff| = {max_diff:.6e} at index {max_idx}")
    print(f"  Tolerance  = {tol:.6e}")
    if max_diff <= tol:
        print(f"  PASS")
        return True, max_diff
    else:
        print(f"  FAIL")
        return False, max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['before', 'after', 'compare'])
    ap.add_argument('--ref-dir', default='/tmp/p2_vext_before')
    ap.add_argument('--out-dir', default='/tmp/p2_vext_after')
    ap.add_argument('--molecules', default='CH4,H2O')
    args = ap.parse_args()

    if args.mode == 'compare':
        all_pass = True
        for label in args.molecules.split(','):
            print(f"\n=== {label} PP V_ext comparison ===")
            ref_path = os.path.join(args.ref_dir, f'{label}_PP', 'V_ext.txt')
            out_path = os.path.join(args.out_dir, f'{label}_PP', 'V_ext.txt')
            ok, max_diff = compare_vext(ref_path, out_path)
            all_pass &= ok
        print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
        sys.exit(0 if all_pass else 1)

    molecules = {
        'CH4': 'C',
        'H2O': 'O',
    }

    for label in args.molecules.split(','):
        smiles = molecules.get(label, label)
        Z, pos = make_molecule(smiles)
        dump_dir = os.path.join(args.out_dir, f'{label}_PP')
        os.makedirs(dump_dir, exist_ok=True)

        print(f"\n=== {label} PP (mode={args.mode}) ===", flush=True)
        vext_path, vext_time = run_and_capture(label, Z, pos, dump_dir)
        print(f"  V_ext dumped to: {vext_path}", flush=True)
        if vext_time is not None:
            print(f"  Warm V_ext assembly time: {vext_time:.4f} s", flush=True)
        else:
            print(f"  WARNING: Could not parse V_ext assembly timing", flush=True)


if __name__ == '__main__':
    main()
