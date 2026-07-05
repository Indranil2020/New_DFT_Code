#!/usr/bin/env python3
"""Contract test for TileMat HDF5 stage dumps."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only on minimal runners.
    print("tilemat_hdf5_roundtrip_test: SKIP h5py is not available", file=sys.stderr)
    raise SystemExit(77)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def inspect_hdf5(path: Path, expect_symmetry: int) -> None:
    with h5py.File(path, "r") as h5:
        group = h5["TileMat"]
        assert group.attrs["schema"] == "tides.tilemat"
        assert int(group.attrs["version"]) == 1
        assert int(group.attrs["symmetry"]) == expect_symmetry
        tile_edge = int(group.attrs["tile_edge"])
        tile_count = int(group["col_ind"].shape[0])
        assert tile_edge in (16, 32, 64)
        assert group["row_ptr"].dtype.kind == "u"
        assert group["col_ind"].dtype.kind == "u"
        assert group["scales"].dtype.kind == "f"
        assert group["values"].dtype.kind == "f"
        assert group["values"].shape == (tile_count, tile_edge, tile_edge)
        assert group["row_ptr"].shape[0] == int(group.attrs["block_rows"]) + 1
        assert group["scales"].shape[0] == tile_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cases = (("general", 0), ("symmetric", 1))
    for mode, symmetry in cases:
        source = args.work_dir / f"{mode}.tilemat.bin"
        dump = args.work_dir / f"{mode}.tilemat.h5"
        restored = args.work_dir / f"{mode}.tilemat.restored.bin"
        run([str(args.fixture), mode, str(source)])
        run(
            [
                sys.executable,
                str(args.bridge),
                "roundtrip-check",
                str(source),
                str(dump),
                str(restored),
            ]
        )
        inspect_hdf5(dump, symmetry)
    print("tilemat_hdf5_roundtrip_test: HDF5 stage dump schema passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
