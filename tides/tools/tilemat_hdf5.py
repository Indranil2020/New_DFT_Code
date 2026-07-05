#!/usr/bin/env python3
"""TileMat binary/HDF5 stage-dump bridge.

The C++ TileMat serializer is the canonical byte-level payload. This tool
mirrors that schema into HDF5 for stage dumps and can reconstruct the exact
binary stream from the HDF5 datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
import sys

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only on minimal runners.
    h5py = None


MAGIC = b"TIDETM1\x00"
SCHEMA_NAME = "tides.tilemat"
GROUP_NAME = "TileMat"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TileMatPayload:
    rows: int
    cols: int
    tile_edge: int
    block_rows: int
    block_cols: int
    dtype: int
    scale_mode: int
    symmetry: int
    row_ptr: np.ndarray
    col_ind: np.ndarray
    scales: np.ndarray
    values: np.ndarray

    @property
    def tile_count(self) -> int:
        return int(self.col_ind.size)


def _read_exact(data: memoryview, offset: int, size: int) -> tuple[memoryview, int]:
    end = offset + size
    if end > len(data):
        raise ValueError("truncated TileMat binary payload")
    return data[offset:end], end


def _read_u32(data: memoryview, offset: int) -> tuple[int, int]:
    raw, offset = _read_exact(data, offset, 4)
    return struct.unpack("<I", raw)[0], offset


def _read_u64(data: memoryview, offset: int) -> tuple[int, int]:
    raw, offset = _read_exact(data, offset, 8)
    return struct.unpack("<Q", raw)[0], offset


def _read_array(
    data: memoryview, offset: int, dtype: np.dtype, count: int
) -> tuple[np.ndarray, int]:
    itemsize = np.dtype(dtype).itemsize
    raw, offset = _read_exact(data, offset, itemsize * count)
    return np.frombuffer(raw, dtype=dtype, count=count).copy(), offset


def load_binary(path: Path) -> TileMatPayload:
    data = memoryview(path.read_bytes())
    magic, offset = _read_exact(data, 0, len(MAGIC))
    if bytes(magic) != MAGIC:
        raise ValueError("bad TileMat binary magic")
    version, offset = _read_u32(data, offset)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported TileMat schema version {version}")
    rows, offset = _read_u64(data, offset)
    cols, offset = _read_u64(data, offset)
    tile_edge, offset = _read_u32(data, offset)
    block_rows, offset = _read_u64(data, offset)
    block_cols, offset = _read_u64(data, offset)
    dtype, offset = _read_u32(data, offset)
    scale_mode, offset = _read_u32(data, offset)
    symmetry, offset = _read_u32(data, offset)
    row_ptr_size, offset = _read_u64(data, offset)
    col_ind_size, offset = _read_u64(data, offset)
    scales_size, offset = _read_u64(data, offset)
    values_size, offset = _read_u64(data, offset)
    row_ptr, offset = _read_array(data, offset, np.dtype("<u8"), row_ptr_size)
    col_ind, offset = _read_array(data, offset, np.dtype("<u4"), col_ind_size)
    scales, offset = _read_array(data, offset, np.dtype("<f8"), scales_size)
    values, offset = _read_array(data, offset, np.dtype("<f8"), values_size)
    if offset != len(data):
        raise ValueError("TileMat binary payload has trailing bytes")
    payload = TileMatPayload(
        rows=rows,
        cols=cols,
        tile_edge=tile_edge,
        block_rows=block_rows,
        block_cols=block_cols,
        dtype=dtype,
        scale_mode=scale_mode,
        symmetry=symmetry,
        row_ptr=row_ptr,
        col_ind=col_ind,
        scales=scales,
        values=values,
    )
    validate_payload(payload)
    return payload


def save_binary(payload: TileMatPayload, path: Path) -> None:
    validate_payload(payload)
    header = struct.pack(
        "<8sIQQIQQIIIQQQQ",
        MAGIC,
        SCHEMA_VERSION,
        payload.rows,
        payload.cols,
        payload.tile_edge,
        payload.block_rows,
        payload.block_cols,
        payload.dtype,
        payload.scale_mode,
        payload.symmetry,
        int(payload.row_ptr.size),
        int(payload.col_ind.size),
        int(payload.scales.size),
        int(payload.values.size),
    )
    with path.open("wb") as out:
        out.write(header)
        out.write(np.asarray(payload.row_ptr, dtype="<u8").tobytes())
        out.write(np.asarray(payload.col_ind, dtype="<u4").tobytes())
        out.write(np.asarray(payload.scales, dtype="<f8").tobytes())
        out.write(np.asarray(payload.values, dtype="<f8").tobytes())


def validate_payload(payload: TileMatPayload) -> None:
    if payload.tile_edge not in (16, 32, 64):
        raise ValueError("tile_edge must be one of 16, 32, 64")
    expected_block_rows = 0 if payload.rows == 0 else (
        (payload.rows + payload.tile_edge - 1) // payload.tile_edge
    )
    expected_block_cols = 0 if payload.cols == 0 else (
        (payload.cols + payload.tile_edge - 1) // payload.tile_edge
    )
    if payload.block_rows != expected_block_rows:
        raise ValueError("block_rows does not match rows/tile_edge")
    if payload.block_cols != expected_block_cols:
        raise ValueError("block_cols does not match cols/tile_edge")
    if payload.dtype != 1:
        raise ValueError("only Float64 TileMat payloads are supported")
    if payload.scale_mode not in (0, 1):
        raise ValueError("unknown TileMat scale_mode")
    if payload.symmetry not in (0, 1):
        raise ValueError("unknown TileMat symmetry flag")
    if payload.symmetry == 1 and payload.rows != payload.cols:
        raise ValueError("symmetric TileMat payload must be square")
    if payload.row_ptr.size != payload.block_rows + 1:
        raise ValueError("row_ptr length does not match block_rows")
    if payload.row_ptr.size == 0 or payload.row_ptr[0] != 0:
        raise ValueError("row_ptr must begin at zero")
    if payload.row_ptr[-1] != payload.tile_count:
        raise ValueError("row_ptr terminal entry must equal tile_count")
    if np.any(payload.row_ptr[1:] < payload.row_ptr[:-1]):
        raise ValueError("row_ptr must be monotone")
    if payload.scales.size != payload.tile_count:
        raise ValueError("scales length does not match tile_count")
    tile_cells = payload.tile_edge * payload.tile_edge
    if payload.values.size != payload.tile_count * tile_cells:
        raise ValueError("values length does not match tile_count*tile_edge^2")
    for block_row in range(payload.block_rows):
        begin = int(payload.row_ptr[block_row])
        end = int(payload.row_ptr[block_row + 1])
        cols = payload.col_ind[begin:end]
        if np.any(cols >= payload.block_cols):
            raise ValueError("col_ind contains an out-of-range block column")
        if np.any(cols[1:] <= cols[:-1]):
            raise ValueError("col_ind must be strictly sorted within each row")
        if payload.symmetry == 1 and np.any(cols < block_row):
            raise ValueError("symmetric TileMat stores a lower-triangle tile")
    _validate_zero_padding(payload)


def _validate_zero_padding(payload: TileMatPayload) -> None:
    edge = payload.tile_edge
    for ordinal in range(payload.tile_count):
        block_row = int(np.searchsorted(payload.row_ptr, ordinal, side="right") - 1)
        block_col = int(payload.col_ind[ordinal])
        row0 = block_row * edge
        col0 = block_col * edge
        row_extent = min(edge, payload.rows - row0)
        col_extent = min(edge, payload.cols - col0)
        tile = payload.values[ordinal * edge * edge : (ordinal + 1) * edge * edge]
        tile = tile.reshape(edge, edge)
        if row_extent < edge and np.any(tile[row_extent:, :] != 0.0):
            raise ValueError("row padding contains nonzero values")
        if col_extent < edge and np.any(tile[:, col_extent:] != 0.0):
            raise ValueError("column padding contains nonzero values")


def write_hdf5(payload: TileMatPayload, path: Path) -> None:
    if h5py is None:
        raise RuntimeError("h5py is not available")
    validate_payload(payload)
    with h5py.File(path, "w") as h5:
        group = h5.create_group(GROUP_NAME)
        group.attrs["schema"] = SCHEMA_NAME
        group.attrs["version"] = SCHEMA_VERSION
        group.attrs["rows"] = payload.rows
        group.attrs["cols"] = payload.cols
        group.attrs["tile_edge"] = payload.tile_edge
        group.attrs["block_rows"] = payload.block_rows
        group.attrs["block_cols"] = payload.block_cols
        group.attrs["dtype"] = payload.dtype
        group.attrs["scale_mode"] = payload.scale_mode
        group.attrs["symmetry"] = payload.symmetry
        group.attrs["storage"] = "csr_of_zero_padded_tiles"
        group.create_dataset("row_ptr", data=payload.row_ptr.astype("<u8"))
        group.create_dataset("col_ind", data=payload.col_ind.astype("<u4"))
        group.create_dataset("scales", data=payload.scales.astype("<f8"))
        group.create_dataset(
            "values",
            data=payload.values.reshape(
                payload.tile_count, payload.tile_edge, payload.tile_edge
            ).astype("<f8"),
        )


def read_hdf5(path: Path) -> TileMatPayload:
    if h5py is None:
        raise RuntimeError("h5py is not available")
    with h5py.File(path, "r") as h5:
        if GROUP_NAME not in h5:
            raise ValueError("HDF5 file has no TileMat group")
        group = h5[GROUP_NAME]
        if group.attrs.get("schema") != SCHEMA_NAME:
            raise ValueError("HDF5 TileMat group has wrong schema")
        if int(group.attrs.get("version", -1)) != SCHEMA_VERSION:
            raise ValueError("HDF5 TileMat group has wrong version")
        payload = TileMatPayload(
            rows=int(group.attrs["rows"]),
            cols=int(group.attrs["cols"]),
            tile_edge=int(group.attrs["tile_edge"]),
            block_rows=int(group.attrs["block_rows"]),
            block_cols=int(group.attrs["block_cols"]),
            dtype=int(group.attrs["dtype"]),
            scale_mode=int(group.attrs["scale_mode"]),
            symmetry=int(group.attrs["symmetry"]),
            row_ptr=np.asarray(group["row_ptr"], dtype="<u8"),
            col_ind=np.asarray(group["col_ind"], dtype="<u4"),
            scales=np.asarray(group["scales"], dtype="<f8"),
            values=np.asarray(group["values"], dtype="<f8").reshape(-1),
        )
    validate_payload(payload)
    return payload


def run_roundtrip_check(
    input_binary: Path, hdf5_path: Path, output_binary: Path
) -> None:
    source = load_binary(input_binary)
    write_hdf5(source, hdf5_path)
    restored = read_hdf5(hdf5_path)
    save_binary(restored, output_binary)
    if input_binary.read_bytes() != output_binary.read_bytes():
        raise ValueError("binary -> HDF5 -> binary changed payload bits")


def _require_h5py_or_skip() -> None:
    if h5py is None:
        print("tilemat_hdf5.py: SKIP h5py is not available", file=sys.stderr)
        raise SystemExit(77)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_hdf5 = subparsers.add_parser("binary-to-hdf5")
    to_hdf5.add_argument("input_binary", type=Path)
    to_hdf5.add_argument("output_hdf5", type=Path)

    to_binary = subparsers.add_parser("hdf5-to-binary")
    to_binary.add_argument("input_hdf5", type=Path)
    to_binary.add_argument("output_binary", type=Path)

    roundtrip = subparsers.add_parser("roundtrip-check")
    roundtrip.add_argument("input_binary", type=Path)
    roundtrip.add_argument("output_hdf5", type=Path)
    roundtrip.add_argument("output_binary", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "binary-to-hdf5":
            _require_h5py_or_skip()
            write_hdf5(load_binary(args.input_binary), args.output_hdf5)
        elif args.command == "hdf5-to-binary":
            _require_h5py_or_skip()
            save_binary(read_hdf5(args.input_hdf5), args.output_binary)
        elif args.command == "roundtrip-check":
            _require_h5py_or_skip()
            run_roundtrip_check(
                args.input_binary, args.output_hdf5, args.output_binary
            )
        else:
            raise AssertionError(args.command)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI reports validation failures.
        print(f"tilemat_hdf5.py: ERROR {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
