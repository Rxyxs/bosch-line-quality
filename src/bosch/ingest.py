"""Download the Bosch competition files and convert each CSV to Parquet.

The CSVs are ~14 GB unzipped and mostly empty cells (each part visits a small
subset of ~50 stations). Parquet keeps the nulls nearly free and loads in
seconds, so the raw zips are deleted after conversion.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl

COMPETITION = "bosch-production-line-performance"
TABLES = ["numeric", "date", "categorical"]


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    subprocess.run([str(kaggle), "competitions", "download", COMPETITION, "-p", str(raw)], check=True)


def unpack_outer(raw: Path) -> None:
    outer = raw / f"{COMPETITION}.zip"
    if outer.exists():
        with zipfile.ZipFile(outer) as zf:
            zf.extractall(raw)
        outer.unlink()


def convert(raw: Path, out: Path, split: str, table: str) -> Path:
    target = out / f"{split}_{table}.parquet"
    if target.exists():
        return target
    src = raw / f"{split}_{table}.csv.zip"
    with zipfile.ZipFile(src) as zf, zf.open(zf.namelist()[0]) as fh:
        data = fh.read()
    # Type inference fails on columns that are empty for thousands of rows, so the
    # schema is declared: categorical cells hold codes like "T1", the rest are floats.
    header = data[: data.index(b"\n")].decode().strip().split(",")
    value = pl.Utf8 if table == "categorical" else pl.Float32
    schema = {c: pl.Int64 if c == "Id" else pl.Int8 if c == "Response" else value for c in header}
    df = pl.read_csv(data, schema=schema)
    del data
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target, compression="zstd")
    print(f"{target.name}: {df.height:,} rows x {df.width:,} cols", flush=True)
    return target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/parquet"))
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()
    if not args.skip_download and not any(args.raw.glob("*.csv.zip")):
        download(args.raw)
    unpack_outer(args.raw)
    for split in ("train", "test"):
        for table in TABLES:
            convert(args.raw, args.out, split, table)
    sub = args.raw / "sample_submission.csv.zip"
    if sub.exists():
        with zipfile.ZipFile(sub) as zf:
            zf.extractall(args.out)


if __name__ == "__main__":
    main()
