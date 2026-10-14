"""Feature tables for train and test parts.

Three groups, so their value can be measured separately:

- ``numeric``: the 968 anonymous measurements as they come.
- ``process``: what the line did to the part — stations visited, time spent
  per line, number of measurements, and how common the part's route is.
- ``sequence``: where the part sits in the production flow. Parts are sorted
  by start time (then Id), and each part gets the Id gap and start-time gap to
  its neighbours plus the size of the batch that started at the same time.
  These use timestamps and Ids only, never a neighbour's label, so they can be
  computed for unlabelled test parts and leak nothing in cross-validation.

Train and test parts come from the same production period and interleave in
time, so ``process`` route counts and ``sequence`` neighbours are computed on
the union of both — information a quality system on the line would have.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

PARQUET = Path("data/parquet")


def _stations(cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in cols:
        m = re.match(r"(L\d+_S\d+)_", c)
        if m:
            groups.setdefault(m.group(1), []).append(c)
    return groups


def load_dates() -> pl.DataFrame:
    tr = pl.read_parquet(PARQUET / "train_date.parquet")
    te = pl.read_parquet(PARQUET / "test_date.parquet")
    return pl.concat([tr.with_columns(pl.lit(True).alias("is_train")),
                      te.with_columns(pl.lit(False).alias("is_train"))])


def process_features(dates: pl.DataFrame, numeric_counts: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in dates.columns if c.startswith("L")]
    st = _stations(cols)
    lines = sorted({k.split("_")[0] for k in st})
    exprs = [pl.col("Id"), pl.col("is_train")]
    exprs += [pl.min_horizontal(cols).alias("start"), pl.max_horizontal(cols).alias("end")]
    exprs += [pl.any_horizontal([pl.col(c).is_not_null() for c in v]).cast(pl.Int8).alias(f"at_{k}")
              for k, v in st.items()]
    for ln in lines:
        lc = [c for c in cols if c.startswith(ln + "_")]
        exprs.append((pl.max_horizontal(lc) - pl.min_horizontal(lc)).alias(f"dur_{ln}"))
    df = dates.select(exprs)
    at = [c for c in df.columns if c.startswith("at_")]
    df = df.with_columns(
        (pl.col("end") - pl.col("start")).alias("duration"),
        pl.sum_horizontal(at).alias("n_stations"),
        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in at]).alias("_route"),
    )
    df = df.with_columns(pl.len().over("_route").alias("route_count")).drop("_route")
    return df.join(numeric_counts, on="Id", how="left")


def sequence_features(proc: pl.DataFrame) -> pl.DataFrame:
    s = proc.select("Id", "start").sort(["start", "Id"], nulls_last=True)
    s = s.with_columns(
        (pl.col("Id") - pl.col("Id").shift(1)).alias("id_gap_prev"),
        (pl.col("Id").shift(-1) - pl.col("Id")).alias("id_gap_next"),
        (pl.col("start") - pl.col("start").shift(1)).alias("t_gap_prev"),
        (pl.col("start").shift(-1) - pl.col("start")).alias("t_gap_next"),
        pl.len().over("start").alias("batch_size"),
        pl.col("Id").rank("ordinal").over("start").alias("batch_pos"),
    )
    s = s.with_columns(
        (pl.col("batch_pos") / pl.col("batch_size")).alias("batch_pos_frac"),
        # Parts started together but with non-consecutive Ids were interleaved
        # with other work; consecutive runs behave like a single lot.
        ((pl.col("id_gap_prev") == 1) & (pl.col("t_gap_prev") == 0)).cast(pl.Int8).alias("same_lot_prev"),
        ((pl.col("id_gap_next") == 1) & (pl.col("t_gap_next") == 0)).cast(pl.Int8).alias("same_lot_next"),
    )
    return s.drop("start")


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    counts = []
    for split in ("train", "test"):
        n = pl.read_parquet(PARQUET / f"{split}_numeric.parquet")
        meas = [c for c in n.columns if c.startswith("L")]
        counts.append(n.select("Id", pl.sum_horizontal([pl.col(c).is_not_null() for c in meas])
                               .alias("n_measurements")))
        if split == "train":
            n.select("Id", "Response").write_parquet(out / "labels.parquet")
    proc = process_features(load_dates(), pl.concat(counts))
    seq = sequence_features(proc)
    proc.join(seq, on="Id").write_parquet(out / "process_sequence.parquet")
    print(f"features: {proc.width - 2 + seq.width - 1} process+sequence columns for {proc.height:,} parts")


def load(split: str, groups: tuple[str, ...], feats: Path = Path("data/features")) -> pl.DataFrame:
    """Join the requested feature groups for one split, keyed and ordered by Id."""
    ps = pl.read_parquet(feats / "process_sequence.parquet").filter(pl.col("is_train") == (split == "train"))
    seq_cols = ["id_gap_prev", "id_gap_next", "t_gap_prev", "t_gap_next", "batch_size",
                "batch_pos", "batch_pos_frac", "same_lot_prev", "same_lot_next"]
    keep = ["Id"]
    if "process" in groups:
        keep += [c for c in ps.columns if c not in seq_cols + ["Id", "is_train"]]
    if "sequence" in groups:
        keep += seq_cols
    df = ps.select(keep)
    if "numeric" in groups:
        num = pl.read_parquet(PARQUET / f"{split}_numeric.parquet").drop("Response", strict=False)
        df = num.join(df, on="Id", how="left") if len(keep) > 1 else num
    return df.sort("Id")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/features"))
    build(p.parse_args().out)


if __name__ == "__main__":
    main()
