"""Training tiles: composites + WorldCover labels -> 256x256 .npy pairs, manifest, stats.

Tiling is a single reshape (no Python loops over pixels). Tiles are split by *region*
(``config.AOI.split``), never randomly per tile, so val/test scores reflect unseen
geography. Outputs:

    tiles/<aoi>/<month>/X_rRRR_cCCC.npy   float16 (7, 256, 256)
    tiles/<aoi>/<month>/Y_rRRR_cCCC.npy   uint8   (256, 256)
    tiles/<aoi>/<month>/tiles.csv         per region-month manifest part
    tiles/<aoi>/<month>/summary.json      clear fraction, tile counts, AOI forest share
    tiles/manifest.csv                    all parts, split taken from config
    models/norm_stats.json                per-channel mean/std over the train split
    docs/class_balance.md                 class-balance report
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr

from pipeline import config as cfg
from pipeline.fetch import NoScenesError, monthly_composite, ndvi
from pipeline.labels import override_path_for, worldcover_labels

MANIFEST_FIELDS = ("region", "split", "month", "row", "col", "path", "y_path", "forest_fraction")
HIST_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


class TileSet(NamedTuple):
    """Kept tiles of one region-month and their positions in the tile grid."""

    x: np.ndarray  # (n, 7, size, size) float16
    y: np.ndarray  # (n, size, size) uint8
    rows: np.ndarray  # (n,) tile-grid row
    cols: np.ndarray  # (n,) tile-grid column
    n_total: int  # tiles before the NaN filter


# --- Pure array helpers ----------------------------------------------------------------
def model_input(data: xr.DataArray) -> np.ndarray:
    """(band, y, x) composite -> (7, H, W) float32: six bands + NDVI."""
    bands = data.sel(band=list(cfg.BANDS)).values.astype("float32")
    red, nir = bands[cfg.BANDS.index("B04")], bands[cfg.BANDS.index("B08")]
    return np.concatenate([bands, ndvi(red, nir)[None]], axis=0)


def cut_tiles(arr: np.ndarray, size: int = cfg.TILE_SIZE) -> np.ndarray:
    """(..., H, W) -> (rows, cols, ..., size, size) with stride ``size``; edges are dropped."""
    *lead, h, w = arr.shape
    nr, nc = h // size, w // size
    k = len(lead)
    tiled = arr[..., : nr * size, : nc * size].reshape(*lead, nr, size, nc, size)
    return tiled.transpose(k, k + 2, *range(k), k + 1, k + 3)


def nan_fraction(x_tiles: np.ndarray) -> np.ndarray:
    """(rows, cols, C, s, s) -> (rows, cols) share of pixels with NaN in any channel."""
    return np.isnan(x_tiles).any(axis=2).mean(axis=(-2, -1))


def fill_nan(x: np.ndarray, channel_means: np.ndarray) -> np.ndarray:
    """Replace NaN in (n, C, s, s) with the per-channel mean."""
    return np.where(np.isnan(x), channel_means[:, None, None], x)


def make_tiles(
    x_full: np.ndarray,
    y_full: np.ndarray,
    size: int = cfg.TILE_SIZE,
    max_nan: float = cfg.MAX_TILE_NAN,
) -> TileSet:
    """Cut (7, H, W) inputs and (H, W) labels; drop tiles with > ``max_nan`` NaN pixels."""
    xt, yt = cut_tiles(x_full, size), cut_tiles(y_full, size)
    keep = nan_fraction(xt) <= max_nan
    rows, cols = np.nonzero(keep)
    means = np.nanmean(x_full, axis=(1, 2))
    x = fill_nan(xt[keep], means).astype("float16")
    return TileSet(x, yt[keep], rows, cols, int(keep.size))


def norm_stats(x_paths: Iterable[Path]) -> dict:
    """Per-channel mean/std over tiles, streamed with memory-mapped loads (float64 sums)."""
    total = sq = None
    n = 0
    for p in x_paths:
        x = np.load(p, mmap_mode="r").astype("float64")
        s, q = x.sum(axis=(1, 2)), np.square(x).sum(axis=(1, 2))
        total, sq = (s, q) if total is None else (total + s, sq + q)
        n += x.shape[1] * x.shape[2]
    if not n:
        raise ValueError("No tiles to compute normalisation statistics from")
    mean = total / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 0.0))
    return {
        "channels": list(cfg.CHANNELS),
        "mean": mean.round(6).tolist(),
        "std": std.round(6).tolist(),
        "n_pixels": n,
    }


def forest_histogram(fractions: np.ndarray) -> list[int]:
    """Tile counts: exactly 0, (0, .25], (.25, .5], (.5, .75], (.75, 1), exactly 1."""
    f = np.asarray(fractions, dtype="float64")
    inner = f[(f > 0) & (f < 1)]
    bins = np.searchsorted(HIST_EDGES[1:-1], inner, side="left")  # right-closed bins
    mids = np.bincount(bins, minlength=len(HIST_EDGES) - 1).tolist()
    return [int((f == 0).sum()), *mids, int((f == 1).sum())]


# --- Region-month I/O ------------------------------------------------------------------
def tile_dir(aoi_id: str, month: str) -> Path:
    """tiles/<aoi>/<month>/"""
    return cfg.TILES_DIR / aoi_id / month


def write_tiles(tiles: TileSet, aoi: cfg.AOI, month: str) -> list[dict]:
    """Write a region-month's tiles (replacing old ones) and its manifest part."""
    out = tile_dir(aoi.id, month)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    fractions = tiles.y.mean(axis=(1, 2), dtype="float64")
    records = []
    for i, (r, c) in enumerate(zip(tiles.rows, tiles.cols, strict=True)):
        stem = f"r{r:03d}_c{c:03d}"
        np.save(out / f"X_{stem}.npy", tiles.x[i])
        np.save(out / f"Y_{stem}.npy", tiles.y[i])
        records.append(
            {
                "region": aoi.id,
                "split": aoi.split,
                "month": month,
                "row": int(r),
                "col": int(c),
                "path": str((out / f"X_{stem}.npy").relative_to(cfg.TILES_DIR)),
                "y_path": str((out / f"Y_{stem}.npy").relative_to(cfg.TILES_DIR)),
                "forest_fraction": round(float(fractions[i]), 4),
            }
        )
    write_csv(out / "tiles.csv", records)
    return records


def write_csv(path: Path, records: Sequence[dict]) -> None:
    """Write manifest-shaped records to CSV."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def build_region_month(aoi: cfg.AOI, month: str) -> dict | None:
    """Composite -> labels -> tiles for one region-month. Returns its summary, or None."""
    try:
        comp = monthly_composite(aoi, month)
    except NoScenesError as err:
        print(f"  {aoi.id} {month}: skipped ({err})")
        return None
    labels = worldcover_labels(
        comp.data,
        cache_path=cfg.CACHE_DIR / aoi.id / f"worldcover_{cfg.WORLDCOVER_LABEL_YEAR}.tif",
        override_path=override_path_for(aoi.id),
    )
    tiles = make_tiles(model_input(comp.data), labels)
    write_tiles(tiles, aoi, month)
    summary = {
        "region": aoi.id,
        "month": month,
        "scenes": comp.n_items,
        "clear_fraction": round(comp.clear_fraction, 4),
        "tiles_total": tiles.n_total,
        "tiles_kept": len(tiles.rows),
        "aoi_forest_fraction": round(float(labels.mean()), 4),
    }
    (tile_dir(aoi.id, month) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  {aoi.id} {month}: clear={comp.clear_fraction:.1%} "
        f"tiles {summary['tiles_kept']}/{summary['tiles_total']} "
        f"forest={summary['aoi_forest_fraction']:.1%}"
    )
    return summary


# --- Dataset-level outputs -------------------------------------------------------------
def combine_manifest() -> list[dict]:
    """Merge all region-month parts into tiles/manifest.csv, with splits from config."""
    records = []
    for part in sorted(cfg.TILES_DIR.glob("*/*/tiles.csv")):
        with part.open() as f:
            for rec in csv.DictReader(f):
                if rec["region"] not in cfg.AOIS:
                    continue
                rec["split"] = cfg.AOIS[rec["region"]].split
                records.append(rec)
    write_csv(cfg.TILES_DIR / "manifest.csv", records)
    return records


def load_summaries() -> list[dict]:
    """All region-month summary.json files for AOIs still in the config."""
    out = [json.loads(p.read_text()) for p in sorted(cfg.TILES_DIR.glob("*/*/summary.json"))]
    return [s for s in out if s["region"] in cfg.AOIS]


def balance_report(records: Sequence[dict], summaries: Sequence[dict]) -> str:
    """Markdown class-balance report from the manifest and region-month summaries."""
    frac = np.array([float(r["forest_fraction"]) for r in records])
    splits = np.array([r["split"] for r in records])
    lines = [
        "# Class balance: training tiles",
        "",
        f"Labels: ESA WorldCover {cfg.WORLDCOVER_LABEL_YEAR} (class {cfg.WORLDCOVER_TREE_CLASS} "
        f"= forest). Tiles: {cfg.TILE_SIZE}x{cfg.TILE_SIZE} px, stride {cfg.TILE_SIZE}, "
        f"dropped if > {cfg.MAX_TILE_NAN:.0%} NaN. Split is by region.",
        "",
        "## Per region-month",
        "",
        "| Region | Split | Month | Scenes | Clear | Tiles kept | Forest (tiles) | Forest (AOI) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        mask = np.array(
            [r["region"] == s["region"] and r["month"] == s["month"] for r in records], dtype=bool
        )
        tile_forest = f"{frac[mask].mean():.1%}" if mask.any() else "n/a"
        lines.append(
            f"| {s['region']} | {cfg.AOIS[s['region']].split} | {s['month']} | {s['scenes']} "
            f"| {s['clear_fraction']:.1%} | {s['tiles_kept']}/{s['tiles_total']} "
            f"| {tile_forest} | {s['aoi_forest_fraction']:.1%} |"
        )
    lines += [
        "",
        "## Per split",
        "",
        "| Split | Tiles | Forest pixels | All non-forest | (0, .25] | (.25, .5] "
        "| (.5, .75] | (.75, 1) | All forest |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        f = frac[splits == split]
        if not f.size:
            continue
        hist = " | ".join(str(v) for v in forest_histogram(f))
        lines.append(f"| {split} | {f.size} | {f.mean():.1%} | {hist} |")
    lines.append("")
    return "\n".join(lines)


def build_dataset_outputs() -> None:
    """Manifest, train-split norm stats and class-balance report from tiles on disk."""
    records = combine_manifest()
    train = [cfg.TILES_DIR / r["path"] for r in records if r["split"] == "train"]
    stats = norm_stats(train)
    stats["source"] = f"train split, {len(train)} tiles"
    cfg.NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.NORM_STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    report = balance_report(records, load_summaries())
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORTS_DIR / "class_balance.md").write_text(report)
    print(report)
    print(f"Wrote {cfg.TILES_DIR / 'manifest.csv'} ({len(records)} tiles), {cfg.NORM_STATS_PATH}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m pipeline.tiles [--aoi ID ...] [--report-only]``."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--aoi", action="append", choices=sorted(cfg.AOIS), help="default: all")
    ap.add_argument("--report-only", action="store_true", help="skip tiling, rebuild outputs")
    args = ap.parse_args(argv)
    if not args.report_only:
        for aoi_id in args.aoi or sorted(cfg.AOIS):
            aoi = cfg.get_aoi(aoi_id)
            for month in aoi.label_months:
                build_region_month(aoi, month)
    build_dataset_outputs()


if __name__ == "__main__":
    main()
