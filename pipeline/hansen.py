"""Hansen Global Forest Change loss year, for *validation only* (never used in training).

Tiles are 10 x 10 degree GeoTIFFs named by their top-left corner (e.g. ``30N_080E`` covers
20-30 N, 80-90 E) on Google Cloud Storage. Only the AOI window is read (HTTP range
requests) and warped onto the composite grid with nearest-neighbour (30 m -> 10 m).
Values: 0 = no loss, k = loss in year 2000 + k.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio

from pipeline import config as cfg
from pipeline.labels import Grid, warp_to_grid

HANSEN_VERSION = "GFC-2025-v1.13"  # loss years 2001-2025
HANSEN_BASE = "https://storage.googleapis.com/earthenginepartners-hansen"


def tile_name(top: int, left: int) -> str:
    """Hansen tile id for a top-left corner in whole degrees, e.g. (30, 80) -> '30N_080E'."""
    lat = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
    lon = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"
    return f"{lat}_{lon}"


def tile_names(bounds_lonlat: tuple[float, float, float, float]) -> list[str]:
    """Every Hansen tile intersecting (west, south, east, north)."""
    w, s, e, n = bounds_lonlat
    tops = range(math.ceil(s / 10) * 10, math.ceil(n / 10) * 10 + 1, 10)
    lefts = range(math.floor(w / 10) * 10, math.floor(e / 10) * 10 + 1, 10)
    # A south edge exactly on a multiple of 10 belongs to the tile below it only.
    tops = [t for t in tops if t - 10 < n and t > s] or [math.ceil(n / 10) * 10]
    return [tile_name(t, left) for t in tops for left in lefts]


def tile_url(name: str, layer: str = "lossyear", version: str = HANSEN_VERSION) -> str:
    """Public GCS URL of one Hansen layer tile."""
    return f"{HANSEN_BASE}/{version}/Hansen_{version}_{layer}_{name}.tif"


def loss_year(grid: Grid, cache_path: Path | None = None) -> np.ndarray:
    """Hansen loss year (uint8, 0 = none) on ``grid``; cached as a COG if ``cache_path``."""
    if cache_path is not None and cache_path.exists():
        with rasterio.open(cache_path) as src:
            if src.transform == grid.transform and src.shape == grid.shape:
                return src.read(1)
    out = np.zeros(grid.shape, dtype="uint8")
    for name in tile_names(grid.bounds_latlon):
        part = warp_to_grid(tile_url(name), grid)
        np.copyto(out, part, where=out == 0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "COG",
            "dtype": "uint8",
            "count": 1,
            "height": grid.shape[0],
            "width": grid.shape[1],
            "crs": grid.crs,
            "transform": grid.transform,
            "compress": "DEFLATE",
        }
        with rasterio.open(cache_path, "w", **profile) as dst:
            dst.write(out, 1)
    return out


def cache_path_for(aoi_id: str) -> Path:
    """cache/<aoi>/hansen_lossyear_<version>.tif"""
    return cfg.CACHE_DIR / aoi_id / f"hansen_lossyear_{HANSEN_VERSION}.tif"
