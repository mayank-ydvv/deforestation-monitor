"""ESA WorldCover tree-cover labels on a composite's exact pixel grid.

Each WorldCover tile is a 3 x 3 degree COG; a ``WarpedVRT`` reads only the window (and
overview level) the AOI needs and reprojects it with nearest-neighbour straight onto the
composite's UTM grid, so cost is O(AOI pixels), not O(tile pixels).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import planetary_computer
import pystac
import pystac_client
import rasterio
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from pipeline import config as cfg

NO_OVERRIDE = 255  # override rasters use this value for "keep the WorldCover label"


class Grid(NamedTuple):
    """A raster pixel grid: CRS, affine transform and (height, width)."""

    crs: CRS
    transform: Affine
    shape: tuple[int, int]

    @property
    def bounds_latlon(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) of the grid."""
        h, w = self.shape
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (w, h)
        return transform_bounds(self.crs, "EPSG:4326", left, bottom, right, top)


def grid_of(data: xr.DataArray) -> Grid:
    """The pixel grid of a georeferenced (…, y, x) DataArray."""
    return Grid(data.rio.crs, data.rio.transform(), (data.sizes["y"], data.sizes["x"]))


def warp_to_grid(
    href: str | Path, grid: Grid, resampling: Resampling = Resampling.nearest
) -> np.ndarray:
    """Read band 1 of any raster reprojected onto ``grid`` (only the needed window)."""
    with (
        rasterio.open(href) as src,
        WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.shape[1],
            height=grid.shape[0],
            resampling=resampling,
        ) as vrt,
    ):
        return vrt.read(1)


def search_worldcover(
    bbox: Sequence[float], year: int = cfg.WORLDCOVER_LABEL_YEAR
) -> list[pystac.Item]:
    """Signed WorldCover items for ``year`` intersecting ``bbox`` (lon/lat)."""
    catalog = pystac_client.Client.open(cfg.STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(collections=[cfg.WORLDCOVER_COLLECTION], bbox=list(bbox))
    return [
        it
        for it in items.item_collection()
        if str(it.properties.get("start_datetime", "")).startswith(str(year))
    ]


def worldcover_classes(grid: Grid, year: int = cfg.WORLDCOVER_LABEL_YEAR) -> np.ndarray:
    """WorldCover class codes (uint8, 0 = no data) mosaicked onto ``grid``."""
    items = search_worldcover(grid.bounds_latlon, year)
    if not items:
        raise LookupError(f"No WorldCover {year} tiles cover {grid.bounds_latlon}")
    out = np.zeros(grid.shape, dtype="uint8")
    for item in items:  # 1-4 tiles; each fills only pixels still empty
        part = warp_to_grid(item.assets["map"].href, grid)
        np.copyto(out, part, where=out == 0)
    return out


def to_labels(classes: np.ndarray) -> np.ndarray:
    """Tree cover (class 10) -> 1, everything else -> 0, as uint8."""
    return (classes == cfg.WORLDCOVER_TREE_CLASS).astype("uint8")


def apply_overrides(labels: np.ndarray, override: np.ndarray) -> np.ndarray:
    """Replace labels with hand-corrected 0/1 values wherever ``override`` != 255."""
    return np.where(override == NO_OVERRIDE, labels, override).astype("uint8")


def override_path_for(aoi_id: str) -> Path:
    """labels/overrides/<aoi>.tif: optional hand-corrected labels (0/1, 255 = keep)."""
    return cfg.LABEL_OVERRIDES_DIR / f"{aoi_id}.tif"


def _write_labels(labels: np.ndarray, grid: Grid, path: Path) -> None:
    """Store labels as a uint8 COG on the composite grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(labels, 1)


def worldcover_labels(
    composite: xr.DataArray,
    cache_path: Path | None = None,
    override_path: Path | None = None,
) -> np.ndarray:
    """0/1 forest labels (uint8) aligned pixel-for-pixel with ``composite``.

    Uses ``cache_path`` if it exists (and matches the grid), otherwise fetches
    WorldCover and caches it. Hand-corrected overrides, if present, are applied last.
    """
    grid = grid_of(composite)
    labels = None
    if cache_path is not None and cache_path.exists():
        with rasterio.open(cache_path) as src:
            if src.transform == grid.transform and src.shape == grid.shape:
                labels = src.read(1)
    if labels is None:
        labels = to_labels(worldcover_classes(grid))
        if cache_path is not None:
            _write_labels(labels, grid, cache_path)
    if override_path is not None and override_path.exists():
        labels = apply_overrides(labels, warp_to_grid(override_path, grid))
    return labels
