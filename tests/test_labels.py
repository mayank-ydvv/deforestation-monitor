"""Offline tests for WorldCover label alignment (local rasters, no network)."""

from __future__ import annotations

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

from pipeline import labels


def _utm_composite(h: int = 40, w: int = 50) -> xr.DataArray:
    """A tiny (band, y, x) composite on a 10 m UTM 44N grid."""
    data = xr.DataArray(
        np.zeros((1, h, w), dtype="float32"),
        dims=("band", "y", "x"),
        coords={"y": 2_520_000 - 5 - 10 * np.arange(h), "x": 700_000 + 5 + 10 * np.arange(w)},
    )
    return data.rio.write_crs(32644)


def _write(path, arr, crs, transform, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="uint8",
        count=1,
        height=arr.shape[0],
        width=arr.shape[1],
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def test_to_labels_only_tree_class():
    classes = np.array([[10, 20, 30], [40, 50, 10], [0, 80, 95]], dtype="uint8")
    np.testing.assert_array_equal(labels.to_labels(classes), [[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert labels.to_labels(classes).dtype == np.uint8


def test_apply_overrides_keeps_255():
    lab = np.array([1, 1, 0, 0], dtype="uint8")
    override = np.array([255, 0, 1, 255], dtype="uint8")
    np.testing.assert_array_equal(labels.apply_overrides(lab, override), [1, 0, 1, 0])


def test_grid_of_matches_coords():
    grid = labels.grid_of(_utm_composite())
    assert grid.crs.to_epsg() == 32644
    assert grid.transform == Affine(10, 0, 700_000, 0, -10, 2_520_000)
    assert grid.shape == (40, 50)
    w, s, e, n = grid.bounds_latlon
    assert 82 < w < e < 84 and 22 < s < n < 23


def test_warp_to_grid_nearest_from_latlon(tmp_path):
    """A lon/lat raster split into a west half (10) and east half (30) lands on the UTM grid."""
    grid = labels.grid_of(_utm_composite())
    w, s, e, n = grid.bounds_latlon
    pad, res = 0.002, 0.0001
    width, height = int((e - w + 2 * pad) / res), int((n - s + 2 * pad) / res)
    src = np.full((height, width), 30, dtype="uint8")
    src[:, : width // 2] = 10
    tif = tmp_path / "wc.tif"
    _write(tif, src, CRS.from_epsg(4326), Affine(res, 0, w - pad, 0, -res, n + pad), nodata=0)
    out = labels.warp_to_grid(tif, grid)
    assert out.shape == grid.shape and out.dtype == np.uint8
    assert set(np.unique(out)) == {10, 30}  # nearest-neighbour: no blended classes
    assert (out[:, 0] == 10).all() and (out[:, -1] == 30).all()


def test_worldcover_labels_uses_cache_and_overrides(tmp_path):
    comp = _utm_composite()
    grid = labels.grid_of(comp)
    cached = np.zeros(grid.shape, dtype="uint8")
    cached[:, :25] = 1
    cache = tmp_path / "worldcover.tif"
    _write(cache, cached, grid.crs, grid.transform)
    override = np.full(grid.shape, 255, dtype="uint8")
    override[:10, :] = 0  # hand-correct the top rows to non-forest
    ov = tmp_path / "override.tif"
    _write(ov, override, grid.crs, grid.transform)

    out = labels.worldcover_labels(comp, cache_path=cache, override_path=ov)  # no network
    assert out[:10].sum() == 0
    np.testing.assert_array_equal(out[10:], cached[10:])
