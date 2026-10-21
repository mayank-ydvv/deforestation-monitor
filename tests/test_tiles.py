"""Offline tests for tiling, NaN filtering, manifest and normalisation statistics."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import xarray as xr

from pipeline import config as cfg
from pipeline import tiles


def test_model_input_appends_ndvi():
    vals = np.full((6, 4, 4), 0.1, dtype="float32")
    vals[cfg.BANDS.index("B08")] = 0.5
    data = xr.DataArray(vals, dims=("band", "y", "x"), coords={"band": list(cfg.BANDS)})
    x = tiles.model_input(data)
    assert x.shape == (7, 4, 4) and x.dtype == np.float32
    assert x[-1, 0, 0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1), abs=1e-5)


def test_cut_tiles_positions_and_dropped_edges():
    arr = np.arange(2 * 10 * 13, dtype="float32").reshape(2, 10, 13)
    t = tiles.cut_tiles(arr, size=4)
    assert t.shape == (2, 3, 2, 4, 4)  # 10 // 4 rows, 13 // 4 cols, then (C, s, s)
    np.testing.assert_array_equal(t[1, 2], arr[:, 4:8, 8:12])
    labels = np.arange(10 * 13).reshape(10, 13)
    np.testing.assert_array_equal(tiles.cut_tiles(labels, 4)[0, 1], labels[0:4, 4:8])


def test_nan_filter_boundary_and_fill():
    size = 10
    x = np.ones((7, size, 3 * size), dtype="float32")
    x[:, 3:5, 4:7] = 2.0  # changes the channel mean
    x[0, 0, 0:10] = np.nan  # tile 0: 10 % NaN -> kept
    x[3, 0, 10:20] = np.nan
    x[3, 1, 10:11] = np.nan  # tile 1: 11 % NaN -> dropped
    y = np.zeros((size, 3 * size), dtype="uint8")
    y[:, 20:] = 1
    out = tiles.make_tiles(x, y, size=size, max_nan=0.10)
    assert out.n_total == 3
    np.testing.assert_array_equal(out.cols, [0, 2])
    assert out.x.dtype == np.float16 and out.y.dtype == np.uint8
    assert not np.isnan(out.x).any()
    mean0 = np.nanmean(x[0])
    assert out.x[0, 0, 0, 0] == pytest.approx(mean0, rel=1e-3)
    np.testing.assert_array_equal(out.y[1], 1)


def test_norm_stats_match_numpy(tmp_path):
    rng = np.random.default_rng(1)
    arrays = [rng.normal(i, i + 1, (7, 8, 8)).astype("float16") for i in range(3)]
    paths = []
    for i, a in enumerate(arrays):
        p = tmp_path / f"X_{i}.npy"
        np.save(p, a)
        paths.append(p)
    stats = tiles.norm_stats(paths)
    stacked = np.stack(arrays).astype("float64")
    np.testing.assert_allclose(stats["mean"], stacked.mean(axis=(0, 2, 3)), atol=1e-5)
    np.testing.assert_allclose(stats["std"], stacked.std(axis=(0, 2, 3)), atol=1e-4)
    assert stats["n_pixels"] == 3 * 64 and stats["channels"] == list(cfg.CHANNELS)


def test_forest_histogram_bins():
    f = np.array([0, 0, 0.1, 0.25, 0.3, 0.6, 0.9, 1, 1, 1])
    assert tiles.forest_histogram(f) == [2, 2, 1, 1, 1, 3]


def test_write_tiles_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "TILES_DIR", tmp_path)
    aoi = cfg.AOI(id="hasdeo", name="t", bbox=(0, 0, 1, 1), epsg=32644, split="test")
    x = np.zeros((2, 7, 4, 4), dtype="float16")
    y = np.stack([np.zeros((4, 4)), np.ones((4, 4))]).astype("uint8")
    ts = tiles.TileSet(x, y, np.array([0, 1]), np.array([3, 0]), 4)
    recs = tiles.write_tiles(ts, aoi, "2021-01")
    assert [r["forest_fraction"] for r in recs] == [0.0, 1.0]
    assert (tmp_path / recs[1]["path"]).name == "X_r001_c000.npy"
    loaded = np.load(tmp_path / recs[1]["y_path"], mmap_mode="r")
    assert loaded.dtype == np.uint8 and loaded.sum() == 16

    manifest = tiles.combine_manifest()
    assert len(manifest) == 2 and {r["split"] for r in manifest} == {"test"}
    with (tmp_path / "manifest.csv").open() as f:
        assert tuple(next(csv.reader(f))) == tiles.MANIFEST_FIELDS
