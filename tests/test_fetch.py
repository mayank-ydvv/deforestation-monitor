"""Offline tests for the compositing helpers (no network)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from pipeline import config as cfg
from pipeline import fetch


def test_month_range_handles_month_lengths():
    assert fetch.month_range("2025-01") == "2025-01-01/2025-01-31"
    assert fetch.month_range("2024-02") == "2024-02-01/2024-02-29"
    assert fetch.month_range("2025-02") == "2025-02-01/2025-02-28"
    with pytest.raises(ValueError):
        fetch.month_range("2025-13")


def test_baseline_offsets():
    offsets = fetch.baseline_offsets(["03.01", "04.00", "05.11", None])
    np.testing.assert_array_equal(offsets, [0, 1000, 1000, 0])


def test_select_items_keeps_clearest_in_date_order():
    clouds = [15.0, 2.0, 9.0, 0.5, 19.0]
    items = [
        SimpleNamespace(datetime=datetime(2025, 1, d + 1), properties={"eo:cloud_cover": c})
        for d, c in enumerate(clouds)
    ]
    picked = fetch.select_items(items, max_n=3)
    assert [it.properties["eo:cloud_cover"] for it in picked] == [2.0, 9.0, 0.5]


def test_harmonise_applies_offset_and_nodata():
    dn = xr.DataArray(
        np.array([[1500.0, 0.0], [1500.0, 500.0]], dtype="float32")[:, None, :, None],
        dims=("time", "band", "y", "x"),
    )
    refl = fetch.harmonise(dn, np.array([0.0, 1000.0], dtype="float32")).values.squeeze()
    assert refl[0, 0] == pytest.approx(0.15)  # old baseline: DN / 10000
    assert np.isnan(refl[0, 1])  # DN 0 is no data
    assert refl[1, 0] == pytest.approx(0.05)  # new baseline: (DN - 1000) / 10000
    assert refl[1, 1] == 0.0  # below the offset clips to 0, not negative


def test_clear_mask_drops_cloud_shadow_and_nodata():
    scl = xr.DataArray(np.array([0, 1, 3, 4, 5, 6, 8, 9, 10, 11, np.nan]))
    expected = [False, False, False, True, True, True, False, False, False, True, False]
    np.testing.assert_array_equal(fetch.clear_mask(scl).values, expected)


def test_ndvi_range_and_zero_division():
    red = np.array([0.05, 0.3, 0.0])
    nir = np.array([0.45, 0.3, 0.0])
    out = fetch.ndvi(red, nir)
    assert out[0] == pytest.approx(0.8, abs=1e-5)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == 0.0  # eps keeps 0/0 finite


def test_clear_fraction_requires_all_bands():
    data = xr.DataArray(np.ones((2, 2, 2), dtype="float32"), dims=("band", "y", "x"))
    data[0, 0, 0] = np.nan  # one band missing -> pixel not clear
    assert fetch.clear_fraction(data) == pytest.approx(0.75)


def test_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, (len(cfg.BANDS), 64, 48)).astype("float32")
    values[:, :4, :4] = np.nan
    data = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={
            "band": list(cfg.BANDS),
            "y": 2_500_000 - 5 - 10 * np.arange(64),
            "x": 600_000 + 5 + 10 * np.arange(48),
        },
    )
    comp = fetch.Composite(data, fetch.clear_fraction(data), 3)
    path = tmp_path / "composite.tif"
    fetch.write_cache(comp, path, epsg=32644)
    back = fetch.read_cache(path)
    assert back.n_items == 3
    assert back.clear_fraction == pytest.approx(comp.clear_fraction, abs=1e-6)
    assert list(back.data.band.values) == list(cfg.BANDS)
    assert back.data.dtype == np.float32
    np.testing.assert_allclose(back.data.values, values, atol=0.5 / cfg.REFLECTANCE_SCALE)
    np.testing.assert_allclose(back.data.x.values, data.x.values)
    np.testing.assert_allclose(back.data.y.values, data.y.values)
    assert back.data.rio.crs.to_epsg() == 32644


def _fake_composite(*_args):
    data = xr.DataArray(
        np.ones((len(cfg.BANDS), 2, 2), dtype="float32"),
        dims=("band", "y", "x"),
        coords={"band": list(cfg.BANDS), "y": [15.0, 5.0], "x": [5.0, 15.0]},
    )
    return data


def test_composite_retries_after_read_error(monkeypatch):
    items = [SimpleNamespace(datetime=datetime(2025, 1, 1), properties={"eo:cloud_cover": 1})]
    searches, calls = [], []
    monkeypatch.setattr(fetch, "search_items", lambda *a: searches.append(a) or items)

    def flaky(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("Error reading Window(...): TIFFReadEncodedTile() failed")
        return _fake_composite()

    monkeypatch.setattr(fetch, "_median_composite", flaky)
    comp = fetch.composite((0, 0, 1, 1), "2025-01-01/2025-01-31", 32644)
    assert len(calls) == 2 and len(searches) == 2  # items re-signed for the retry
    assert comp.clear_fraction == 1.0 and comp.data.rio.crs.to_epsg() == 32644


def test_composite_gives_up_after_max_attempts(monkeypatch):
    items = [SimpleNamespace(datetime=datetime(2025, 1, 1), properties={"eo:cloud_cover": 1})]
    monkeypatch.setattr(fetch, "search_items", lambda *a: items)
    calls = []

    def broken(*args):
        calls.append(args)
        raise RuntimeError("stalled")

    monkeypatch.setattr(fetch, "_median_composite", broken)
    with pytest.raises(RuntimeError):
        fetch.composite((0, 0, 1, 1), "2025-01-01/2025-01-31", 32644)
    assert len(calls) == cfg.FETCH_ATTEMPTS
