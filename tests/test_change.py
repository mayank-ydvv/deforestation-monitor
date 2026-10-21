"""Change detection on synthetic arrays."""

from __future__ import annotations

import time

import numpy as np
import pytest
from affine import Affine

from pipeline import change


def _scene(shape=(100, 100)):
    """Forest everywhere at t1, NDVI 0.8; t2 identical (nothing cleared yet)."""
    p1 = np.full(shape, 0.9, dtype="float32")
    p2 = p1.copy()
    n1 = np.full(shape, 0.8, dtype="float32")
    n2 = n1.copy()
    return p1, p2, n1, n2


def _clear(p2, n2, rows, cols):
    p2[rows, cols] = 0.1
    n2[rows, cols] = 0.3


def test_exact_patch_count():
    p1, p2, n1, n2 = _scene()
    for r0, c0 in [(5, 5), (5, 50), (60, 20)]:
        _clear(p2, n2, slice(r0, r0 + 10), slice(c0, c0 + 10))
    kept, lab, sizes = change.cleared(p1, p2, n1, n2)
    assert len(np.unique(lab[kept])) == 3
    assert kept.sum() == 300


def test_area_filter_boundary_49_vs_50():
    p1, p2, n1, n2 = _scene()
    _clear(p2, n2, slice(10, 17), slice(10, 17))  # 7 x 7 = 49 px -> dropped
    _clear(p2, n2, slice(40, 45), slice(40, 50))  # 5 x 10 = 50 px -> kept
    kept, lab, _ = change.cleared(p1, p2, n1, n2, min_px=50)
    assert not kept[10:17, 10:17].any()
    assert kept[40:45, 40:50].all() and kept.sum() == 50


def test_opening_removes_one_pixel_noise():
    p1, p2, n1, n2 = _scene()
    rng = np.random.default_rng(0)
    noise = rng.random(p1.shape) < 0.02
    noise[20:30, 20:30] = False
    p2[noise], n2[noise] = 0.1, 0.3  # isolated pixels
    p2[60, 5:95], n2[60, 5:95] = 0.1, 0.3  # 1-px wide line, 90 px long
    _clear(p2, n2, slice(20, 30), slice(20, 30))  # one real 100 px patch
    kept, _, _ = change.cleared(p1, p2, n1, n2, min_px=1)
    assert kept.sum() == 100 and kept[20:30, 20:30].all()


def test_both_signals_required_and_nan_never_cleared():
    p1, p2, n1, n2 = _scene()
    p2[0:10, 0:10] = 0.1  # model says cleared, NDVI unchanged -> no
    n2[20:30, 20:30] = 0.3  # NDVI drop, model still forest -> no
    _clear(p2, n2, slice(40, 50), slice(40, 50))
    p2[45, :] = np.nan  # no data across the real patch
    kept, _, _ = change.cleared(p1, p2, n1, n2, min_px=1)
    assert not kept[0:10, 0:10].any() and not kept[20:30, 20:30].any()
    assert not kept[45].any() and kept[40:45, 40:50].sum() > 0


def test_patch_stats_and_confirmation():
    p1, p2, n1, n2 = _scene()
    _clear(p2, n2, slice(0, 10), slice(0, 10))
    _clear(p2, n2, slice(50, 60), slice(50, 60))
    kept, lab, sizes = change.cleared(p1, p2, n1, n2)
    stats = change.patch_stats(lab, sizes, p1, p2, n1 - n2)
    assert sorted(s["area_ha"] for s in stats.values()) == [1.0, 1.0]  # 100 px x 0.01 ha
    assert all(s["ndvi_drop"] == pytest.approx(0.5) for s in stats.values())
    assert all(s["confidence"] == pytest.approx(0.9 * 0.9, abs=1e-3) for s in stats.values())
    p_next = p2.copy()
    p_next[50:60, 50:60] = 0.9  # second patch regrew / was a shadow
    confirmed = change.confirmed_ids(lab, sizes, p_next)
    assert confirmed == {int(lab[5, 5])}


def test_features_are_lonlat_with_contract_properties():
    p1, p2, n1, n2 = _scene()
    _clear(p2, n2, slice(10, 30), slice(10, 40))
    transform = Affine(10, 0, 700_000, 0, -10, 2_520_000)  # UTM 44N near Hasdeo
    feats = change.features(p1, p2, n1, n2, transform, 32644, "2025-01", "2026-01", "hasdeo")
    assert len(feats) == 1
    props = feats[0]["properties"]
    assert props == {
        "id": "hasdeo-2026-01-0001",
        "area_ha": 6.0,
        "t1": "2025-01",
        "t2": "2026-01",
        "ndvi_drop": 0.5,
        "confidence": pytest.approx(0.81, abs=1e-3),
        "confirmed": False,
    }
    ring = np.array(feats[0]["geometry"]["coordinates"][0])
    assert (ring[:, 0] > 82).all() and (ring[:, 0] < 84).all()
    assert (ring[:, 1] > 22).all() and (ring[:, 1] < 23).all()


def test_reference_month_modes():
    assert change.reference_month("2026-01", "yoy") == "2025-01"
    assert change.reference_month("2025-08", "consecutive", ["2025-05", "2025-07"]) == "2025-07"
    assert change.reference_month("2025-01", "consecutive") == "2024-12"


def test_runtime_5000_squared_under_3s():
    rng = np.random.default_rng(1)
    shape = (5000, 5000)
    p1 = rng.random(shape, dtype="float32")
    p2 = rng.random(shape, dtype="float32")
    n1 = rng.random(shape, dtype="float32")
    n2 = rng.random(shape, dtype="float32")
    t0 = time.perf_counter()
    change.cleared(p1, p2, n1, n2)
    assert time.perf_counter() - t0 < 3.0
