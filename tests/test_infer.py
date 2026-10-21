"""Sliding-window inference with fake ONNX sessions (no model file needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from pipeline import config as cfg
from pipeline import infer


@pytest.fixture
def norm_stats(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"mean": [0.0] * 7, "std": [1.0] * 7}))
    return path


def _composite(values: np.ndarray) -> xr.DataArray:
    """(6, H, W) reflectance composite with band names and a 10 m grid."""
    _, h, w = values.shape
    return xr.DataArray(
        values.astype("float32"),
        dims=("band", "y", "x"),
        coords={
            "band": list(cfg.BANDS),
            "y": 2_500_000 - 5 - 10 * np.arange(h),
            "x": 600_000 + 5 + 10 * np.arange(w),
        },
    )


class _ConstantSession:
    def __init__(self, value: float = 0.7):
        self.value, self.calls = value, 0

    def run(self, names, feeds):
        self.calls += 1
        n, _, h, w = feeds["x"].shape
        return [np.full((n, 1, h, w), self.value, dtype="float32")]


class _WindowMeanSession:
    """Each window predicts its own mean blue value: overlapping windows disagree."""

    def run(self, names, feeds):
        x = feeds["x"]
        means = 1 / (1 + np.exp(-x[:, 0].mean(axis=(1, 2))))
        return [np.broadcast_to(means[:, None, None, None], (x.shape[0], 1, *x.shape[2:])).copy()]


def test_window_starts_cover_axis():
    assert infer.window_starts(256) == [0]
    assert infer.window_starts(100) == [0]
    starts = infer.window_starts(1000, 256, 32)
    assert starts[0] == 0 and starts[-1] == 1000 - 256
    assert all(b - a <= 224 for a, b in zip(starts, starts[1:], strict=False))


def test_blend_weights_ramp_only_in_overlap():
    w = infer.blend_weights(256, 32)
    assert (w > 0).all()
    assert np.all(w[32:-32, 32:-32] == 1.0)
    assert w[0, 128] < w[16, 128] < w[31, 128] <= 1.0


def test_output_shape_matches_input_and_constant_has_no_seams(norm_stats):
    comp = _composite(np.full((6, 700, 530), 0.1))
    out = infer.predict(comp, _ConstantSession(0.7), norm_stats=norm_stats, chunk=300)
    assert out.shape == (700, 530) and out.dtype == np.float16
    assert np.all(out == np.float16(0.7))


def test_chunked_equals_single_pass(norm_stats):
    rng = np.random.default_rng(0)
    comp = _composite(rng.random((6, 600, 650)))
    session = _WindowMeanSession()
    full = infer.predict(comp, session, norm_stats=norm_stats, chunk=4096)
    chunked = infer.predict(comp, session, norm_stats=norm_stats, chunk=200)
    np.testing.assert_array_equal(full, chunked)


def test_nan_stays_nan_and_empty_windows_are_skipped(norm_stats, tmp_path):
    values = np.full((6, 512, 512), 0.1)
    values[:, :, 250:] = np.nan  # right part has no data: windows starting at 256 are empty
    values[2, 10, 10] = np.nan  # one missing pixel in one band
    session = _ConstantSession(0.4)
    out = infer.predict(
        _composite(values), session, norm_stats=norm_stats, out_path=tmp_path / "p.npy", batch=1
    )
    assert np.isnan(out[:, 250:]).all() and np.isnan(out[10, 10])
    assert np.all(out[:, :250][~np.isnan(out[:, :250])] == np.float16(0.4))
    assert session.calls == 3 * 2  # windows start at 0, 224, 256: 3 rows x 2 cols with data
    assert np.load(tmp_path / "p.npy", mmap_mode="r").shape == (512, 512)


def test_small_image_is_padded(norm_stats):
    out = infer.predict(_composite(np.full((6, 100, 80), 0.1)), _ConstantSession(), norm_stats)
    assert out.shape == (100, 80) and np.all(out == np.float16(0.7))
