"""Offline tests for the training Dataset (NumPy only; torch not required)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pipeline import tiles
from training import dataset as ds


@pytest.fixture
def tile_root(tmp_path):
    """Three 7x8x8 tiles (train, train, val) with a manifest and norm stats."""
    rng = np.random.default_rng(0)
    rows = []
    for i, split in enumerate(["train", "train", "val"]):
        y = (rng.random((8, 8)) > 0.5).astype("uint8")
        x = rng.normal(0.2, 0.05, (7, 8, 8)).astype("float16")
        x[0] = y  # channel 0 mirrors the label so geometry can be checked after augment
        np.save(tmp_path / f"X_{i}.npy", x)
        np.save(tmp_path / f"Y_{i}.npy", y)
        rows.append(
            {
                "region": f"r{i}",
                "split": split,
                "month": "2021-01",
                "row": 0,
                "col": i,
                "path": f"X_{i}.npy",
                "y_path": f"Y_{i}.npy",
                "forest_fraction": float(y.mean()),
            }
        )
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tiles.MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    stats = {"mean": [0.0] * 7, "std": [1.0] * 7}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    return tmp_path


def test_augment_moves_x_and_y_together():
    y = np.arange(16).reshape(4, 4)
    x = np.stack([y, y * 10])
    for k in range(4):
        for fh in (False, True):
            for fv in (False, True):
                xa, ya = ds.augment(x, y, k, fh, fv)
                np.testing.assert_array_equal(xa[0], ya)
                np.testing.assert_array_equal(xa[1], ya * 10)
    xa, ya = ds.augment(x, y, 1, False, False)
    np.testing.assert_array_equal(ya, np.rot90(y))


def test_normalise_uses_channel_stats():
    x = np.full((2, 3, 3), 5.0, dtype="float16")
    mean = np.array([1.0, 5.0], dtype="float32")[:, None, None]
    std = np.array([2.0, 1.0], dtype="float32")[:, None, None]
    out = ds.normalise(x, mean, std)
    assert out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx(2.0) and out[1, 0, 0] == pytest.approx(0.0)


def test_dataset_split_shapes_and_dtypes(tile_root):
    train = ds.TileDataset(tile_root / "manifest.csv", ["train"], tile_root / "norm_stats.json")
    val = ds.TileDataset(tile_root / "manifest.csv", ["val"], tile_root / "norm_stats.json")
    assert len(train) == 2 and len(val) == 1
    x, y = val[0]
    assert x.shape == (7, 8, 8) and x.dtype == np.float32 and x.flags.c_contiguous
    assert y.shape == (1, 8, 8) and y.dtype == np.float32 and set(np.unique(y)) <= {0.0, 1.0}
    np.testing.assert_array_equal(x[0], y[0])  # eval mode: no augmentation
    assert 0 < train.forest_fraction() < 1


def test_train_mode_augments_consistently(tile_root):
    data = ds.TileDataset(
        tile_root / "manifest.csv", ["train"], tile_root / "norm_stats.json", train=True, seed=3
    )
    raw_y = np.load(tile_root / "Y_0.npy")
    seen_changed = False
    for _ in range(20):
        x, y = data[0]
        np.testing.assert_array_equal(x[0], y[0])  # label and image stay aligned
        seen_changed |= not np.array_equal(y[0], raw_y)
    assert seen_changed  # some draws actually rotate/flip
