"""Offline tests for forest-mask metrics (NumPy only; no model or torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_dataset import tile_root  # noqa: F401  (fixture)
from training import evaluate as ev


def test_confusion_and_scores():
    prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7])
    truth = np.array([1, 0, 1, 0, 1])
    counts = ev.confusion(prob, truth, thr=0.5)
    np.testing.assert_array_equal(counts, [2, 1, 1, 1])  # tp, fp, fn, tn
    s = ev.scores(counts)
    assert s["iou"] == pytest.approx(2 / 4)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["baseline_iou"] == pytest.approx(3 / 5)  # forest share: tp + fn over all


def test_scores_empty_is_zero_not_nan():
    expected = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "baseline_iou": 0.0}
    assert ev.scores([0, 0, 0, 10]) == expected


class _ChannelZeroSession:
    """Stands in for ONNX Runtime: 'predicts' forest wherever channel 0 is > 0.5."""

    def run(self, names, feeds):
        assert names == ["prob"]
        return [(feeds["x"][:, :1] > 0.5).astype("float32")]


def test_region_counts_and_metrics(tile_root):  # noqa: F811
    per_region = ev.region_counts(
        _ChannelZeroSession(),
        tile_root / "manifest.csv",
        tile_root / "norm_stats.json",
        ["train", "val"],
        batch_size=2,
    )
    assert set(per_region) == {"r0", "r1", "r2"}
    assert per_region["r2"]["split"] == "val" and per_region["r2"]["n_tiles"] == 1
    metrics = ev.build_metrics(per_region, headline_split="val", model="fake.onnx")
    assert metrics["iou"] == 1.0  # channel 0 mirrors the label in the fixture
    assert metrics["evaluated_on"] == "val" and metrics["model"] == "fake.onnx"
    assert set(metrics["per_region"]) == {"r0", "r1", "r2"}
    assert metrics["per_region"]["r2"]["months"]["2021-01"]["iou"] == 1.0  # per-month scores


def test_build_metrics_requires_headline_split():
    with pytest.raises(ValueError):
        ev.build_metrics({"r": {"split": "val", "n_tiles": 1, "counts": np.array([1, 0, 0, 0])}})
