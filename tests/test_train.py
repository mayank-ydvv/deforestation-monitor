"""Training-code tests; skipped where torch is not installed (e.g. the inference env)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("onnxruntime")

from training import train  # noqa: E402

TINY = train.TrainConfig(encoder="resnet18", encoder_weights=None, amp=False)


def test_confusion_counts_match_numpy():
    logits = torch.tensor([[2.0, -1.0, 3.0, -2.0]])
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(train.confusion_counts(logits, target, 0.5).numpy(), [1, 1, 1, 1])


def test_loss_is_lower_for_correct_logits():
    loss = train.DiceBCELoss()
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, :16] = 1
    good = (target * 2 - 1) * 8
    assert loss(good, target) < loss(-good, target)


def test_onnx_export_dynamic_batch_and_parity(tmp_path):
    model = train.build_model(TINY).eval()
    path = train.export_onnx(model, tmp_path / "m.onnx", TINY, size=64)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 7, 64, 64)).astype("float32")  # batch 3 != export batch 1
    diff = train.check_parity(model, path, x)
    assert diff <= 1e-3
