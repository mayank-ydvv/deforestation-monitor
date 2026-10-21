"""U-Net (ResNet34) forest segmentation: training loop, ONNX export and parity check.

Used by ``train.ipynb`` on a Kaggle/Colab GPU; also runs (slowly) on CPU for smoke tests.
Speed settings: AMP (autocast + GradScaler), channels_last, pinned memory with
non-blocking copies, and ``zero_grad(set_to_none=True)``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dataset import TileDataset, worker_init_fn
from training.evaluate import scores


@dataclass
class TrainConfig:
    """Hyper-parameters; defaults follow the project spec."""

    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 7
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    threshold: float = 0.5
    amp: bool = True
    seed: int = 42
    max_steps_per_epoch: int | None = None  # smoke tests only


# --- Model and loss --------------------------------------------------------------------
def build_model(cfg: TrainConfig) -> nn.Module:
    """smp U-Net; with ImageNet weights the first conv is adapted to 7 channels."""
    return smp.Unet(
        cfg.encoder, encoder_weights=cfg.encoder_weights, in_channels=cfg.in_channels, classes=1
    )


class DiceBCELoss(nn.Module):
    """Dice + BCE-with-logits: Dice copes with class imbalance, BCE keeps gradients smooth."""

    def __init__(self) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss("binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, target) + self.bce(logits, target)


class ProbabilityModel(nn.Module):
    """Wraps the U-Net so the exported graph outputs forest probability, not logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


# --- Metrics ---------------------------------------------------------------------------
def confusion_counts(logits: torch.Tensor, target: torch.Tensor, thr: float) -> torch.Tensor:
    """[tp, fp, fn, tn] for the forest class, as an int64 tensor."""
    pred = torch.sigmoid(logits.float()) > thr
    truth = target > 0.5
    return torch.stack(
        [
            (pred & truth).sum(),
            (pred & ~truth).sum(),
            (~pred & truth).sum(),
            (~pred & ~truth).sum(),
        ]
    ).long()


# --- Data ------------------------------------------------------------------------------
def make_loaders(
    manifest: Path, norm_stats: Path, cfg: TrainConfig, root: Path | None = None
) -> tuple[DataLoader, DataLoader]:
    """Train loader (augmented, shuffled) and val loader, split by region via the manifest."""
    common = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "persistent_workers": cfg.num_workers > 0,
    }
    train_ds = TileDataset(manifest, ["train"], norm_stats, root=root, train=True, seed=cfg.seed)
    val_ds = TileDataset(manifest, ["val"], norm_stats, root=root)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )


# --- Loops -----------------------------------------------------------------------------
def _to(x: torch.Tensor, device: torch.device, channels_last: bool = False) -> torch.Tensor:
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    return x.to(device, non_blocking=True, memory_format=fmt)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    """One pass over the train loader; returns the mean loss."""
    model.train()
    use_amp = cfg.amp and device.type == "cuda"
    total, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if cfg.max_steps_per_epoch is not None and step >= cfg.max_steps_per_epoch:
            break
        x, y = _to(x, device, channels_last=True), _to(y, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Loss and forest IoU/F1/precision/recall over a loader (pixel-level, pooled)."""
    model.eval()
    use_amp = cfg.amp and device.type == "cuda"
    counts = torch.zeros(4, dtype=torch.long, device=device)
    total, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if cfg.max_steps_per_epoch is not None and step >= cfg.max_steps_per_epoch:
            break
        x, y = _to(x, device, channels_last=True), _to(y, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            total += loss_fn(logits, y).item()
        counts += confusion_counts(logits, y, cfg.threshold)
        n += 1
    return {"loss": total / max(n, 1), **scores(counts.cpu())}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed for third-party code
    torch.manual_seed(seed)


def fit(
    manifest: Path,
    norm_stats: Path,
    out_dir: Path,
    cfg: TrainConfig | None = None,
    root: Path | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Train, keeping the checkpoint with the best val IoU at ``out_dir/best.pt``.

    Returns the best model (loaded, on the training device) and the per-epoch history,
    which is also written to ``out_dir/history.json``.
    """
    cfg = cfg or TrainConfig()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_loaders(manifest, norm_stats, cfg, root)
    model = build_model(cfg).to(device, memory_format=torch.channels_last)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = len(train_dl) if cfg.max_steps_per_epoch is None else cfg.max_steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * min(steps, len(train_dl))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

    best_iou, history = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_dl, loss_fn, opt, sched, scaler, device, cfg)
        val = evaluate(model, val_dl, loss_fn, device, cfg)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        row["seconds"] = round(time.perf_counter() - t0, 1)
        history.append(row)
        improved = val["iou"] > best_iou
        if improved:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out_dir / "best.pt")
        print(
            f"epoch {epoch:02d}  train_loss {train_loss:.4f}  val_loss {val['loss']:.4f}  "
            f"val_iou {val['iou']:.4f} (all-forest {val['baseline_iou']:.4f})  "
            f"val_f1 {val['f1']:.4f}  {row['seconds']}s" + ("  *best" if improved else "")
        )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model, history


# --- Export ----------------------------------------------------------------------------
def export_onnx(model: nn.Module, path: Path, cfg: TrainConfig, size: int = 256) -> Path:
    """Export the probability model (opset 17, dynamic batch axis) on CPU in float32."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = ProbabilityModel(model).cpu().float().eval()
    dummy = torch.zeros(1, cfg.in_channels, size, size)
    kwargs = {
        "input_names": ["x"],
        "output_names": ["prob"],
        "dynamic_axes": {"x": {0: "batch"}, "prob": {0: "batch"}},
        "opset_version": 17,
    }
    try:  # torch >= 2.5: pin the TorchScript exporter (dynamo ignores dynamic_axes)
        torch.onnx.export(wrapped, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch has no ``dynamo`` argument
        torch.onnx.export(wrapped, (dummy,), str(path), **kwargs)
    return path


def check_parity(model: nn.Module, onnx_path: Path, x: np.ndarray, atol: float = 1e-3) -> float:
    """Max |PyTorch - ONNX Runtime| probability on ``x`` (N, C, H, W); raises above ``atol``."""
    import onnxruntime as ort

    wrapped = ProbabilityModel(model).cpu().float().eval()
    with torch.inference_mode():
        expected = wrapped(torch.from_numpy(x.astype("float32"))).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(["prob"], {"x": x.astype("float32")})[0]
    diff = float(np.abs(expected - got).max())
    if diff > atol:
        raise AssertionError(f"ONNX parity failed: max |diff| {diff:.2e} > {atol:.0e}")
    return diff
