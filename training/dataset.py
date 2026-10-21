"""Memory-mapped tile Dataset for U-Net training.

Tiles are read with ``np.load(..., mmap_mode="r")`` so RAM stays flat however large the
dataset is. Items are NumPy arrays; PyTorch's default collate turns them into tensors, so
this module imports without torch (handy for tests and data checks on a laptop).
Augmentation is geometric only (flips + rot90): spectral values carry meaning, so there is
no colour jitter.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:  # subclass torch's Dataset when available so DataLoader workers behave as usual
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch-free environments
    Dataset = object


def read_manifest(path: Path, splits: Sequence[str]) -> list[dict]:
    """Manifest rows whose ``split`` is in ``splits``."""
    with Path(path).open() as f:
        return [row for row in csv.DictReader(f) if row["split"] in splits]


def load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) as float32 arrays shaped (C, 1, 1) for broadcasting."""
    stats = json.loads(Path(path).read_text())
    mean = np.asarray(stats["mean"], dtype="float32")[:, None, None]
    std = np.asarray(stats["std"], dtype="float32")[:, None, None]
    return mean, np.maximum(std, 1e-6)


def normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(C, H, W) float16/32 -> standardised float32."""
    return (x.astype("float32") - mean) / std


def augment(
    x: np.ndarray, y: np.ndarray, k: int, flip_h: bool, flip_v: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same rot90 (k quarter turns) and flips to (C, H, W) x and (H, W) y."""
    x, y = np.rot90(x, k, axes=(-2, -1)), np.rot90(y, k, axes=(-2, -1))
    if flip_h:
        x, y = x[..., ::-1], y[..., ::-1]
    if flip_v:
        x, y = x[..., ::-1, :], y[..., ::-1, :]
    return x, y


class TileDataset(Dataset):
    """Tiles of the given splits from ``manifest.csv``.

    Each item is ``(x, y)``: x float32 (7, H, W) standardised; y float32 (1, H, W) in {0, 1}.
    With ``train=True`` a random rot90/flip is applied per item.
    """

    def __init__(
        self,
        manifest: Path,
        splits: Sequence[str],
        norm_stats: Path,
        root: Path | None = None,
        train: bool = False,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else Path(manifest).parent
        self.rows = read_manifest(manifest, splits)
        self.mean, self.std = load_norm_stats(norm_stats)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[i]
        x = np.load(self.root / row["path"], mmap_mode="r")
        y = np.load(self.root / row["y_path"], mmap_mode="r")
        if self.train:
            k = int(self.rng.integers(4))
            flip_h, flip_v = (bool(b) for b in self.rng.integers(2, size=2))
            x, y = augment(x, y, k, flip_h, flip_v)
        x = normalise(x, self.mean, self.std)
        y = np.ascontiguousarray(y, dtype="float32")[None]
        return np.ascontiguousarray(x), y

    def forest_fraction(self) -> float:
        """Mean label value over all items, from the manifest (no tile reads)."""
        return float(np.mean([float(r["forest_fraction"]) for r in self.rows]))


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation RNG.

    Workers start with copies of the parent's RNG state, so without this every worker
    would draw identical flips/rotations. Pass as ``DataLoader(..., worker_init_fn=...)``.
    """
    from torch.utils.data import get_worker_info

    info = get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(info.seed % 2**32)
