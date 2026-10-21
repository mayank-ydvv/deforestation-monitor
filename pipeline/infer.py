"""Sliding-window forest-probability inference with ONNX Runtime (CPU, no torch).

Windows of 256 px with 32 px overlap sit on one *global* grid. Each 4096 x 4096 chunk of
the output gathers every window that touches it, so chunk borders get exactly the same
values as a single full-image pass (no seams), while memory stays bounded. Predictions are
blended with weights that ramp linearly only inside the overlap strips; windows without any
valid pixel are skipped; pixels without data stay NaN. Output is float16 probability,
optionally streamed to a memory-mapped ``.npy`` file.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import xarray as xr

from pipeline import config as cfg
from pipeline.tiles import model_input
from training.dataset import load_norm_stats, normalise

Origin = tuple[int, int]


def load_session(model_path: Path = cfg.MODEL_PATH):
    """ONNX Runtime CPU session using every core."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 1
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def window_starts(n: int, size: int = cfg.TILE_SIZE, overlap: int = cfg.TILE_OVERLAP) -> list[int]:
    """Window origins along one axis: stride ``size - overlap``, last one flush with the edge."""
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, size - overlap))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


def blend_weights(size: int = cfg.TILE_SIZE, overlap: int = cfg.TILE_OVERLAP) -> np.ndarray:
    """(size, size) weights: 1 in the interior, linear ramp (never 0) across overlap strips."""
    i = np.arange(size, dtype="float32")
    ramp = np.minimum(i + 0.5, size - i - 0.5) / max(overlap, 1)
    w1 = np.clip(ramp, 0.0, 1.0)
    return np.outer(w1, w1)


def chunk_bounds(n: int, chunk: int) -> Iterator[tuple[int, int]]:
    """[start, stop) ranges of length ``chunk`` covering 0..n."""
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def blend_windows(
    session,
    x: np.ndarray,
    origins: list[Origin],
    weights: np.ndarray,
    batch: int = cfg.INFER_BATCH,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched windows of normalised x (C, H, W); return (sum prob*w, sum w), (H, W)."""
    size = weights.shape[0]
    acc = np.zeros(x.shape[1:], dtype="float32")
    wsum = np.zeros(x.shape[1:], dtype="float32")
    for start in range(0, len(origins), batch):
        group = origins[start : start + batch]
        xb = np.stack([x[:, r : r + size, c : c + size] for r, c in group])
        prob = session.run(["prob"], {"x": xb})[0][:, 0]
        for (r, c), p in zip(group, prob, strict=True):
            acc[r : r + size, c : c + size] += p * weights
            wsum[r : r + size, c : c + size] += weights
    return acc, wsum


def predict(
    composite: xr.DataArray,
    session,
    norm_stats: Path = cfg.NORM_STATS_PATH,
    out_path: Path | None = None,
    chunk: int = cfg.INFER_CHUNK,
    batch: int = cfg.INFER_BATCH,
    size: int = cfg.TILE_SIZE,
    overlap: int = cfg.TILE_OVERLAP,
) -> np.ndarray:
    """Forest probability (float16, NaN = no data) on ``composite``'s grid."""
    height, width = composite.sizes["y"], composite.sizes["x"]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16", shape=(height, width))
    else:
        out = np.empty((height, width), dtype="float16")
    out[:] = np.nan
    mean, std = load_norm_stats(norm_stats)
    weights = blend_weights(size, overlap)
    all_rows = window_starts(max(height, size), size, overlap)
    all_cols = window_starts(max(width, size), size, overlap)

    for r0, r1 in chunk_bounds(height, chunk):
        rows = [r for r in all_rows if r < r1 and r + size > r0]
        for c0, c1 in chunk_bounds(width, chunk):
            cols = [c for c in all_cols if c < c1 and c + size > c0]
            # Every window touching this chunk, in global coordinates -> one region to read.
            top, bottom, left, right = rows[0], rows[-1] + size, cols[0], cols[-1] + size
            region = composite.isel(
                y=slice(top, min(bottom, height)), x=slice(left, min(right, width))
            )
            x = model_input(region)
            pad = ((0, 0), (0, bottom - top - x.shape[1]), (0, right - left - x.shape[2]))
            x = np.pad(x, pad, constant_values=np.nan)
            valid = np.isfinite(x).all(axis=0)
            x = np.nan_to_num(normalise(x, mean, std), nan=0.0)
            origins = [
                (r - top, c - left)
                for r in rows
                for c in cols
                if valid[r - top : r - top + size, c - left : c - left + size].any()
            ]
            if not origins:
                continue
            acc, wsum = blend_windows(session, x, origins, weights, batch)
            prob = np.divide(acc, wsum, out=np.full_like(acc, np.nan), where=wsum > 0)
            prob[~valid] = np.nan
            out[r0:r1, c0:c1] = prob[r0 - top : r1 - top, c0 - left : c1 - left]
    if isinstance(out, np.memmap):
        out.flush()
    return out
