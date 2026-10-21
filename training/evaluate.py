"""Forest-mask accuracy of the exported ONNX model on held-out regions.

Runs with ONNX Runtime + NumPy only (no torch), so it works in the inference environment:

    python -m training.evaluate --model models/unet_r34.onnx

Pixel counts are pooled per region; the headline numbers pool all ``test`` regions. Writes
``web/public/data/metrics.json``. (Change-detection precision/recall vs Hansen is added in
Phase 4.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pipeline import config as cfg
from training.dataset import TileDataset

METRICS_PATH = cfg.WEB_DATA_DIR / "metrics.json"


def confusion(prob: np.ndarray, target: np.ndarray, thr: float = cfg.PROB_THR) -> np.ndarray:
    """[tp, fp, fn, tn] (int64) for the forest class over any matching shapes."""
    pred = prob > thr
    truth = target > 0.5
    return np.array(
        [
            (pred & truth).sum(),
            (pred & ~truth).sum(),
            (~pred & truth).sum(),
            (~pred & ~truth).sum(),
        ],
        dtype="int64",
    )


def scores(counts: Sequence[float]) -> dict[str, float]:
    """IoU, F1, precision and recall from [tp, fp, fn, tn].

    ``baseline_iou`` is what predicting "forest everywhere" would score (the forest share
    of the pixels): forest IoU flatters a lazy model in mostly-forest regions, so report both.
    """
    tp, fp, fn, tn = (float(c) for c in counts)
    eps = 1e-9
    precision, recall = tp / (tp + fp + eps), tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * precision * recall / (precision + recall + eps),
        "precision": precision,
        "recall": recall,
        "baseline_iou": (tp + fn) / (tp + fp + fn + tn + eps),
    }


def make_session(model_path: Path):
    """CPU ONNX Runtime session using every core."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 1
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def region_counts(
    session,
    manifest: Path,
    norm_stats: Path,
    splits: Sequence[str],
    root: Path | None = None,
    batch_size: int = cfg.INFER_BATCH,
    thr: float = cfg.PROB_THR,
) -> dict[str, dict]:
    """Per-region {split, n_tiles, counts[4], months: {month: counts[4]}} from ONNX predictions.

    Month-level counts matter: a model can do well on leaf-on imagery and fail on leaf-off
    imagery of the same forest, which a single per-region number hides.
    """
    data = TileDataset(manifest, splits, norm_stats, root=root)
    out: dict[str, dict] = {}
    for start in range(0, len(data), batch_size):
        idx = range(start, min(start + batch_size, len(data)))
        items = [data[i] for i in idx]
        x = np.stack([it[0] for it in items])
        y = np.stack([it[1] for it in items])
        prob = session.run(["prob"], {"x": x})[0]
        for j, i in enumerate(idx):
            row = data.rows[i]
            entry = out.setdefault(
                row["region"],
                {"split": row["split"], "n_tiles": 0, "counts": np.zeros(4, "int64"), "months": {}},
            )
            counts = confusion(prob[j], y[j], thr)
            entry["n_tiles"] += 1
            entry["counts"] += counts
            month = entry["months"].setdefault(row["month"], np.zeros(4, "int64"))
            month += counts
    return out


def build_metrics(per_region: dict[str, dict], headline_split: str = "test", **meta) -> dict:
    """metrics.json payload: headline scores over ``headline_split`` plus per-region scores."""
    pooled = sum(
        (r["counts"] for r in per_region.values() if r["split"] == headline_split),
        np.zeros(4, "int64"),
    )
    if not pooled.any():
        raise ValueError(f"No {headline_split!r} regions were evaluated")
    round4 = lambda d: {k: round(v, 4) for k, v in d.items()}  # noqa: E731
    return {
        **round4(scores(pooled)),
        "evaluated_on": headline_split,
        "per_region": {
            name: {
                "split": r["split"],
                "n_tiles": r["n_tiles"],
                **round4(scores(r["counts"])),
                "months": {
                    month: round4(scores(c)) for month, c in sorted(r.get("months", {}).items())
                },
            }
            for name, r in sorted(per_region.items())
        },
        **meta,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=cfg.MODEL_PATH)
    ap.add_argument("--manifest", type=Path, default=cfg.TILES_DIR / "manifest.csv")
    ap.add_argument("--norm-stats", type=Path, default=cfg.NORM_STATS_PATH)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--threshold", type=float, default=cfg.PROB_THR)
    ap.add_argument("--out", type=Path, default=METRICS_PATH)
    args = ap.parse_args(argv)

    per_region = region_counts(
        make_session(args.model), args.manifest, args.norm_stats, args.splits, thr=args.threshold
    )
    metrics = build_metrics(
        per_region,
        threshold=args.threshold,
        label_source=(
            f"ESA WorldCover {cfg.WORLDCOVER_LABEL_YEAR} (class {cfg.WORLDCOVER_TREE_CLASS})"
        ),
        model=args.model.name,
        generated=dt.date.today().isoformat(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
