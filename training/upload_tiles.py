"""Publish the training tiles (+ norm stats and the code train.ipynb needs) to Hugging Face.

    hf auth login                    # once, with a write token (older hub: huggingface-cli login)
    python -m training.upload_tiles --repo <user>/deforestation-monitor-tiles

Resumable on both huggingface_hub generations (``upload_large_folder`` on 0.x, the batching
``upload_folder`` on 2.x): if the connection drops, run it again and it continues where it
stopped. Layout of the dataset repo:

    manifest.csv, <aoi>/<month>/{X,Y}_*.npy     tiles (as in tiles/)
    norm_stats.json                             per-channel mean/std (train split)
    code/pipeline/{__init__,config}.py          imported by the training code
    code/training/*.py                          dataset, train, evaluate
    README.md                                   dataset card with licences
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pipeline import config as cfg

CODE_FILES = (
    "pipeline/__init__.py",
    "pipeline/config.py",
    "training/__init__.py",
    "training/dataset.py",
    "training/evaluate.py",
    "training/train.py",
)

CARD = """---
license: cc-by-4.0
task_categories: [image-segmentation]
tags: [remote-sensing, sentinel-2, deforestation, forest]
---
# Deforestation Monitor: forest-segmentation training tiles

256x256 Sentinel-2 L2A median composites (B02, B03, B04, B08, B11, B12 + NDVI; float16,
reflectance) with ESA WorldCover 2021 tree-cover labels (uint8 0/1), split by region.
See `manifest.csv` for region, split, month and forest fraction per tile.

Sources: contains modified Copernicus Sentinel data (2021), via Microsoft Planetary
Computer. Labels: ESA WorldCover 2021 v200, (c) ESA WorldCover project, CC BY 4.0.
"""


def stage_extras(dest: Path) -> None:
    """Copy norm stats, the dataset card and the code the notebook imports into ``dest``."""
    shutil.copy2(cfg.NORM_STATS_PATH, dest / "norm_stats.json")
    (dest / "README.md").write_text(CARD)
    for rel in CODE_FILES:
        target = dest / "code" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.ROOT / rel, target)


TILE_PATTERNS = ["manifest.csv", "*/*/*.npy", "*/*/*.json", "*/*/*.csv"]


def publish(api, repo: str, private: bool = False, tiles_dir: Path | None = None) -> None:
    """Create the dataset repo, upload extras, then the tiles with the resumable uploader."""
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        extras = Path(tmp)
        stage_extras(extras)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=extras)
    # 0.x has a dedicated resumable uploader; 2.x folded batching + resume into upload_folder.
    upload = getattr(api, "upload_large_folder", None) or api.upload_folder
    upload(
        repo_id=repo,
        repo_type="dataset",
        folder_path=tiles_dir or cfg.TILES_DIR,
        allow_patterns=TILE_PATTERNS,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="<user>/<name> of the HF dataset repo")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    publish(HfApi(), args.repo, args.private)
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
