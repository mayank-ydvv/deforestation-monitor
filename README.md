# Deforestation Monitor

A web app that shows on an interactive map where forest was cleared between two dates, using free Sentinel-2 imagery, a U-Net forest-segmentation model and multi-date change detection, updated monthly at ₹0 hosting cost.

> Work in progress: **Phase 1 (data)**, **Phase 2 (labels and tiles)**, **Phase 3 (training)** and **Phase 4 (change detection)** are in place. Full README (live link, metrics, architecture, cost, limitations, attributions) lands in Phase 6.

## Setup

```bash
brew install uv
uv python install 3.11
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements-infer.txt -r requirements-dev.txt
```

## Phase 1: monthly composites

```bash
# Cloud-masked median composite for one AOI and month (cached as a COG)
.venv/bin/python -m pipeline.fetch --aoi hasdeo --month 2025-01

# RGB + NDVI plots for three AOIs
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_composites.ipynb

# Lint and tests (offline)
.venv/bin/ruff check . && .venv/bin/python -m pytest -q
```

- AOIs, bands, thresholds and paths live in `pipeline/config.py`.
- `pipeline/fetch.py` searches Planetary Computer (scenes < 20 % cloud, up to the 6 clearest per month), stacks the 10 m bands, the 20 m SWIR bands (bilinear to 10 m) and SCL lazily, removes the +1000 DN offset of processing baseline ≥ 04.00, masks SCL classes 0/1/3/8/9/10 and takes the per-pixel median.
- Composites are cached at `cache/<aoi>/<YYYY-MM>/composite.tif` (uint16 reflectance × 10000, DEFLATE COG); a month is skipped downstream when its clear fraction is below 60 %.

- Network reads that drop or stall (< 5 kB/s for 60 s) fail fast, and the whole composite is retried up to 3 times with freshly signed URLs; a composite with read holes is never cached.

## Phase 2: labels and training tiles

```bash
# All AOIs' label months: composites -> WorldCover labels -> 256x256 tiles -> manifest/stats/report
.venv/bin/python -m pipeline.tiles

# One AOI, or just rebuild manifest, norm stats and report from tiles on disk
.venv/bin/python -m pipeline.tiles --aoi rondonia
.venv/bin/python -m pipeline.tiles --report-only
```

- `pipeline/labels.py` warps ESA WorldCover 2021 (class 10 = tree cover → 1) onto each composite's exact 10 m grid with nearest-neighbour, reading only the AOI window. Labels are cached at `cache/<aoi>/worldcover_2021.tif`.
- Training composites come from **2021** (`label_months` per AOI in `config.py`) so imagery matches the label year; pairing 2021 labels with later imagery would label recent clearings as forest.
- Hand-corrected labels: drop a GeoTIFF at `labels/overrides/<aoi>.tif` (0/1, 255 = keep WorldCover); it is reprojected onto the grid and applied last.
- `pipeline/tiles.py` cuts 7-channel inputs (six bands + NDVI) into 256×256 tiles with stride 256, drops tiles with > 10 % NaN pixels, fills remaining NaN with the channel mean, and saves `X_*.npy` (float16) / `Y_*.npy` (uint8) under `tiles/<aoi>/<month>/`.
- Splits are by **region** (`split` per AOI), never per tile. Outputs: `tiles/manifest.csv`, `models/norm_stats.json` (per-channel mean/std from the train split only) and `docs/class_balance.md`.

## Phase 3: training (Kaggle/Colab GPU) and evaluation

```bash
# 1. Publish tiles + norm stats + training code to a Hugging Face dataset (resumable)
.venv/bin/hf auth login
.venv/bin/python -m training.upload_tiles --repo <you>/deforestation-monitor-tiles

# 2. Open training/train.ipynb on Kaggle (GPU + Internet on) or Colab (T4), set HF_REPO, run all.
#    Download outputs/unet_r34.onnx -> models/, outputs/metrics.json -> web/public/data/

# 3. Re-score locally with ONNX Runtime only (no torch)
.venv/bin/python -m training.evaluate --model models/unet_r34.onnx
```

- `training/train.py`: `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=7, classes=1)`, Dice + BCE-with-logits, AdamW (lr 1e-3, wd 1e-4), OneCycleLR, batch 16, AMP, `channels_last`, pinned memory, `non_blocking` copies, `zero_grad(set_to_none=True)`; keeps the checkpoint with the best validation IoU.
- Export: the model is wrapped with a sigmoid so the ONNX graph (opset 17, dynamic batch) outputs forest probability; `check_parity` asserts ONNX Runtime matches PyTorch within 1e-3.
- `training/evaluate.py` scores forest IoU, F1, precision and recall per region and pooled over the held-out `test` region(s), writing `web/public/data/metrics.json`. Each score comes with `baseline_iou`, what "forest everywhere" would get, because forest IoU flatters lazy models in mostly-forest regions.
**Results (current model, run 2, held-out Hasdeo):** IoU **0.806** overall (all-forest baseline 0.618), F1 0.893, precision 0.940, recall 0.850. By season: January (leaf-on) **0.879**, which meets the > 0.85 target; April (leaf-off sal) 0.731, which does not. Validation (Western Ghats) IoU 0.933. See [docs/training_runs/](docs/training_runs/README.md) for both runs and what changed between them.

- Local smoke test without a GPU: `uv venv -p 3.11 .venv-train && uv pip install -p .venv-train/bin/python -r requirements-train.txt pytest`, then `.venv-train/bin/python -m pytest tests/test_train.py`. Torch stays out of `.venv` (the inference environment).

## Phase 4: change detection and export

```bash
# One AOI, one before/after pair: composites -> forest probability -> cleared patches -> data contract
.venv/bin/python -m pipeline.process --aoi hasdeo --t1 2025-01 --t2 2026-01

# Validate against Hansen GFC loss (merged into web/public/data/metrics.json)
.venv/bin/python -m training.evaluate --hansen hasdeo:2025-01:2026-01:2025 rondonia:2024-07:2025-07:2024,2025
```

- `pipeline/infer.py`: ONNX Runtime sliding window (256 px, 32 px overlap) on one global window grid, so 4096 px chunks join without seams; linear blending only in overlap strips; float16 output streamed to a memory-mapped file. Hasdeo (9 M px) takes about 9 s on an M4.
- `pipeline/change.py`: a pixel is cleared only if the model says forest then non-forest **and** NDVI dropped by more than 0.2; 3×3 opening, 4-connected patches of at least 0.5 ha, all per-patch statistics in one `bincount` pass, polygons simplified (5 m) in UTM and then reprojected to lon/lat. Seasonal AOIs (`compare="yoy"` in config) are compared with the same month a year earlier.
- `pipeline/export.py`: writes the data contract ([docs/data-contract.json](docs/data-contract.json)): `cleared.geojson` (auto re-simplified to stay under 1 MB), forest COG, `index.json`, `aois.json` and true-colour XYZ tiles z 8–14 from a small Python tiler (rasterio + Pillow, no GDAL command-line tools). Rasters are git-ignored; JSON is committed.
- `pipeline/hansen.py`: Hansen GFC-2025-v1.13 loss year for the AOI window only; validation, never training.

**Results vs Hansen:** Hasdeo (mine expansion, Jan 2025 → Jan 2026) pixel precision 0.725, recall 0.801 (165 ha detected vs 149 ha Hansen loss). Rondônia (Jul 2024 → Jul 2025) scores 0.142 / 0.024 against same-year loss, because Hansen's 2024–25 loss there is mostly fire-damaged standing forest, while 99 % of our detections are re-clearing on land Hansen first recorded as lost in earlier years. Details: [docs/change_validation.md](docs/change_validation.md).

## Data attributions

Contains modified Copernicus Sentinel data (ESA), accessed via Microsoft Planetary Computer.
ESA WorldCover 2021 v200 © ESA WorldCover project, licensed CC BY 4.0.
Hansen/UMD/Google/USGS/NASA Global Forest Change 2000–2025 (v1.13), CC BY 4.0, used for validation only.
