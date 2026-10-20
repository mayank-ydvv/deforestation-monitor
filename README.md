# Deforestation Monitor

A web app that shows on an interactive map where forest was cleared between two dates, using free Sentinel-2 imagery, a U-Net forest-segmentation model and multi-date change detection, updated monthly at ₹0 hosting cost.

> Work in progress: **Phase 1 (data)** is done. Full README (live link, metrics, architecture, cost, limitations, attributions) lands in Phase 6.

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

## Data attributions

Contains modified Copernicus Sentinel data (ESA), accessed via Microsoft Planetary Computer.
