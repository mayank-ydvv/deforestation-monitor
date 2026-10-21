"""Write the data contract the static site reads (see docs/data-contract.json).

    web/public/data/
      aois.json                          [{id, name, bbox, center, months}]
      metrics.json                       written by training/evaluate.py
      <aoi>/index.json                   [{month, t1, hectares, patches}]
      <aoi>/<YYYY-MM>/cleared.geojson    FeatureCollection, properties from change.py
      <aoi>/<YYYY-MM>/forest.tif         COG, uint8 1 = forest, 0 = not, 255 = no data
      <aoi>/<YYYY-MM>/rgb/{z}/{x}/{y}.png  true-colour XYZ tiles (z 8-14)

JSON stays in git; ``forest.tif`` and ``rgb/`` are large and are git-ignored here (uploaded
to the raster store in the monthly job). ``aois.json`` gives each AOI a ``raster_base``:
"" (paths relative to the data root) or the raster store URL; the site loads
``{raster_base}{aoi}/{month}/forest.tif`` and ``.../rgb/{z}/{x}/{y}.png``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from pipeline import config as cfg

MAX_GEOJSON_BYTES = 1_000_000
FOREST_NODATA = 255
RGB_MAX_REFLECTANCE = 0.25  # fixed stretch so every month renders on the same scale
RGB_GAMMA = 1 / 1.6
TILE_PX = 256
ZOOMS = range(8, 15)
WEB_MERCATOR_HALF = 20037508.342789244


# --- GeoJSON ---------------------------------------------------------------------------
def feature_collection(features: list[dict]) -> dict:
    """Wrap features as a GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": features}


def write_cleared_geojson(
    make_features: Callable[[float], list[dict]],
    path: Path,
    tolerance_m: float = cfg.SIMPLIFY_M,
    max_bytes: int = MAX_GEOJSON_BYTES,
) -> list[dict]:
    """Write cleared.geojson, doubling the simplify tolerance until it fits in ``max_bytes``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        features = make_features(tolerance_m)
        text = json.dumps(feature_collection(features), separators=(",", ":"))
        if len(text.encode()) <= max_bytes or tolerance_m > 160:
            path.write_text(text)
            return features
        tolerance_m *= 2


# --- Rasters ---------------------------------------------------------------------------
def write_forest_cog(
    prob: np.ndarray, transform: Affine, epsg: int, path: Path, thr: float = cfg.PROB_THR
) -> None:
    """Model forest mask as a DEFLATE COG: 1 forest, 0 non-forest, 255 no data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p = prob.astype("float32")
    mask = np.where(np.isnan(p), FOREST_NODATA, p > thr).astype("uint8")
    profile = {
        "driver": "COG",
        "dtype": "uint8",
        "count": 1,
        "height": mask.shape[0],
        "width": mask.shape[1],
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": FOREST_NODATA,
        "compress": "DEFLATE",
        "overview_resampling": "nearest",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)


def rgba_8bit(composite: xr.DataArray) -> np.ndarray:
    """(4, H, W) uint8 true colour with a fixed stretch; alpha 0 where there is no data."""
    rgb = composite.sel(band=["B04", "B03", "B02"]).values.astype("float32")
    valid = np.isfinite(rgb).all(axis=0)
    scaled = np.clip(np.nan_to_num(rgb) / RGB_MAX_REFLECTANCE, 0, 1) ** RGB_GAMMA
    out = np.empty((4, *rgb.shape[1:]), dtype="uint8")
    out[:3] = np.round(scaled * 255)
    out[3] = np.where(valid, 255, 0)
    return out


def tile_range(bounds_lonlat: tuple[float, float, float, float], zoom: int) -> tuple[range, range]:
    """XYZ tile x and y ranges covering (west, south, east, north) at ``zoom``."""
    w, s, e, n = bounds_lonlat
    n_tiles = 2**zoom

    def tx(lon: float) -> int:
        return min(n_tiles - 1, max(0, int((lon + 180) / 360 * n_tiles)))

    def ty(lat: float) -> int:
        r = math.radians(lat)
        y = (1 - math.asinh(math.tan(r)) / math.pi) / 2 * n_tiles
        return min(n_tiles - 1, max(0, int(y)))

    return range(tx(w), tx(e) + 1), range(ty(n), ty(s) + 1)


def tile_transform(x: int, y: int, zoom: int, px: int = TILE_PX) -> Affine:
    """Affine transform of XYZ tile (x, y, zoom) in EPSG:3857."""
    span = 2 * WEB_MERCATOR_HALF / 2**zoom
    left = -WEB_MERCATOR_HALF + x * span
    top = WEB_MERCATOR_HALF - y * span
    return Affine(span / px, 0, left, 0, -span / px, top)


def write_xyz_tiles(
    rgba: np.ndarray,
    transform: Affine,
    epsg: int,
    out_dir: Path,
    zooms: Iterable[int] = ZOOMS,
) -> int:
    """Warp an RGBA image into {z}/{x}/{y}.png tiles; empty tiles are skipped. Returns count."""
    h, w = rgba.shape[1:]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    bounds = transform_bounds(f"EPSG:{epsg}", "EPSG:4326", left, bottom, right, top)
    src_res = abs(transform.a)
    written = 0
    for z in zooms:
        tile_res = 2 * WEB_MERCATOR_HALF / 2**z / TILE_PX
        # Downsampling a lot -> average (no aliasing); near native -> bilinear.
        resampling = Resampling.average if tile_res > 2 * src_res else Resampling.bilinear
        xs, ys = tile_range(bounds, z)
        for x in xs:
            for y in ys:
                dst = np.zeros((4, TILE_PX, TILE_PX), dtype="uint8")
                reproject(
                    rgba,
                    dst,
                    src_transform=transform,
                    src_crs=f"EPSG:{epsg}",
                    dst_transform=tile_transform(x, y, z),
                    dst_crs="EPSG:3857",
                    resampling=resampling,
                )
                if not dst[3].any():
                    continue
                path = out_dir / str(z) / str(x) / f"{y}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.moveaxis(dst, 0, -1), "RGBA").save(path, optimize=True)
                written += 1
    return written


# --- Index files -----------------------------------------------------------------------
def month_dir(aoi_id: str, month: str, root: Path = cfg.WEB_DATA_DIR) -> Path:
    """web/public/data/<aoi>/<YYYY-MM>/"""
    return root / aoi_id / month


def update_index(
    aoi_id: str,
    entry: dict,
    root: Path = cfg.WEB_DATA_DIR,
) -> list[dict]:
    """Insert or replace ``entry`` (keyed by month) in <aoi>/index.json, sorted by month."""
    path = root / aoi_id / "index.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [r for r in rows if r["month"] != entry["month"]] + [entry]
    rows.sort(key=lambda r: r["month"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def update_aois(root: Path = cfg.WEB_DATA_DIR, raster_base: str = "") -> list[dict]:
    """Rebuild aois.json from the demo AOIs and the months present in each index.json."""
    out = []
    for aoi in cfg.AOIS.values():
        if not aoi.demo:
            continue
        index = root / aoi.id / "index.json"
        rows = json.loads(index.read_text()) if index.exists() else []
        months = sorted({m for r in rows for m in (r["t1"], r["month"])})
        lat, lon = aoi.center
        out.append(
            {
                "id": aoi.id,
                "name": aoi.name,
                "bbox": list(aoi.bbox),
                "center": [round(lat, 5), round(lon, 5)],
                "months": months,
                "compare": aoi.compare,
                "raster_base": raster_base,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "aois.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
