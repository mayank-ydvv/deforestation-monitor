"""Cloud-masked monthly Sentinel-2 median composites from Planetary Computer.

Everything is lazy (stackstac + dask) until the final ``.compute()``: only the AOI window
of each COG is read over HTTP range requests, and the median is vectorised over time.
"""

from __future__ import annotations

import argparse
import calendar
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import planetary_computer
import pystac
import pystac_client
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import stackstac
import xarray as xr
from rasterio.enums import Resampling

from pipeline import config as cfg

CACHE_NODATA = 65535  # composites are cached as uint16 reflectance x 10000
GDAL_ENV = stackstac.DEFAULT_GDAL_ENV.updated(
    always={
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        # Fail a stalled request (< 5 kB/s for 60 s) instead of hanging; composite() retries.
        "GDAL_HTTP_TIMEOUT": "120",
        "GDAL_HTTP_LOW_SPEED_LIMIT": "5000",
        "GDAL_HTTP_LOW_SPEED_TIME": "60",
    }
)
READ_ERRORS = (RuntimeError, rasterio.errors.RasterioIOError)  # stackstac wraps GDAL errors


class Composite(NamedTuple):
    """A median composite and how much of the AOI it actually covers."""

    data: xr.DataArray  # (band=6, y, x) float32 surface reflectance in [0, 1], NaN = no data
    clear_fraction: float  # share of pixels with a valid value in every band
    n_items: int  # scenes that went into the median


class NoScenesError(LookupError):
    """No Sentinel-2 scene matched the AOI, dates and cloud filter."""


# --- Pure helpers ----------------------------------------------------------------------
def month_range(month: str) -> str:
    """'2025-01' -> '2025-01-01/2025-01-31' (a STAC datetime interval)."""
    year, mon = (int(p) for p in month.split("-"))
    if not 1 <= mon <= 12:
        raise ValueError(f"Bad month {month!r}; expected YYYY-MM")
    last = calendar.monthrange(year, mon)[1]
    return f"{year:04d}-{mon:02d}-01/{year:04d}-{mon:02d}-{last:02d}"


def baseline_offsets(baselines: Sequence[str | None]) -> np.ndarray:
    """Per-scene DN offset: 1000 for processing baseline >= 04.00, else 0."""
    versions = np.array([float(b) if b else 0.0 for b in baselines])
    return np.where(versions >= cfg.OFFSET_BASELINE, cfg.BOA_ADD_OFFSET, 0.0).astype("float32")


def harmonise(dn: xr.DataArray, offsets: np.ndarray) -> xr.DataArray:
    """L2A digital numbers (time, band, y, x) -> reflectance in [0, 1].

    DN 0 is Sentinel-2's no-data value and becomes NaN.
    """
    offset = xr.DataArray(offsets, dims="time")
    refl = (dn - offset) / cfg.REFLECTANCE_SCALE
    return refl.where(dn > 0).clip(0.0, 1.0)


def clear_mask(scl: xr.DataArray) -> xr.DataArray:
    """True where the scene classification marks a usable (cloud/shadow-free) pixel."""
    return scl.notnull() & ~scl.isin(cfg.SCL_MASK_CLASSES)


def ndvi(red: xr.DataArray | np.ndarray, nir: xr.DataArray | np.ndarray):
    """(NIR - red) / (NIR + red + eps); works on NumPy and xarray alike."""
    return (nir - red) / (nir + red + cfg.NDVI_EPS)


def clear_fraction(data: xr.DataArray) -> float:
    """Share of pixels that have a finite value in every band."""
    return float(np.isfinite(data).all("band").mean())


# --- STAC + stacking -------------------------------------------------------------------
def search_items(
    bbox: Sequence[float], date_range: str, max_cloud: float = cfg.MAX_SCENE_CLOUD
) -> list[pystac.Item]:
    """Signed Sentinel-2 L2A items intersecting ``bbox``, oldest first."""
    catalog = pystac_client.Client.open(cfg.STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[cfg.S2_COLLECTION],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return sorted(search.item_collection(), key=lambda it: it.datetime)


def select_items(items: Sequence[pystac.Item], max_n: int = cfg.MAX_SCENES) -> list[pystac.Item]:
    """The ``max_n`` least-cloudy scenes, returned oldest first."""
    clearest = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))[:max_n]
    return sorted(clearest, key=lambda it: it.datetime)


def _stack(
    items: list[pystac.Item],
    assets: Sequence[str],
    bbox: Sequence[float],
    epsg: int,
    resampling: Resampling,
) -> xr.DataArray:
    """Lazy (time, band, y, x) stack of ``assets`` on the AOI's 10 m UTM grid."""
    return stackstac.stack(
        items,
        assets=list(assets),
        epsg=epsg,
        resolution=cfg.RESOLUTION_M,
        bounds_latlon=tuple(bbox),
        dtype="float32",
        fill_value=np.float32(np.nan),
        rescale=False,
        resampling=resampling,
        sortby_date=False,  # keep our order so offsets line up with the time axis
        xy_coords="center",
        properties=False,
        chunksize=cfg.DASK_CHUNK,
        gdal_env=GDAL_ENV,
    )


def _median_composite(items: list[pystac.Item], bbox: Sequence[float], epsg: int) -> xr.DataArray:
    """Build the lazy masked median; one stack per resampling rule on the same grid."""
    dn10 = _stack(items, cfg.BANDS_10M, bbox, epsg, Resampling.nearest)
    dn20 = _stack(items, cfg.BANDS_20M, bbox, epsg, Resampling.bilinear)
    scl = _stack(items, [cfg.SCL_ASSET], bbox, epsg, Resampling.nearest).sel(band=cfg.SCL_ASSET)
    dn = xr.concat([dn10, dn20], dim="band", coords="minimal", compat="override", join="exact")
    offsets = baseline_offsets([it.properties.get("s2:processing_baseline") for it in items])
    refl = harmonise(dn, offsets).where(clear_mask(scl))
    med = refl.chunk({"time": -1}).median("time", skipna=True)
    return med.transpose("band", "y", "x").astype("float32")


# --- Cache (uint16 COG) ----------------------------------------------------------------
def write_cache(comp: Composite, path: Path, epsg: int) -> None:
    """Store a composite as a DEFLATE-compressed uint16 COG with its stats in tags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = (comp.data * cfg.REFLECTANCE_SCALE).round()
    out = scaled.fillna(CACHE_NODATA).astype("uint16")
    out = out.drop_vars([c for c in out.coords if c not in ("band", "y", "x")])
    out = out.assign_coords(band=np.arange(1, out.sizes["band"] + 1))
    out = out.rio.write_crs(epsg).rio.write_nodata(CACHE_NODATA)
    out.rio.to_raster(
        path,
        driver="COG",
        compress="DEFLATE",
        predictor=2,
        tags={
            "CLEAR_FRACTION": f"{comp.clear_fraction:.6f}",
            "N_ITEMS": str(comp.n_items),
            "BANDS": ",".join(cfg.BANDS),
        },
    )


def read_cache(path: Path) -> Composite:
    """Load a cached composite back as float32 reflectance."""
    with rasterio.open(path) as src:
        tags = src.tags()
    raw = rioxarray.open_rasterio(path, masked=True)
    data = (raw / cfg.REFLECTANCE_SCALE).astype("float32")
    data = data.assign_coords(band=list(tags["BANDS"].split(",")))
    return Composite(data, float(tags["CLEAR_FRACTION"]), int(tags["N_ITEMS"]))


# --- Public API ------------------------------------------------------------------------
def composite(
    bbox: Sequence[float], date_range: str, epsg: int, cache_path: Path | None = None
) -> Composite:
    """Cloud-masked median composite of the 6 input bands over ``date_range``.

    Returns the cached result when ``cache_path`` exists; otherwise computes it and,
    if ``cache_path`` is given, writes it there. Raises ``NoScenesError`` when no scene
    passes the cloud filter. Callers skip the month when ``clear_fraction`` is below
    ``config.MIN_CLEAR_FRACTION``.
    """
    if cache_path is not None and cache_path.exists():
        return read_cache(cache_path)
    for attempt in range(1, cfg.FETCH_ATTEMPTS + 1):
        items = select_items(search_items(bbox, date_range))  # re-signed every attempt
        if not items:
            raise NoScenesError(f"No scenes < {cfg.MAX_SCENE_CLOUD}% cloud for {date_range}")
        try:
            with warnings.catch_warnings():  # pixels with no clear scene stay NaN, by design
                warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
                data = _median_composite(items, bbox, epsg).compute()
            break
        except READ_ERRORS as err:
            if attempt == cfg.FETCH_ATTEMPTS:
                raise
            print(
                f"  read failed for {date_range} (attempt {attempt}/{cfg.FETCH_ATTEMPTS}): "
                f"{str(err).splitlines()[0][:120]}; retrying",
                file=sys.stderr,
            )
    comp = Composite(data.rio.write_crs(epsg), clear_fraction(data), len(items))
    if cache_path is not None:
        write_cache(comp, cache_path, epsg)
    return comp


def cache_path_for(aoi_id: str, month: str) -> Path:
    """cache/<aoi>/<YYYY-MM>/composite.tif"""
    return cfg.CACHE_DIR / aoi_id / month / "composite.tif"


def monthly_composite(aoi: cfg.AOI, month: str) -> Composite:
    """Cached composite for one AOI and calendar month ('YYYY-MM')."""
    return composite(aoi.bbox, month_range(month), aoi.epsg, cache_path_for(aoi.id, month))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m pipeline.fetch --aoi hasdeo --month 2025-01``."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--aoi", required=True, choices=sorted(cfg.AOIS))
    ap.add_argument("--month", required=True, help="YYYY-MM")
    args = ap.parse_args(argv)
    aoi = cfg.get_aoi(args.aoi)
    t0 = time.perf_counter()
    comp = monthly_composite(aoi, args.month)
    dt = time.perf_counter() - t0
    usable = "ok" if comp.clear_fraction >= cfg.MIN_CLEAR_FRACTION else "too cloudy, skip"
    print(
        f"{aoi.id} {args.month}: shape={tuple(comp.data.shape)} scenes={comp.n_items} "
        f"clear={comp.clear_fraction:.2%} ({usable}) in {dt:.1f}s -> "
        f"{cache_path_for(aoi.id, args.month)}"
    )


if __name__ == "__main__":
    main()
