"""Change detection: cleared-forest patches between two dates, as GeoJSON-ready features.

A pixel is cleared only when the model says forest at t1 *and* non-forest at t2 *and* NDVI
dropped by more than 0.2; requiring both signals suppresses false alarms from haze and
leaf-off. Every step is O(N) in pixels: boolean masks, one morphological opening, one
connected-component pass, and ``bincount`` for all per-patch statistics at once.
Polygonising is O(boundary pixels).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

from pipeline import config as cfg

Compare = Literal["consecutive", "yoy"]
PIXEL_HA = cfg.RESOLUTION_M**2 / 10_000  # 10 m pixel = 0.01 ha
OPENING = np.ones((3, 3), dtype=bool)  # 3x3 square: drops speckle and 1-2 px slivers


def cleared(
    p1: np.ndarray,
    p2: np.ndarray,
    ndvi1: np.ndarray,
    ndvi2: np.ndarray,
    thr: float = cfg.PROB_THR,
    dndvi: float = cfg.NDVI_DROP,
    min_px: int = cfg.MIN_AREA_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kept mask, label image, size per label) of cleared patches of at least ``min_px``.

    NaN anywhere (no data at either date) compares False, so it never counts as cleared.
    """
    m = (p1 > thr) & (p2 < 1 - thr) & ((ndvi1 - ndvi2) > dndvi)
    m = ndimage.binary_opening(m, structure=OPENING)
    lab, _ = ndimage.label(m)  # 4-connected components
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab], lab, sizes


def patch_stats(
    lab: np.ndarray,
    sizes: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    ndvi_drop: np.ndarray,
    min_px: int = cfg.MIN_AREA_PX,
) -> dict[int, dict[str, float]]:
    """Per kept label: area (ha), mean NDVI drop and confidence = mean of p1 * (1 - p2)."""
    ids = np.flatnonzero(sizes >= min_px)
    ids = ids[ids != 0]
    flat = lab.ravel()
    n = len(sizes)
    conf = np.nan_to_num((p1.astype("float32") * (1 - p2.astype("float32"))).ravel())
    drop = np.nan_to_num(ndvi_drop.astype("float32").ravel())
    conf_mean = np.bincount(flat, weights=conf, minlength=n) / np.maximum(sizes, 1)
    drop_mean = np.bincount(flat, weights=drop, minlength=n) / np.maximum(sizes, 1)
    return {
        int(i): {
            "area_ha": round(float(sizes[i] * PIXEL_HA), 2),
            "ndvi_drop": round(float(drop_mean[i]), 3),
            "confidence": round(float(conf_mean[i]), 3),
        }
        for i in ids
    }


def confirmed_ids(
    lab: np.ndarray, sizes: np.ndarray, p_next: np.ndarray, thr: float = cfg.PROB_THR
) -> set[int]:
    """Labels still non-forest in the next composite (mean p_next < 1 - thr). v2 feature."""
    p = np.nan_to_num(p_next.astype("float32"), nan=1.0).ravel()  # no data -> not confirmed
    mean = np.bincount(lab.ravel(), weights=p, minlength=len(sizes)) / np.maximum(sizes, 1)
    return {int(i) for i in np.flatnonzero(mean < 1 - thr) if i != 0}


def polygons(
    lab: np.ndarray,
    kept: np.ndarray,
    transform: Affine,
    epsg: int,
    tolerance_m: float = cfg.SIMPLIFY_M,
) -> dict[int, dict]:
    """GeoJSON geometries (EPSG:4326) per kept label, simplified in metres before reprojecting."""
    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True)
    geoms: dict[int, list] = {}
    for geom, value in shapes(lab.astype("int32"), mask=kept, transform=transform):
        geoms.setdefault(int(value), []).append(shape(geom))
    out = {}
    for value, parts in geoms.items():
        merged = shapely.unary_union(parts).simplify(tolerance_m, preserve_topology=True)
        lonlat = shapely.transform(merged, lambda xy: np.column_stack(to_lonlat.transform(*xy.T)))
        out[value] = shapely.geometry.mapping(shapely.set_precision(lonlat, 1e-6))
    return out


def features(
    p1: np.ndarray,
    p2: np.ndarray,
    ndvi1: np.ndarray,
    ndvi2: np.ndarray,
    transform: Affine,
    epsg: int,
    t1: str,
    t2: str,
    aoi_id: str,
    p_next: np.ndarray | None = None,
    tolerance_m: float = cfg.SIMPLIFY_M,
) -> list[dict]:
    """GeoJSON Features for patches cleared between t1 and t2, largest first."""
    kept, lab, sizes = cleared(p1, p2, ndvi1, ndvi2)
    stats = patch_stats(lab, sizes, p1, p2, ndvi1 - ndvi2)
    confirmed = confirmed_ids(lab, sizes, p_next) if p_next is not None else set()
    geoms = polygons(lab, kept, transform, epsg, tolerance_m)
    order = sorted(stats, key=lambda i: -stats[i]["area_ha"])
    return [
        {
            "type": "Feature",
            "geometry": geoms[i],
            "properties": {
                "id": f"{aoi_id}-{t2}-{n:04d}",
                "area_ha": stats[i]["area_ha"],
                "t1": t1,
                "t2": t2,
                "ndvi_drop": stats[i]["ndvi_drop"],
                "confidence": stats[i]["confidence"],
                "confirmed": i in confirmed,
            },
        }
        for n, i in enumerate(order, start=1)
    ]


def reference_month(month: str, mode: Compare, available: Sequence[str] = ()) -> str:
    """The 'before' month for ``month``.

    ``yoy``: the same month a year earlier (seasonal forests: avoids leaf-off false alarms).
    ``consecutive``: the latest available month before ``month`` (cloudy months are skipped).
    """
    year, mon = (int(p) for p in month.split("-"))
    if mode == "yoy":
        return f"{year - 1:04d}-{mon:02d}"
    earlier = sorted(m for m in available if m < month)
    if earlier:
        return earlier[-1]
    prev_year, prev_mon = (year, mon - 1) if mon > 1 else (year - 1, 12)
    return f"{prev_year:04d}-{prev_mon:02d}"
