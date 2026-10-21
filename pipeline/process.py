"""One change-detection step for an AOI: composites -> forest probability -> cleared -> export.

    python -m pipeline.process --aoi hasdeo --t1 2025-01 --t2 2026-01

Probability maps are cached as float16 ``cache/<aoi>/<YYYY-MM>/prob.npy`` so no month is
inferred twice. ``run_month.py`` (Phase 6) calls :func:`process_pair` for each new month.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pipeline import change, export, infer
from pipeline import config as cfg
from pipeline.fetch import Composite, cache_path_for, monthly_composite, ndvi
from pipeline.labels import grid_of


def prob_path_for(aoi_id: str, month: str) -> Path:
    """cache/<aoi>/<YYYY-MM>/prob.npy"""
    return cfg.CACHE_DIR / aoi_id / month / "prob.npy"


def forest_probability(aoi: cfg.AOI, month: str, session) -> tuple[Composite, np.ndarray]:
    """Composite and float16 forest probability for one month (both cached)."""
    comp = monthly_composite(aoi, month)
    path = prob_path_for(aoi.id, month)
    if path.exists():
        prob = np.load(path, mmap_mode="r")
        if prob.shape == (comp.data.sizes["y"], comp.data.sizes["x"]):
            return comp, prob
    return comp, infer.predict(comp.data, session, out_path=path)


def composite_ndvi(comp: Composite) -> np.ndarray:
    """NDVI (float32) of a composite."""
    return ndvi(comp.data.sel(band="B04"), comp.data.sel(band="B08")).values.astype("float32")


def export_month_rasters(aoi: cfg.AOI, month: str, comp: Composite, prob: np.ndarray) -> None:
    """Forest COG and RGB tiles for one month, unless already written."""
    out = export.month_dir(aoi.id, month)
    transform = grid_of(comp.data).transform
    if not (out / "forest.tif").exists():
        export.write_forest_cog(prob, transform, aoi.epsg, out / "forest.tif")
    if not (out / "rgb").exists():
        export.write_xyz_tiles(export.rgba_8bit(comp.data), transform, aoi.epsg, out / "rgb")


def process_pair(
    aoi: cfg.AOI,
    t1: str,
    t2: str,
    session,
    t_next: str | None = None,
    rasters: bool = True,
) -> dict:
    """Detect clearing between t1 and t2, write the month's outputs, return its index entry."""
    c1, p1 = forest_probability(aoi, t1, session)
    c2, p2 = forest_probability(aoi, t2, session)
    g1, g2 = grid_of(c1.data), grid_of(c2.data)
    if g1.transform != g2.transform or g1.shape != g2.shape:
        raise ValueError(f"{aoi.id}: {t1} and {t2} composites are on different grids")
    p_next = forest_probability(aoi, t_next, session)[1] if t_next else None
    n1, n2 = composite_ndvi(c1), composite_ndvi(c2)

    def make_features(tolerance_m: float) -> list[dict]:
        return change.features(
            p1, p2, n1, n2, g2.transform, aoi.epsg, t1, t2, aoi.id, p_next, tolerance_m
        )

    path = export.month_dir(aoi.id, t2) / "cleared.geojson"
    feats = export.write_cleared_geojson(make_features, path)
    if rasters:
        export_month_rasters(aoi, t1, c1, p1)
        export_month_rasters(aoi, t2, c2, p2)
    entry = {
        "month": t2,
        "t1": t1,
        "hectares": round(sum(f["properties"]["area_ha"] for f in feats), 2),
        "patches": len(feats),
    }
    export.update_index(aoi.id, entry)
    export.update_aois()
    return entry


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", required=True, choices=sorted(cfg.AOIS))
    ap.add_argument("--t1", help="before month YYYY-MM (default: from the AOI's compare mode)")
    ap.add_argument("--t2", required=True, help="after month YYYY-MM")
    ap.add_argument("--no-rasters", action="store_true", help="skip forest COG and RGB tiles")
    args = ap.parse_args(argv)
    aoi = cfg.get_aoi(args.aoi)
    available = sorted(p.parent.name for p in (cfg.CACHE_DIR / aoi.id).glob("*/composite.tif"))
    t1 = args.t1 or change.reference_month(args.t2, aoi.compare, available)
    if not cache_path_for(aoi.id, t1).exists() and not args.t1:
        print(f"note: fetching {aoi.id} {t1} (not cached)")
    t0 = time.perf_counter()
    entry = process_pair(aoi, t1, args.t2, infer.load_session(), rasters=not args.no_rasters)
    print(f"{aoi.id} {t1} -> {args.t2}: {entry} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
