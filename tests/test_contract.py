"""Exported files validate against docs/data-contract.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from jsonschema import Draft202012Validator
from PIL import Image

from pipeline import change, export
from pipeline import config as cfg

SCHEMA = json.loads((cfg.ROOT / "docs" / "data-contract.json").read_text())
UTM_TRANSFORM = Affine(10, 0, 700_000, 0, -10, 2_520_000)  # UTM 44N near Hasdeo


def validate(name: str, data) -> None:
    """Validate ``data`` against the contract's ``$defs/<name>``."""
    Draft202012Validator({**SCHEMA, "$ref": f"#/$defs/{name}"}).validate(data)


def _synthetic_features(tolerance_m: float = cfg.SIMPLIFY_M) -> list[dict]:
    p1 = np.full((120, 120), 0.9, dtype="float32")
    n1 = np.full((120, 120), 0.8, dtype="float32")
    p2, n2 = p1.copy(), n1.copy()
    for r0, c0 in [(10, 10), (60, 70)]:
        p2[r0 : r0 + 20, c0 : c0 + 25] = 0.1
        n2[r0 : r0 + 20, c0 : c0 + 25] = 0.3
    return change.features(
        p1, p2, n1, n2, UTM_TRANSFORM, 32644, "2025-01", "2026-01", "hasdeo", None, tolerance_m
    )


def test_schema_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


def test_cleared_geojson_index_and_aois(tmp_path):
    path = export.month_dir("hasdeo", "2026-01", tmp_path) / "cleared.geojson"
    feats = export.write_cleared_geojson(_synthetic_features, path)
    data = json.loads(path.read_text())
    validate("cleared", data)
    assert len(data["features"]) == len(feats) == 2
    coords = np.concatenate([np.array(f["geometry"]["coordinates"][0]) for f in feats])
    assert (np.abs(coords[:, 0]) <= 180).all() and (np.abs(coords[:, 1]) <= 90).all()

    entry = {"month": "2026-01", "t1": "2025-01", "hectares": 10.0, "patches": 2}
    rows = export.update_index("hasdeo", entry, root=tmp_path)
    export.update_index("hasdeo", {**entry, "hectares": 9.5}, root=tmp_path)  # replaces
    rows = json.loads((tmp_path / "hasdeo" / "index.json").read_text())
    validate("index", rows)
    assert len(rows) == 1 and rows[0]["hectares"] == 9.5

    aois = export.update_aois(root=tmp_path, raster_base="https://example.org/data/")
    validate("aois", aois)
    hasdeo = next(a for a in aois if a["id"] == "hasdeo")
    assert hasdeo["months"] == ["2025-01", "2026-01"] and hasdeo["compare"] == "yoy"
    assert {a["id"] for a in aois} == {a.id for a in cfg.AOIS.values() if a.demo}


def test_geojson_is_resimplified_to_fit(tmp_path):
    calls = []

    def bulky(tolerance_m: float) -> list[dict]:
        calls.append(tolerance_m)
        n = 2000 if tolerance_m < 20 else 10  # pretend coarser geometry is much smaller
        return [_synthetic_features()[0]] * n

    export.write_cleared_geojson(bulky, tmp_path / "c.geojson", max_bytes=100_000)
    assert calls == [5.0, 10.0, 20.0]
    assert (tmp_path / "c.geojson").stat().st_size <= 100_000


def test_forest_cog(tmp_path):
    prob = np.full((600, 700), 0.8, dtype="float16")
    prob[:100] = 0.2
    prob[-5:, -5:] = np.nan
    path = tmp_path / "forest.tif"
    export.write_forest_cog(prob, UTM_TRANSFORM, 32644, path)
    with rasterio.open(path) as src:
        data = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata == 255
        assert src.profile["tiled"] and src.crs.to_epsg() == 32644
    assert set(np.unique(data)) == {0, 1, 255}
    assert data[0, 0] == 0 and data[300, 300] == 1 and data[-1, -1] == 255


def test_xyz_tiles(tmp_path):
    rgba = np.zeros((4, 300, 300), dtype="uint8")
    rgba[0], rgba[3] = 200, 255
    n = export.write_xyz_tiles(rgba, UTM_TRANSFORM, 32644, tmp_path, zooms=[10, 13])
    pngs = sorted(tmp_path.rglob("*.png"))
    assert n == len(pngs) > 0
    z, x, y = Path(pngs[0].relative_to(tmp_path)).with_suffix("").parts
    assert z in {"10", "13"} and x.isdigit() and y.isdigit()
    img = Image.open(pngs[-1])
    assert img.size == (256, 256) and img.mode == "RGBA"


def test_tile_math():
    assert export.tile_range((-1, -1, 1, 1), 0) == (range(0, 1), range(0, 1))
    xs, ys = export.tile_range((82.66, 22.71, 82.96, 22.97), 14)
    assert 0 < len(xs) < 30 and 0 < len(ys) < 30
    t = export.tile_transform(0, 0, 0)
    assert t.c == pytest.approx(-export.WEB_MERCATOR_HALF) and t.f == pytest.approx(
        export.WEB_MERCATOR_HALF
    )


@pytest.mark.parametrize(
    ("name", "pattern"),
    [
        ("aois", "aois.json"),
        ("metrics", "metrics.json"),
        ("index", "*/index.json"),
        ("cleared", "*/*/cleared.geojson"),
    ],
)
def test_repo_data_files_follow_contract(name, pattern):
    files = sorted(cfg.WEB_DATA_DIR.glob(pattern))
    if not files:
        pytest.skip(f"no {pattern} generated yet")
    for path in files:
        validate(name, json.loads(path.read_text()))
        if name == "cleared":
            assert path.stat().st_size <= export.MAX_GEOJSON_BYTES


class _FakeHub:
    """huggingface_hub 2.x stand-in: records calls, no network."""

    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw["repo_id"], kw))


def test_upload_rasters_targets_only_rasters(tmp_path):
    hub = _FakeHub()
    export.upload_rasters(hub, "me/data", root=tmp_path)
    assert hub.calls[0][0] == "create_repo" and hub.calls[0][2]["private"] is False
    kw = hub.calls[1][2]
    assert kw["repo_type"] == "dataset" and kw["folder_path"] == tmp_path
    assert kw["allow_patterns"] == ["*/*/forest.tif", "*/*/rgb/**/*.png"]


def test_aois_point_at_raster_store_by_default(tmp_path):
    aois = export.update_aois(root=tmp_path)
    validate("aois", aois)
    assert all(a["raster_base"] == cfg.RASTER_BASE_URL for a in aois)
    assert cfg.RASTER_BASE_URL.startswith("https://") and cfg.RASTER_BASE_URL.endswith("/")
