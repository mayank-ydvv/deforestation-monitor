"""Single source of truth for AOIs, bands, thresholds and storage paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# --- Data access -----------------------------------------------------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_LABEL_YEAR = 2021  # v200; training composites must come from this year
WORLDCOVER_TREE_CLASS = 10

# --- Bands -----------------------------------------------------------------------------
BANDS_10M: tuple[str, ...] = ("B02", "B03", "B04", "B08")
BANDS_20M: tuple[str, ...] = ("B11", "B12")  # resampled to 10 m with bilinear
BANDS: tuple[str, ...] = BANDS_10M + BANDS_20M
SCL_ASSET = "SCL"
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud, 10 thin cirrus.
SCL_MASK_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10)
CHANNELS: tuple[str, ...] = BANDS + ("NDVI",)  # 7 model input channels

# --- Radiometry ------------------------------------------------------------------------
REFLECTANCE_SCALE = 10_000.0
# Processing baseline >= 04.00 (from 2022-01-25) adds +1000 to every L2A DN.
BOA_ADD_OFFSET = 1_000.0
OFFSET_BASELINE = 4.0
NDVI_EPS = 1e-6

# --- Compositing -----------------------------------------------------------------------
RESOLUTION_M = 10
MAX_SCENE_CLOUD = 20  # eo:cloud_cover upper bound, percent
MAX_SCENES = 6  # least-cloudy scenes per month; caps bytes read, plenty for a median
FETCH_ATTEMPTS = 3  # whole-composite retries after a dropped or stalled HTTP read
MIN_CLEAR_FRACTION = 0.6  # skip the month below this
DASK_CHUNK = 2048

# --- Model / change detection ----------------------------------------------------------
TILE_SIZE = 256
TILE_OVERLAP = 32
MAX_TILE_NAN = 0.10
INFER_BATCH = 16
INFER_CHUNK = 4096
PROB_THR = 0.5
NDVI_DROP = 0.2
MIN_AREA_PX = 50  # 50 px x 100 m^2 = 0.5 ha
SIMPLIFY_M = 5.0

# --- Storage ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
TILES_DIR = ROOT / "tiles"
LABEL_OVERRIDES_DIR = ROOT / "labels" / "overrides"  # hand-corrected <aoi>.tif (0/1, 255=keep)
REPORTS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"
WEB_DATA_DIR = ROOT / "web" / "public" / "data"
NORM_STATS_PATH = MODELS_DIR / "norm_stats.json"
MODEL_PATH = MODELS_DIR / "unet_r34.onnx"

# --- Areas of interest -----------------------------------------------------------------
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class AOI:
    """An area of interest.

    ``bbox`` is (min_lon, min_lat, max_lon, max_lat); ``epsg`` is the local UTM zone;
    ``months_skip`` lists calendar months (1-12) that are too cloudy to be worth fetching;
    ``split`` is the region-level train/val/test assignment; ``demo`` AOIs appear on the site;
    ``label_months`` are clear months in the WorldCover label year used to build training tiles;
    ``compare`` is how change detection picks the "before" month: ``yoy`` (same month a year
    earlier, for seasonal/deciduous forest) or ``consecutive`` (latest earlier usable month).
    """

    id: str
    name: str
    bbox: tuple[float, float, float, float]
    epsg: int
    months_skip: tuple[int, ...] = ()
    split: Split = "train"
    demo: bool = False
    label_months: tuple[str, ...] = ()
    compare: Literal["consecutive", "yoy"] = "consecutive"

    @property
    def center(self) -> tuple[float, float]:
        """(lat, lon) centre of the bbox, as Leaflet expects."""
        w, s, e, n = self.bbox
        return ((s + n) / 2, (w + e) / 2)


AOIS: dict[str, AOI] = {
    a.id: a
    for a in (
        AOI(
            id="hasdeo",
            name="Hasdeo Arand, Chhattisgarh (India)",
            bbox=(82.66, 22.71, 82.96, 22.97),  # centred on PEKB mine (22.838 N, 82.814 E)
            epsg=32644,
            months_skip=(6, 7, 8, 9),
            split="test",
            label_months=("2021-01", "2021-04"),
            compare="yoy",
            demo=True,
        ),
        AOI(
            id="rondonia",
            name="Jaci-Paraná, Rondônia (Brazil)",
            bbox=(-64.40, -9.60, -64.00, -9.24),
            epsg=32720,
            months_skip=(12, 1, 2, 3),
            split="train",
            label_months=("2021-07", "2021-08"),
            demo=True,
        ),
        AOI(
            id="mato_grosso",
            name="Northern Mato Grosso (Brazil)",
            bbox=(-55.60, -11.90, -55.30, -11.60),
            epsg=32721,
            months_skip=(12, 1, 2, 3),
            split="train",
            label_months=("2021-06", "2021-08"),
        ),
        AOI(
            id="prey_lang",
            name="Prey Lang, Cambodia",
            bbox=(105.30, 13.20, 105.60, 13.50),
            epsg=32648,
            months_skip=(6, 7, 8, 9, 10),
            split="train",
            label_months=("2021-03", "2021-12"),
        ),
        AOI(
            id="kanha",
            name="Kanha, Madhya Pradesh (India)",
            bbox=(80.55, 22.15, 80.85, 22.45),  # dry deciduous sal, like Hasdeo
            epsg=32644,
            months_skip=(6, 7, 8, 9),
            split="train",
            label_months=("2021-01", "2021-04"),  # leaf-on and leaf-off, matching Hasdeo
            compare="yoy",
        ),
        AOI(
            id="western_ghats",
            name="Western Ghats, Karnataka (India)",
            bbox=(75.15, 13.10, 75.45, 13.40),
            epsg=32643,
            months_skip=(6, 7, 8, 9),
            split="val",
            label_months=("2021-01", "2021-02"),
            compare="yoy",
        ),
    )
}


def get_aoi(aoi_id: str) -> AOI:
    """Look up an AOI by id, with a helpful error for typos."""
    try:
        return AOIS[aoi_id]
    except KeyError:
        raise KeyError(f"Unknown AOI {aoi_id!r}; choose from {sorted(AOIS)}") from None
