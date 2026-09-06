"""AOI, epoch and class definitions for TerraShift.

Three study areas, each chosen for a well-documented, visually verifiable
change signal between 2019 and 2026:

- ncr-gurugram : Delhi NCR urban expansion along the Dwarka Expressway /
  Manesar industrial corridor (urban growth).
- papum-pare   : Arunachal Pradesh, Papum Pare district around Itanagar
  (documented forest loss / shifting cultivation + construction).
- jharia       : Jharkhand, Jharia coalfield (open-cast mining expansion
  and subsidence-driven land change).

Grid convention
---------------
Every AOI is analysed on ONE fixed EPSG:4326 grid at 1/1200 deg (~90 m N-S).
That resolution is exactly 10x the ESA WorldCover 10 m pixel (1/12000 deg),
so label aggregation is a pure integer reshape, and every AOI bbox is a
multiple of 0.05 deg so windows align exactly on source rasters.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
GRID_RES = 1.0 / 1200.0  # ~90 m N-S, exactly 10x WorldCover pixel
PX_AREA_HA_NOMINAL = 0.81  # 90 m x 90 m nominal; true areas are geodesic

# ---------------------------------------------------------------------------
# Epochs (dry season composites; NE India is cloudiest in monsoon)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Epoch:
    key: str
    label: str
    start: str
    end: str
    collection: str  # earth-search collection id


EPOCHS: Dict[str, Epoch] = {
    "2019": Epoch(
        "2019", "2019 dry season (Jan-Mar)",
        "2019-01-01T00:00:00Z", "2019-03-31T23:59:59Z",
        "sentinel-2-l2a",
    ),
    "2021": Epoch(
        "2021", "2021 dry season (Jan-Mar)",
        "2021-01-01T00:00:00Z", "2021-03-31T23:59:59Z",
        "sentinel-2-l2a",
    ),
    "2026": Epoch(
        "2026", "2026 dry season (Jan-Mar)",
        "2026-01-01T00:00:00Z", "2026-03-31T23:59:59Z",
        "sentinel-2-c1-l2a",
    ),
}

# E0 = first epoch compared, E1 = training epoch (matches WorldCover 2021
# label year), E2 = latest epoch compared.
EPOCH_TRAIN = "2021"
EPOCH_BEFORE = "2019"
EPOCH_AFTER = "2026"


# ---------------------------------------------------------------------------
# Areas of interest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AOI:
    id: str
    name: str
    focus: str
    bbox: Tuple[float, float, float, float]  # west, south, east, north
    story: str


AOIS: List[AOI] = [
    AOI(
        "ncr-gurugram",
        "NCR - Gurugram-Manesar corridor",
        "urban-growth",
        (76.85, 28.15, 77.25, 28.55),
        "Delhi NCR expansion belt between Dwarka Expressway and the Manesar "
        "industrial corridor; built-up land has grown rapidly since 2019.",
    ),
    AOI(
        "papum-pare",
        "Arunachal Pradesh - Papum Pare (Itanagar)",
        "forest-loss",
        (93.60, 26.85, 94.00, 27.25),
        "Foothill belt around Itanagar; documented forest loss from "
        "shifting cultivation, settlement growth and road building.",
    ),
    AOI(
        "jharia",
        "Jharkhand - Jharia coalfield",
        "mining",
        (86.20, 23.65, 86.50, 23.95),
        "One of India's largest coalfields; open-cast pit expansion and "
        "overburden dumps enlarge bare land at the expense of vegetation "
        "and settlements.",
    ),
]

AOI_BY_ID = {a.id: a for a in AOIS}


# ---------------------------------------------------------------------------
# Land-cover classes (RF target). WorldCover v200 codes are mapped onto
# these; ambiguous codes (shrub, wetland, mangrove, moss, snow) are excluded
# from training but still inferenceable.
# ---------------------------------------------------------------------------
CLASSES: List[str] = ["forest", "grass_agri", "bare", "built", "water"]
N_CLASSES = len(CLASSES)

# WorldCover v200 code -> class index
WC_MAP: Dict[int, int] = {
    10: 0,   # tree cover -> forest
    30: 1,   # grassland -> grass_agri
    40: 1,   # cropland  -> grass_agri
    60: 2,   # bare / sparse vegetation
    50: 3,   # built-up
    80: 4,   # permanent water
}
# Codes deliberately excluded from training labels (ambiguous at 90 m or
# rare in the AOIs): shrub(20), snow(70), wetland(90), mangrove(95),
# moss/lichen(100), nodata(0).
WC_EXCLUDED = {0, 20, 70, 90, 95, 100}
LABEL_UNMAPPED = -1

# Minimum purity (fraction of 10 m pixels inside a 90 m cell matching the
# dominant class) required for a training label.
LABEL_MIN_PURITY = 0.70

# ---------------------------------------------------------------------------
# Change products. Disjoint by (from_class, to_class) pair so areas never
# double-count. Every product carries a display colour used by the web map.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Product:
    key: str
    title: str
    from_classes: frozenset
    to_classes: frozenset
    color: str
    description: str


PRODUCTS: Dict[str, Product] = {
    "urban_growth": Product(
        "urban_growth", "Urban growth",
        frozenset({1, 2}), frozenset({3}), "#ff6b35",
        "Agriculture/bare land converted to built-up.",
    ),
    "forest_loss": Product(
        "forest_loss", "Forest loss",
        frozenset({0}), frozenset({1, 2, 3}), "#e63946",
        "Tree cover replaced by fields, bare land or construction.",
    ),
    "bare_expansion": Product(
        "bare_expansion", "Bare / mining expansion",
        frozenset({1}), frozenset({2}), "#ffd166",
        "Vegetated or agricultural land stripped to bare soil (mining, "
        "overburden dumps, excavation).",
    ),
    "new_water": Product(
        "new_water", "New water",
        frozenset({0, 1, 2, 3}), frozenset({4}), "#4cc9f0",
        "Land became permanent water (reservoirs, subsidence pools).",
    ),
    "water_loss": Product(
        "water_loss", "Water loss",
        frozenset({4}), frozenset({0, 1, 2, 3}), "#b5179e",
        "Water bodies replaced by land (drying, infilling, mining cut).",
    ),
}

# Minimum geodesic patch area kept in the published GeoJSON.
MIN_PATCH_HA = 6.0
