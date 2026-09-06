"""Change detection: epoch-to-epoch class transitions -> products -> GeoJSON.

Products are disjoint by (from_class, to_class) pair, so areas never
double-count. Only cells observed in both epochs (valid in both) count as
change. Patches < MIN_PATCH_HA are dropped; areas are geodesic.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from pyproj import Geod

from lib.aois import AOI, CLASSES, EPOCH_AFTER, EPOCH_BEFORE, GRID_RES, \
    MIN_PATCH_HA, PRODUCTS
from lib.stac_io import DERIVED_DIR, grid_shape

GEOD = Geod(ellps="WGS84")


def _load(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return z["pred"]


def transition_map_from_arrays(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """int16 codes: from_cls * 5 + to_cls; -1 where either epoch is -1."""
    valid = (p0 != -1) & (p1 != -1)
    tr = np.full(p0.shape, -1, dtype="int16")
    tr[valid] = p0[valid] * 5 + p1[valid]
    return tr


def transition_map(aoi: AOI) -> Optional[np.ndarray]:
    """Transition grid for an AOI from saved epoch predictions."""
    p0 = _load(os.path.join(DERIVED_DIR, f"{aoi.id}_{EPOCH_BEFORE}_pred.npz"))
    p1 = _load(os.path.join(DERIVED_DIR, f"{aoi.id}_{EPOCH_AFTER}_pred.npz"))
    if p0 is None or p1 is None:
        return None
    return transition_map_from_arrays(p0, p1)


def transition_image(aoi: AOI, out_png: str, tr: Optional[np.ndarray] = None) -> bool:
    """RGBA PNG of the transition grid with a legend-ready colour ramp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    if tr is None:
        tr = transition_map(aoi)
    if tr is None:
        return False
    rows, cols = tr.shape

    # class colors: forest dark green, grass yellow-green, bare tan, built red, water blue
    cls_colors = ["#1a7a3c", "#a4c96b", "#d9b26f", "#c94f4f", "#3a7ca5"]
    tr_color = np.zeros((rows, cols, 3), dtype="uint8")
    for from_i, from_c in enumerate(CLASSES):
        for to_i, to_c in enumerate(CLASSES):
            code = from_i * 5 + to_i
            m = tr == code
            if not m.any():
                continue
            if from_i == to_i:
                col = cls_colors[from_i]
            else:
                col = cls_colors[to_i]
            rgb = tuple(int(col[i:i+2], 16) for i in (1, 3, 5))
            tr_color[m] = rgb

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(tr_color)
    ax.set_title(f"{aoi.name}: land-cover change {EPOCH_BEFORE} → {EPOCH_AFTER}")
    ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=cls_colors[i], label=CLASSES[i])
               for i in range(len(CLASSES))]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), os.path) if os.path.dirname(out_png) else None
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def product_mask(tr: np.ndarray, product_key: str) -> np.ndarray:
    """Boolean mask of cells matching the product's transition pairs."""
    prod = PRODUCTS[product_key]
    m = np.zeros(tr.shape, dtype=bool)
    for f in prod.from_classes:
        for t in prod.to_classes:
            m |= (tr == f * 5 + t)
    return m


def patches_geojson(aoi: AOI, product_key: str, tr: np.ndarray,
                    min_ha: float = MIN_PATCH_HA) -> Optional[dict]:
    """Vectorize a product mask into area-filtered GeoJSON polygons.

    Cells are 1/1200 deg; polygons are cell-unions. Geodesic areas via
    pyproj.Geod; patches < min_ha dropped; local crs EPSG:4326.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union as _uu

    m = product_mask(tr, product_key)
    if not m.any():
        return None
    rows, cols = m.shape
    w, s, e, n = aoi.bbox
    # grid transform: origin (w, n), res 1/1200
    polys = []
    rr, cc = np.nonzero(m)
    cells = [box(w + c * GRID_RES, n - (r + 1) * GRID_RES,
                 w + (c + 1) * GRID_RES, n - r * GRID_RES)
             for r, c in zip(rr, cc)]
    merged = _uu(cells)
    if merged.is_empty:
        return None
    geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
    feats = []
    total_ha = 0.0
    for g in geoms:
        if not g.is_valid:
            g = g.buffer(0)
        area_m2, _ = GEOD.geometry_area_perimeter(g)
        ha = abs(area_m2) / 10000.0
        if ha < min_ha:
            continue
        total_ha += ha
        feats.append({
            "type": "Feature",
            "properties": {
                "product": product_key,
                "area_ha": round(ha, 2),
            },
            "geometry": g.__geo_interface__,
        })
    if not feats:
        return None
    return {
        "type": "FeatureCollection",
        "features": feats,
        "properties": {"aoi": aoi.id, "product": product_key,
                       "total_ha": round(total_ha, 1),
                       "n_patches": len(feats)},
    }
