"""ESA WorldCover 2021 v200 10 m labels aggregated to the 90 m grid.

WorldCover tiles are 3x3 deg EPSG:4326 at 10 m (1/12000 deg). Our grid is
1/1200 deg = exactly 10 WorldCover pixels, and every AOI bbox is a multiple
of 0.05 deg, so a 90 m cell is exactly a 10x10 block of WorldCover pixels
with integer window offsets — pure reshape aggregation (skill technique).

An AOI can straddle two tiles (papum-pare crosses lat 27), so per-class
counts are accumulated ACROSS tiles before labels/purity are derived.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import rasterio
from rasterio.windows import Window

from lib.aois import AOI, CLASSES, LABEL_MIN_PURITY, LABEL_UNMAPPED, WC_EXCLUDED, WC_MAP
from lib.stac_io import DERIVED_DIR, grid_shape

WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com"
          "/v200/2021/map/ESA_WorldCover_10m_2021_v200_"
          "{ns}{lat:02d}{ew}{lon:03d}_Map.tif")
WC_PX = 1.0 / 12000.0

GDAL_ENV = {
    "GDAL_HTTP_MAX_RETRY": "5",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "8388608",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


def wc_tile_url(lat_ll: int, lon_ll: int) -> str:
    return WC_URL.format(
        ns="N" if lat_ll >= 0 else "S", lat=abs(lat_ll),
        ew="E" if lon_ll >= 0 else "W", lon=abs(lon_ll))


def tiles_for_aoi(aoi: AOI):
    """All 3-deg tiles whose [lat, lat+3) x [lon, lon+3) intersects the bbox."""
    w, s, e, n = aoi.bbox
    tiles = []
    la = int(np.floor(s / 3.0)) * 3
    while la < n:
        lo = int(np.floor(w / 3.0)) * 3
        while lo < e:
            tiles.append((la, lo))
            lo += 3
        la += 3
    return tiles


def _read_tile_raw(url: str, wi) -> np.ndarray:
    """Read one WorldCover window (boundless, fill 0). Seam for tests."""
    with rasterio.open(url) as src:
        return src.read(1, window=wi, boundless=True, fill_value=0)


def build_labels(aoi: AOI) -> Dict[str, np.ndarray]:
    """Per-cell dominant class, purity and mapped flag on the 90 m grid.

    Returns {"labels": int16 [rows, cols], "purity": float32, "mapped": bool}.
    labels = class index where purity >= LABEL_MIN_PURITY else LABEL_UNMAPPED.
    """
    rows, cols = grid_shape(aoi)
    w, s, e, n = aoi.bbox
    counts = np.zeros((rows, cols, len(CLASSES)), dtype="int32")
    excluded_counts = np.zeros((rows, cols), dtype="int32")

    with rasterio.env.Env(**GDAL_ENV):
        for la, lo in tiles_for_aoi(aoi):
            url = wc_tile_url(la, lo)
            try:
                # integer window by construction (bbox = 0.05-deg multiples)
                col_off = (w - lo) / WC_PX
                row_off = ((la + 3) - n) / WC_PX
                win_w = (e - w) / WC_PX
                win_h = (n - s) / WC_PX
                wi = Window(int(round(col_off)), int(round(row_off)),
                            int(round(win_w)), int(round(win_h)))
                raw = _read_tile_raw(url, wi)
            except Exception as exc:  # noqa: BLE001
                print(f"  [labels {aoi.id}] tile {la}/{lo} failed: {exc}")
                continue

            # Valid region = tile bbox ∩ AOI bbox, in AOI grid indices.
            # (boundless reads pad with fill=0 outside the tile footprint)
            res = 1.0 / 1200.0
            row_lo = int(round((n - min(n, la + 3)) / res))
            row_hi = int(round((n - max(s, la)) / res))
            col_lo = int(round((max(w, lo) - w) / res))
            col_hi = int(round((min(e, lo + 3) - w) / res))
            if row_hi <= row_lo or col_hi <= col_lo:
                continue
            row_lo, row_hi = max(row_lo, 0), min(row_hi, rows)
            col_lo, col_hi = max(col_lo, 0), min(col_hi, cols)
            if row_hi <= row_lo or col_hi <= col_lo:
                continue

            sub = raw[row_lo * 10: row_hi * 10, col_lo * 10: col_hi * 10]
            if sub.shape != ((row_hi - row_lo) * 10, (col_hi - col_lo) * 10):
                continue  # window clipped at raster edge; skip partial tile
            blocks = sub.reshape(row_hi - row_lo, 10, col_hi - col_lo, 10)
            for code, cls in WC_MAP.items():
                counts[row_lo:row_hi, col_lo:col_hi, cls] += \
                    (blocks == code).sum(axis=(1, 3))
            excluded_counts[row_lo:row_hi, col_lo:col_hi] += \
                np.isin(blocks, list(WC_EXCLUDED)).sum(axis=(1, 3))

    mapped_px = counts.sum(axis=2)
    total_px = mapped_px + excluded_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        dom_cls = np.where(mapped_px > 0, counts.argmax(axis=2),
                           LABEL_UNMAPPED)
        dom_count = counts.max(axis=2)
        purity = np.where(total_px > 0, dom_count / np.maximum(total_px, 1), 0.0)

    labels = np.where(
        (mapped_px > 0) & (purity >= LABEL_MIN_PURITY),
        dom_cls, LABEL_UNMAPPED).astype("int16")
    mapped = mapped_px > 0

    out = {"labels": labels,
           "purity": purity.astype("float32"),
           "mapped": mapped}
    np.savez_compressed(
        os.path.join(DERIVED_DIR, f"{aoi.id}_labels.npz"),
        labels=labels, purity=purity, mapped=mapped)
    return out
