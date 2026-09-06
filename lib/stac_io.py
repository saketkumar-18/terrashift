"""STAC search + windowed COG reads for Sentinel-2 L2A via Earth Search.

Approach (per remote-sensing skill):
- Search Earth Search STAC v1 for scenes covering the AOI per epoch.
- Read ONLY the AOI window from each scene's COG overviews using
  WarpedVRT reprojecting to the shared EPSG:4326 ~90 m grid in one pass.
- Mask each scene with its SCL cloud mask before compositing.
- Disk-cache every scene window (npz); reruns are then seconds.
- Median-composite the epoch; keep per-pixel observation counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from lib.aois import AOI, EPOCHS, GRID_RES

STAC_URL = "https://earth-search.aws.element84.com/v1"

# SCL classes allowed into composites:
# 2 dark-area (coal/burnt surfaces and some shadows - medians absorb them),
# 4 vegetation, 5 not-vegetated, 6 water, 7 unclassified.
# Excluded: 0 nodata, 1 saturated, 3 cloud-shadow, 8/9 clouds, 10 cirrus, 11 snow.
SCL_GOOD = (2, 4, 5, 6, 7)

BANDS = ("green", "red", "nir", "swir16", "scl")

CACHE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cache"))
DERIVED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "derived"))


def ensure_dirs() -> None:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    os.makedirs(DERIVED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared grid geometry
# ---------------------------------------------------------------------------
def grid_shape(aoi: AOI) -> Tuple[int, int]:
    """(rows, cols) of the AOI on the 1/1200 deg grid."""
    w, s, e, n = aoi.bbox
    cols = int(round((e - w) / GRID_RES))
    rows = int(round((n - s) / GRID_RES))
    return rows, cols


def grid_transform(aoi: AOI):
    from rasterio.transform import from_origin
    w, s, e, n = aoi.bbox
    return from_origin(w, n, GRID_RES, GRID_RES)


def _gdal_env() -> dict:
    return {
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "8388608",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    }


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------
def _http_json(method: str, url: str, payload: Optional[dict] = None,
               timeout: int = 90, retries: int = 4):
    import requests

    last = None
    for i in range(retries):
        try:
            if method == "POST":
                r = requests.post(url, json=payload, timeout=timeout,
                                  headers={"User-Agent": "terrashift/1.0"})
            else:
                r = requests.get(url, timeout=timeout,
                                 headers={"User-Agent": "terrashift/1.0"})
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"HTTP {method} {url} failed after {retries} tries: {last}")


def search_items(aoi: AOI, epoch_key: str, max_items: int = 12) -> List[dict]:
    """Scenes covering the AOI in the epoch window, cloud < 30%, newest first."""
    ep = EPOCHS[epoch_key]
    body = {
        "collections": [ep.collection],
        "bbox": list(aoi.bbox),
        "datetime": f"{ep.start}/{ep.end}",
        "query": {"eo:cloud_cover": {"lt": 30}},
        "limit": 200,
    }
    js = _http_json("POST", f"{STAC_URL}/search", payload=body)
    feats = js.get("features", [])
    # lowest cloud cover first - we only keep max_items scenes
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    return feats[:max_items]


# ---------------------------------------------------------------------------
# Scene window read + cache
# ---------------------------------------------------------------------------
def cache_path(aoi_id: str, item_id: str, collection: str = "") -> str:
    """Cache key includes the collection (radiometry differs between the
    e84-legacy and c1 collections) so a radiometry fix invalidates stale
    entries automatically."""
    coll_tag = "c1" if collection.endswith("-c1-l2a") else "e84"
    h = hashlib.md5(f"{aoi_id}:{item_id}:{coll_tag}".encode()).hexdigest()[:10]
    return os.path.join(CACHE_ROOT, f"{aoi_id}__{item_id}__{coll_tag}_{h}.npz")


def _band_specs(item: dict) -> Optional[Dict[str, dict]]:
    """href + nodata/scale/offset per band, parsed from STAC asset metadata.

    RADIOMETRY FIX (validated by QA on raw DN): element84's legacy
    `sentinel-2-l2a` COGs store UNshifted DN while advertising the
    processing-baseline offset (-0.1) - applying it makes reflectance ~0.05
    too low. The official `sentinel-2-c1-l2a` COGs store shifted DN
    (AOI-wide DN tail ~+870 vs legacy at the same land) and the offset
    IS correct there. So: apply the advertised offset only for c1 items.
    """
    assets = item.get("assets", {})
    coll = (item.get("collection") or "")
    is_c1 = coll.endswith("-c1-l2a")
    out: Dict[str, dict] = {}
    for band in BANDS:
        a = assets.get(band)
        if not a or "href" not in a:
            return None
        rb = {}
        rbs = a.get("raster:bands")
        if isinstance(rbs, list) and rbs:
            rb = rbs[0] or {}
        out[band] = {
            "href": a["href"],
            "nodata": float(rb.get("nodata", 0)),
            "scale": float(rb.get("scale", 0.0001)),
            "offset": float(rb.get("offset", 0.0)) if is_c1 else 0.0,
        }
    return out


def read_scene_window(aoi: AOI, item: dict) -> Optional[Dict[str, np.ndarray]]:
    """AOI window for one scene on the shared grid.

    Returns {"green":..., "red":..., "nir":..., "swir16":..., "good": mask}
    with bad/out-of-footprint pixels as NaN, or None if unusable.
    """
    specs = _band_specs(item)
    if not specs:
        return None
    rows, cols = grid_shape(aoi)
    dst_tr = grid_transform(aoi)

    out: Dict[str, np.ndarray] = {}
    with rasterio.env.Env(**_gdal_env()):
        # SCL (nearest) first - geometry/coverage check
        s = specs["scl"]
        try:
            with rasterio.open(s["href"]) as src:
                with WarpedVRT(src, crs="EPSG:4326", transform=dst_tr,
                               width=cols, height=rows,
                               resampling=Resampling.nearest) as vrt:
                    scl = vrt.read(1).astype("int16")
        except Exception:
            return None
        good = np.isin(scl, SCL_GOOD)
        if float(good.mean()) < 0.10:  # scene barely touches the AOI
            return None

        for band in ("green", "red", "nir", "swir16"):
            s = specs[band]
            try:
                with rasterio.open(s["href"]) as src:
                    with WarpedVRT(src, crs="EPSG:4326", transform=dst_tr,
                                   width=cols, height=rows,
                                   resampling=Resampling.bilinear) as vrt:
                        dn = vrt.read(1).astype("float32")
            except Exception:
                return None
            nd = s["nodata"]
            with np.errstate(invalid="ignore"):
                valid = (dn != nd) & (dn > 0)
                refl = np.where(valid, dn * s["scale"] + s["offset"], np.nan)
                refl = np.where((refl < 0) | (refl > 1.6), np.nan, refl)
            out[band] = refl.astype("float32")

        out["good"] = good
        out["scl"] = scl
        return out


def load_or_fetch(aoi: AOI, item: dict) -> Optional[Dict[str, np.ndarray]]:
    """Disk-cached scene window fetch."""
    ensure_dirs()
    p = cache_path(aoi.id, item["id"], item.get("collection", ""))
    if os.path.exists(p):
        try:
            z = np.load(p, allow_pickle=False)
            return {k: z[k] for k in z.files}
        except Exception:
            os.remove(p)
    win = read_scene_window(aoi, item)
    if win is not None:
        np.savez_compressed(p, **win)
    return win


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def build_composite(aoi: AOI, epoch_key: str, min_scenes: int = 3,
                    min_obs: int = 3, max_items: int = 12) -> Optional[dict]:
    """Median composite over usable scenes; None if < min_scenes usable.

    Scene windows are fetched in parallel (3 workers max - slow link).
    Result saved to data/derived/{aoi}_{epoch}.npz with:
      green/red/nir/swir16 : float32 median reflectance (NaN where unusable)
      n_obs                : observations per pixel
      features             : [rows, cols, 8] (bands + ndvi/ndbi/mndwi)
      scene_ids            : usable scene ids (json string)
    """
    ensure_dirs()
    items = search_items(aoi, epoch_key, max_items=max_items)
    if not items:
        print(f"  [{aoi.id}/{epoch_key}] STAC: no scenes matched")
        return None

    import concurrent.futures as cf
    windows: List[Optional[Dict[str, np.ndarray]]] = [None] * len(items)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(load_or_fetch, aoi, it): i
                for i, it in enumerate(items)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                windows[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{aoi.id}/{epoch_key}] fetch error {items[i]['id']}: {exc}")
                windows[i] = None

    rows, cols = grid_shape(aoi)
    stack = {b: [] for b in ("green", "red", "nir", "swir16")}
    good_stack: List[np.ndarray] = []
    scene_ids: List[str] = []
    for it, win in zip(items, windows):
        if win is None:
            print(f"  [{aoi.id}/{epoch_key}] skip {it['id']} (unusable)")
            continue
        scene_ids.append(it["id"])
        for b in stack:
            arr = win[b].copy()
            arr[~win["good"]] = np.nan
            stack[b].append(arr)
        good_stack.append(win["good"])
        if len(scene_ids) >= max_items:
            break
    if len(scene_ids) < min_scenes:
        print(f"  [{aoi.id}/{epoch_key}] only {len(scene_ids)} usable scenes"
              f" (< {min_scenes})")
        return None

    comp: Dict[str, np.ndarray] = {}
    obs = np.zeros((rows, cols), dtype="int16")
    for gmask in good_stack:
        obs += gmask.astype("int16")
    for b in stack:
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(np.stack(stack[b]), axis=0)
        comp[b] = med.astype("float32")

    usable = obs >= min_obs
    for b in comp:
        comp[b][~usable] = np.nan

    feats = build_features(comp["green"], comp["red"], comp["nir"],
                            comp["swir16"])
    out_path = os.path.join(DERIVED_DIR, f"{aoi.id}_{epoch_key}.npz")
    np.savez_compressed(
        out_path,
        green=comp["green"], red=comp["red"], nir=comp["nir"],
        swir16=comp["swir16"], n_obs=obs, features=feats,
        scene_ids=np.array(json.dumps(scene_ids)),
    )
    print(f"  [{aoi.id}/{epoch_key}] composite ok: {len(scene_ids)} scenes,"
          f" usable px={int(usable.sum())}/{rows*cols}")
    return {"path": out_path, "n_scenes": len(scene_ids),
            "usable_frac": float(usable.mean())}


def build_features(green: np.ndarray, red: np.ndarray, nir: np.ndarray,
                   swir16: np.ndarray) -> np.ndarray:
    """[rows, cols, 8]: green, red, nir, swir16, ndvi, ndbi, mndwi, brightness."""
    def ratio(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (a - b) / (a + b)
        return np.clip(r, -1.0, 1.0)

    ndvi = ratio(nir, red)
    ndbi = ratio(swir16, nir)
    mndwi = ratio(green, swir16)
    brightness = (green + red + nir) / 3.0
    feats = np.stack([green, red, nir, swir16, ndvi, ndbi, mndwi,
                      brightness], axis=-1).astype("float32")
    return feats


FEATURE_NAMES = ["green", "red", "nir", "swir16",
                 "ndvi", "ndbi", "mndwi", "brightness"]
