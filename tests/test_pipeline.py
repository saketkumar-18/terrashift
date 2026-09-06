"""Offline unit tests on synthetic rasters - no network, no real data.

Covers: grid geometry, feature math (NDVI/NDBI/MNDWI), SCL cloud masking in
composites, Otsu-free median compositing purity, label aggregation
(purity/multi-tile), model train/predict round-trip on synthetic data,
change transition codes and GeoJSON area filtering.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from lib.aois import AOI, CLASSES, GRID_RES, MIN_PATCH_HA, PRODUCTS  # noqa: E402
from lib import stac_io, change as change_mod, model as model_mod  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synth_aoi():
    return AOI("synth", "Synthetic 0.2deg AOI", "test",
               (10.00, 10.00, 10.20, 10.20), "test AOI")


def test_grid_shape_exact(synth_aoi):
    rows, cols = stac_io.grid_shape(synth_aoi)
    assert rows == cols == int(0.20 / GRID_RES)  # 240 = 0.2*1200
    tr = stac_io.grid_transform(synth_aoi)
    # GDAL corner convention: c/f are the OUTER corner of the grid
    w, s, e, n = synth_aoi.bbox
    assert tr.c == pytest.approx(w)
    assert tr.f == pytest.approx(n)
    assert tr.a == pytest.approx(GRID_RES)
    assert tr.e == pytest.approx(-GRID_RES)
    # pixel centers must stay inside the bbox
    assert tr.c + tr.a / 2 > w and tr.f + tr.e / 2 < n


def test_build_features_math():
    rng = np.random.default_rng(0)
    g, r, nir, swir = (rng.uniform(0.05, 0.4, (5, 5)).astype("float32")
                       for _ in range(4))
    f = stac_io.build_features(g, r, nir, swir)
    assert f.shape == (5, 5, 8)
    assert f.dtype == np.float32
    with np.errstate(all="ignore"):
        assert np.allclose(f[..., 4], np.clip((nir - r) / (nir + r), -1, 1))
        assert np.allclose(f[..., 5], np.clip((swir - nir) / (swir + nir), -1, 1))
        assert np.allclose(f[..., 6], np.clip((g - swir) / (g + swir), -1, 1))


def test_build_features_nan_propagation():
    g = np.full((3, 3), 0.2, "float32")
    r = np.full((3, 3), 0.1, "float32")
    nir = g.copy(); nir[1, 1] = np.nan
    swir = np.full((3, 3), 0.3, "float32")
    f = stac_io.build_features(g, r, nir, swir)
    # raw green/red/swir stay finite; nir + nir-derived ratios go NaN
    # (mndwi uses green+swir only - no nir - so stays finite)
    assert np.isfinite(f[1, 1, 0]) and np.isfinite(f[1, 1, 1])
    assert np.isnan(f[1, 1, 2])          # nir
    assert np.isnan(f[1, 1, 4]).all() is True or np.isnan(f[1, 1, 4])  # ndvi
    assert np.isnan(f[1, 1, 5])          # ndbi
    assert np.isfinite(f[1, 1, 6])      # mndwi (no nir dependency)
    assert np.isnan(f[1, 1, 7])          # brightness (nir avg)
    assert np.isfinite(f[0, 0]).all()


# ---------------------------------------------------------------------------
# compositing with SCL masking
# ---------------------------------------------------------------------------
def _make_scene(rows, cols, base, scl, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "green": base + rng.normal(0, 0.01, (rows, cols)).astype("float32"),
        "red":   base + rng.normal(0, 0.01, (rows, cols)).astype("float32"),
        "nir":   base + rng.normal(0, 0.01, (rows, cols)).astype("float32"),
        "swir16": base + rng.normal(0, 0.01, (rows, cols)).astype("float32"),
        "scl": scl,
        "good": np.isin(scl, stac_io.SCL_GOOD),
    }


def test_composite_masks_clouds(monkeypatch, synth_aoi, tmp_path):
    rows, cols = stac_io.grid_shape(synth_aoi)
    scl_clear = np.full((rows, cols), 4, dtype="int16")
    scl_cloudy = scl_clear.copy()
    scl_cloudy[: rows // 2, :] = 8  # top half cloud
    s1 = _make_scene(rows, cols, 0.30, scl_clear, seed=1)
    s2 = _make_scene(rows, cols, 0.50, scl_cloudy, seed=2)

    calls = {"n": 0}

    def fake_load_or_fetch(aoi, item):
        calls["n"] += 1
        return [s1, s2][calls["n"] - 1]

    def fake_search(aoi, ek, max_items=12):
        return [{"id": f"scene{i}"} for i in range(2)]

    monkeypatch.setattr(stac_io, "load_or_fetch", fake_load_or_fetch)
    monkeypatch.setattr(stac_io, "search_items", fake_search)
    monkeypatch.setattr(stac_io, "DERIVED_DIR", str(tmp_path))

    res = stac_io.build_composite(synth_aoi, "2026", min_scenes=2, min_obs=2)
    assert res is not None
    z = np.load(res["path"])
    # top half: 1 clear obs (s1) + 1 clouded obs (s2) -> obs count 1 < 2
    # because clouded pixels are NaN'd by the good-mask before median.
    assert np.isnan(z["green"][: rows // 2, :]).all()
    # bottom half: 2 clear obs -> median of 0.30/0.50 ≈ 0.40 (noise σ=0.01)
    assert np.allclose(z["green"][rows // 2:, :], 0.40, atol=0.06)


# ---------------------------------------------------------------------------
# labels aggregation
# ---------------------------------------------------------------------------
def test_label_purity_threshold(synth_aoi, tmp_path, monkeypatch):
    from lib import labels as labels_mod

    rows, cols = stac_io.grid_shape(synth_aoi)
    w, s, e, n = synth_aoi.bbox
    lo, la = 9, 9  # tile covering lat [9,12), lon [9,12)

    # Full-size virtual tile: 3 deg at 10 m = 36000x36000. We never
    # materialize it - _read_tile_raw is faked to return just the window.
    WC_PX = 1.0 / 12000.0
    win_cols = int(round((e - w) / WC_PX))  # 2400
    win_rows = win_cols

    def fake_read(url, wi):
        # window col_off=12000 row_off=21600 w=h=2400 for this AOI
        assert (wi.col_off, wi.row_off) == (12000, 21600)
        raw = np.full((wi.height, wi.width), 50, dtype="uint8")
        # window-local px coords = grid-local * 10
        raw[100:110, 100:110] = 10   # forest block -> cell (10,10)
        raw[120:130, 120:130] = 30   # cropland -> cell (12,12)
        raw[140:150, 140:150] = 50   # built base
        raw[140:143, 140:143] = 30   # 9 cropland px -> 91% built cell (14,14)
        return raw

    monkeypatch.setattr(labels_mod, "wc_tile_url", lambda a, b: "unused")
    monkeypatch.setattr(labels_mod, "_read_tile_raw", fake_read)
    monkeypatch.setattr(labels_mod, "DERIVED_DIR", str(tmp_path))

    out = labels_mod.build_labels(synth_aoi)
    labs = out["labels"]
    assert labs[10, 10] == 0   # pure forest
    assert labs[12, 12] == 1   # pure cropland
    assert labs[14, 14] == 3   # 91% built -> built
    assert labs[0, 0] == 3     # background built-up
    # purity values sanity
    assert out["purity"][10, 10] == pytest.approx(1.0)
    assert out["purity"][14, 14] == pytest.approx(0.91, abs=0.02)


# ---------------------------------------------------------------------------
# model round-trip
# ---------------------------------------------------------------------------
def test_model_train_predict_roundtrip(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    # 5 well-separated class blobs in 8-D feature space
    centers = rng.uniform(0.05, 0.6, (5, 8))
    n = 400
    X = np.concatenate([
        centers[c] + rng.normal(0, 0.02, (n, 8)) for c in range(5)])
    y = np.repeat(np.arange(5), n)
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0)
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(str(tmp_path), "rf_landcover.joblib"))
    pred = rf.predict(X)
    assert (pred == y).mean() > 0.95
    # predict_epoch must skip non-finite cells (-1) and classify the rest
    import lib.model as M
    aoi = AOI("t", "t", "t", (0, 0, 0.05, 0.05), "t")
    feats_full = rng.uniform(0, 1, (60, 60, 8)).astype("float32")
    feats_full[0, 0, :] = np.nan
    np.savez(os.path.join(str(tmp_path), "t_2026.npz"), features=feats_full)
    monkeypatch.setattr(M, "DERIVED_DIR", str(tmp_path))
    pred_arr = M.predict_epoch(aoi, "2026", rf)
    assert pred_arr[0, 0] == -1
    assert pred_arr.shape == (60, 60)
    valid = pred_arr >= 0
    assert valid.sum() == 60 * 60 - 1
    # predicted class distribution sanity: all classes present
    assert len(np.unique(pred_arr[valid])) == 5


# ---------------------------------------------------------------------------
# change detection
# ---------------------------------------------------------------------------
def test_transition_codes_and_products(synth_aoi):
    p0 = np.full((60, 60), 1, "int16")  # grass
    p1 = np.full((60, 60), 1, "int16")
    p1[10:20, 10:20] = 3   # grass -> built
    p1[30:40, 30:40] = 0   # grass -> forest (afforestation)
    tr = change_mod.transition_map_from_arrays(p0, p1)
    assert tr[5, 5] == 1 * 5 + 1
    assert tr[15, 15] == 1 * 5 + 3
    assert tr[35, 35] == 1 * 5 + 0
    # urban_growth product should match grass->built
    m = change_mod.product_mask(tr, "urban_growth")
    assert m[15, 15] and not m[35, 35]
    # forest_loss from->forest only; grass->forest is NOT forest loss
    m = change_mod.product_mask(tr, "forest_loss")
    assert not m[15, 15] and not m[35, 35]
    # new_water from any land -> water
    p1b = p1.copy(); p1b[50:55, 50:55] = 4
    trb = change_mod.transition_map_from_arrays(p0, p1b)
    m = change_mod.product_mask(trb, "new_water")
    assert m[52, 52]
    # invalid propagation
    p0i = p0.copy(); p0i[0, 0] = -1
    tri = change_mod.transition_map_from_arrays(p0i, p1)
    assert tri[0, 0] == -1


def test_patches_geojson_area_filter(synth_aoi, tmp_path):
    tr = np.full((240, 240), -1, "int16")
    # a 20x20-cell block of grass->built = 400 cells * 0.81 ha ≈ 324 ha
    tr[10:30, 10:30] = 1 * 5 + 3
    gj = change_mod.patches_geojson(synth_aoi, "urban_growth", tr,
                                    min_ha=6.0)
    assert gj is not None
    assert len(gj["features"]) >= 1
    assert gj["properties"]["total_ha"] > 300
    # tiny patch below threshold -> dropped
    tr2 = tr.copy()
    tr2[10:30, 10:30] = -1
    tr2[100:102, 100:102] = 1 * 5 + 3  # 4 cells ~ 3.2 ha < 6 ha
    gj2 = change_mod.patches_geojson(synth_aoi, "urban_growth", tr2,
                                     min_ha=6.0)
    assert gj2 is None
