"""API tests: manifest, per-AOI detail, product GeoJSON, 404s, traversal."""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "api"))

from fastapi.testclient import TestClient  # noqa: E402

from api.index import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # fabricate a manifest + one product file into a temp api/data dir
    tmp = tmp_path_factory.mktemp("apidata")
    aoi_dir = tmp / "ncr-gurugram"
    aoi_dir.mkdir()
    manifest = [{
        "aoi": "ncr-gurugram", "name": "NCR", "focus": "urban-growth",
        "story": "test", "bbox": [76.85, 28.15, 77.25, 28.55],
        "epochs": ["2019", "2026"],
        "products": [{
            "key": "urban_growth", "title": "Urban growth",
            "color": "#ff6b35", "description": "d",
            "total_ha": 100.5, "n_patches": 2}]}]
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "properties": {"product": "urban_growth", "area_ha": 60.2},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[77.0, 28.2], [77.01, 28.2],
                                      [77.01, 28.21], [77.0, 28.21],
                                      [77.0, 28.2]]]}}],
        "properties": {"aoi": "ncr-gurugram", "product": "urban_growth",
                       "total_ha": 100.5, "n_patches": 2}}
    (aoi_dir / "urban_growth.geojson").write_text(json.dumps(gj))

    import api.index as idx
    old = idx.DATA
    idx.DATA = str(tmp)
    yield TestClient(app)
    idx.DATA = old


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_aois"] == 1


def test_aois_manifest(client):
    r = client.get("/api/aois")
    assert r.status_code == 200
    m = r.json()
    assert isinstance(m, list) and len(m) == 1
    assert m[0]["aoi"] == "ncr-gurugram"
    assert m[0]["products"][0]["total_ha"] == 100.5


def test_aoi_detail_and_404(client):
    assert client.get("/api/aois/ncr-gurugram").status_code == 200
    r = client.get("/api/aois/nope")
    assert r.status_code == 404


def test_product_geojson(client):
    r = client.get("/api/aois/ncr-gurugram/urban_growth.geojson")
    assert r.status_code == 200
    gj = r.json()
    assert gj["type"] == "FeatureCollection"
    assert gj["properties"]["total_ha"] == 100.5
    # suffix-less variant also works
    r2 = client.get("/api/aois/ncr-gurugram/urban_growth")
    assert r2.status_code == 200


def test_product_missing_404(client):
    assert client.get("/api/aois/ncr-gurugram/forest_loss").status_code == 404


def test_path_traversal_blocked(client):
    r = client.get("/api/aois/ncr-gurugram/..%2F..%2Fmanifest.json")
    assert r.status_code in (400, 404)


def test_model_metrics_missing(client):
    # no models dir in this env -> 404, not crash
    r = client.get("/api/model")
    assert r.status_code in (200, 404)
