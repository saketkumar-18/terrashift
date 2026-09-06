"""TerraShift API: serves precomputed change artifacts (no raster work).

Endpoints:
- GET /api/health           -> status
- GET /api/aois             -> manifest (all AOIs, products, totals)
- GET /api/aois/{aoi_id}    -> one AOI summary
- GET /api/aois/{aoi_id}/{product}.geojson -> product patches
- GET /api/model            -> RF metrics
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.normpath(os.path.join(HERE, "..", "models"))

app = FastAPI(title="TerraShift API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
    allow_headers=["*"],
)


def _read_json(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, f"not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/api/health")
def health():
    manifest = os.path.join(DATA, "manifest.json")
    n_aois = 0
    if os.path.exists(manifest):
        try:
            n_aois = len(_read_json(manifest))
        except Exception:
            n_aois = 0
    return {"status": "ok", "service": "terrashift",
            "n_aois": n_aois, "version": "1.0.0"}


@app.get("/api/aois")
def aois():
    return _read_json(os.path.join(DATA, "manifest.json"))


@app.get("/api/aois/{aoi_id}")
def aoi_detail(aoi_id: str):
    m = _read_json(os.path.join(DATA, "manifest.json"))
    for a in m:
        if a["aoi"] == aoi_id:
            return a
    raise HTTPException(404, f"unknown aoi: {aoi_id}")


@app.get("/api/aois/{aoi_id}/{product}")
def aoi_product(aoi_id: str, product: str):
    if not product.endswith(".geojson"):
        product += ".geojson"
    p = os.path.normpath(os.path.join(DATA, aoi_id, product))
    if not p.startswith(DATA):  # path traversal guard
        raise HTTPException(400, "bad path")
    return JSONResponse(_read_json(p))


@app.get("/api/model")
def model_metrics():
    return _read_json(os.path.join(MODELS, "metrics.json"))
