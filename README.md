# TerraShift — Satellite Change Detection (2019 → 2026)

**Live:** https://terrashift.vercel.app · Sentinel-2 L2A · Random-forest land-cover classification · change products per AOI

TerraShift monitors three Indian change hotspots — the Delhi-NCR Gurugram
urbanisation corridor, Papum Pare (Arunachal Pradesh) forest loss, and the
Jharia coalfield mining belt — by comparing dry-season Sentinel-2
composites from **2019 vs 2026** on a fixed 90 m grid and publishing
geodesic-area change products (urban growth, forest loss, bare/mining
expansion, new water, water loss).

## How it works

```
Earth Search STAC (Sentinel-2 L2A COGs)
  → windowed WarpedVRT reads onto a shared 1/1200° EPSG:4326 grid   (lib/stac_io.py)
  → per-scene SCL cloud mask → median composite per epoch            (2019 / 2021 / 2026)
  → ESA WorldCover 2021 labels aggregated 10×10 to 90 m purity      (lib/labels.py)
  → RandomForest (8 spectral features, spatial-block 5-fold CV)      (lib/model.py)
  → epoch classification → transition grid → disjoint products      (lib/change.py)
  → area-filtered GeoJSON patches (geodesic areas, ≥ 6 ha)          → api/data/
```

- **Model features:** green, red, NIR, SWIR16 reflectance + NDVI, NDBI,
  MNDWI, brightness — 8 features per cell.
- **Labels:** WorldCover 2021 v200 (10 m) dominant-class per 90 m cell,
  purity ≥ 0.70, mapped onto 5 classes (forest, grass/agri, bare, built,
  water).
- **Validation:** GroupKFold(5) on 5×5-cell spatial blocks (no spatial
  leakage), class-balanced training folds, unbalanced held-out folds —
  honest priors. See `models/metrics.json`.
- **Products are disjoint** by (from, to) class pair — areas never
  double-count across products.

## Repo layout

| Path | What |
|---|---|
| `lib/` | grid, STAC reads, labels, model, change products |
| `scripts/run_pipeline.py` | end-to-end pipeline (idempotent, cached) |
| `api/` | FastAPI serverless serving precomputed GeoJSON |
| `web/` | MapLibre dark frontend on Esri World Imagery |
| `models/` | trained RF + metrics |
| `tests/` | offline unit tests on synthetic rasters |
| `reports/REPORT.md` | full report (data provenance, method, results) |

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt       # or: uv pip install numpy rasterio ...
python scripts/run_pipeline.py        # fetch + composite + train + products
python -m pytest tests/ -q            # offline tests
uvicorn api.index:app --port 8000     # API at localhost:8000/api/*
# web/index.html expects same-origin /api — serve with a small static
# proxy or deploy to Vercel (vercel.json wires /api → api/index.py).
```

## Data & provenance (all open, no auth)

| Source | Use | License |
|---|---|---|
| Sentinel-2 L2A (via Earth Search STAC) | 2019/2021/2026 dry-season composites | Copernicus/ESA free use |
| ESA WorldCover 2021 v200 | training labels | CC BY 4.0 |
| Esri World Imagery | basemap | attribution required |

Composites: median reflectance over ≥ 3 cloud-masked scenes per epoch;
scenes with < 10 % usable pixels in the AOI are dropped. Scene IDs used
are recorded inside each composite npz (`scene_ids`).

## Verification

- `python -m pytest tests/ -q` — 8 offline tests (grid, features, cloud
  masking, label purity, model round-trip, transitions, area filtering)
- CI: GitHub Actions runs the same suite on push/PR.
- Live: homepage 200, `/api/health` ok, `/api/aois` returns manifest,
  product GeoJSONs fetchable — all curl-verified after deploy.

## Known limitations

- 90 m grid: change smaller than ~0.8 ha is invisible; linear features
  (roads) only show up where ≥ 3 cells wide.
- WorldCover 2021 labels are imperfect in rapidly-changing areas (label
  year is 2021; epoch-2019/2026 composites stretch the temporal match).
- Optical-only: persistent cloud cover or haze can leave NaN cells that
  are excluded from products (counted in n_obs).
- RF cannot model spatial context; per-cell classification means salt-
  and-pepper noise is possible (median composites + min 6 ha patches
  suppress it).

## License

Code: MIT. Data: as per sources above.
