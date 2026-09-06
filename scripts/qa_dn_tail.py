"""QA: AOI-wide DN low-tail test for shift status.

Unshifted DN: water ~150-600, dark land ~100-300, land 100-3000.
Shifted (+1000): water ~1150-1600, dark ~1100-1300, land 1100-4000.
If a 2026 c1 scene has essentially no DN < 1000 AOI-wide while a 2019
scene has a fat tail below 600, the c1 DN is shifted (apply -0.1).
"""
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin

BBOX = (76.85, 28.15, 77.25, 28.55)  # NCR AOI
GRID = 1.0 / 1200.0
W, S, E, N = BBOX
COLS = int(round((E - W) / GRID)) // 4  # decimate 4x for speed
ROWS = int(round((N - S) / GRID)) // 4
TR = from_origin(W, N, GRID * 4, GRID * 4)

queries = [
    ("2019 e84-legacy", "sentinel-2-l2a", "2019-01-01T00:00:00Z/2019-03-31T23:59:59Z"),
    ("2021 e84-legacy", "sentinel-2-l2a", "2021-01-01T00:00:00Z/2021-03-31T23:59:59Z"),
    ("2026 c1-official", "sentinel-2-c1-l2a", "2026-01-01T00:00:00Z/2026-03-31T23:59:59Z"),
]

for label, coll, dt in queries:
    r = requests.post(
        "https://earth-search.aws.element84.com/v1/search",
        json={"collections": [coll], "bbox": list(BBOX), "datetime": dt,
              "query": {"eo:cloud_cover": {"lt": 20}}, "limit": 1},
        timeout=60)
    item = r.json()["features"][0]
    a = item["assets"]["green"]
    rb = (a.get("raster:bands") or [{}])[0]
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="5"):
        with rasterio.open(a["href"]) as src:
            with WarpedVRT(src, crs="EPSG:4326", transform=TR,
                           width=COLS, height=ROWS,
                           resampling=Resampling.bilinear) as vrt:
                dn = vrt.read(1).astype("float32")
    dn = dn[dn > 0]  # drop nodata
    qs = np.percentile(dn, [1, 5, 25, 50, 75, 99])
    below600 = float((dn < 600).mean())
    below1000 = float((dn < 1000).mean())
    print(f"{label} {item['id'][:40]}")
    print(f"   p1={qs[0]:.0f} p5={qs[1]:.0f} p25={qs[2]:.0f} p50={qs[3]:.0f} "
          f"p75={qs[4]:.0f} p99={qs[5]:.0f}  | frac<600: {below600:.3f}  frac<1000: {below1000:.3f}")
