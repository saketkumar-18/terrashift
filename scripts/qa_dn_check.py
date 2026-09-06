"""QA: read raw green DN at one fixed point from 2019/2021/2026 scenes.

If 2021 DN is ~1000 lower than 2026 DN at the same stable spot while both
advertise offset -0.1, the e84 sentinel-2-l2a COGs store unshifted DN and
the advertised offset must NOT be applied for that collection.
"""
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin

LON, LAT = 77.05, 28.35  # inside NCR AOI, stable mixed land
GRID = 1.0 / 1200.0

queries = [
    ("2019", "sentinel-2-l2a",
     "2019-01-01T00:00:00Z/2019-03-31T23:59:59Z"),
    ("2021", "sentinel-2-l2a",
     "2021-01-01T00:00:00Z/2021-03-31T23:59:59Z"),
    ("2026", "sentinel-2-c1-l2a",
     "2026-01-01T00:00:00Z/2026-03-31T23:59:59Z"),
]

for ek, coll, dt in queries:
    r = requests.post(
        "https://earth-search.aws.element84.com/v1/search",
        json={"collections": [coll],
              "bbox": [LON - 0.01, LAT - 0.01, LON + 0.01, LAT + 0.01],
              "datetime": dt,
              "query": {"eo:cloud_cover": {"lt": 20}},
              "limit": 1},
        timeout=60)
    item = r.json()["features"][0]
    a = item["assets"]["green"]
    rb = (a.get("raster:bands") or [{}])[0]
    scale, offset = rb.get("scale"), rb.get("offset")

    tr = from_origin(LON - 5 * GRID, LAT + 5 * GRID, GRID, GRID)
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="5"):
        with rasterio.open(a["href"]) as src:
            with WarpedVRT(src, crs="EPSG:4326", transform=tr,
                           width=10, height=10,
                           resampling=Resampling.bilinear) as vrt:
                dn = vrt.read(1).astype("float32")
    center = dn[5, 5]
    refl_advertised = center * scale + offset
    refl_scaleonly = center * scale
    print(f"{ek} {item['id'][:45]:<45} scale={scale} offset={offset} "
          f"DN(center)={center:8.1f} -> advert={refl_advertised:.4f} "
          f"scale-only={refl_scaleonly:.4f}")
    print(f"     DN 5x5 window:\n{dn[:5, :5].astype(int)}")
