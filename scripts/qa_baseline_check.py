"""QA diagnostic: processing baseline / scale-offset across the 2021 scenes."""
import json

import numpy as np
import requests

z = np.load("data/derived/ncr-gurugram_2021.npz", allow_pickle=False)
ids = json.loads(str(z["scene_ids"]))
print("2021 scenes used:", ids)


def meta(item_id, coll):
    r = requests.post(
        "https://earth-search.aws.element84.com/v1/search",
        json={"collections": [coll], "ids": [item_id], "limit": 1},
        timeout=60)
    f = r.json()["features"]
    return f[0] if f else None


for iid in ids:
    for coll in ("sentinel-2-l2a", "sentinel-2-c1-l2a"):
        it = meta(iid, coll)
        if it:
            a = it["assets"]["green"]
            rb = (a.get("raster:bands") or [{}])[0]
            props = it["properties"]
            print(f"{iid} [{coll}] baseline={props.get('s2:processing_baseline')} "
                  f"scale={rb.get('scale')} offset={rb.get('offset')}")
            break
    else:
        print(iid, "not found in either collection")
