"""Pipeline entry: fetch -> composite -> label -> train -> predict -> products.

Usage: python scripts/run_pipeline.py [--aoi ncr-gurugram] [--skip-fetch]
Stages are idempotent; artifacts land in data/derived/ and api/data/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from lib.aois import AOIS, AOI_BY_ID, AOI, EPOCH_BEFORE, EPOCH_AFTER, EPOCH_TRAIN, EPOCHS, PRODUCTS  # noqa: E402
from lib import stac_io, labels as labels_mod, model as model_mod, change as change_mod  # noqa: E402

API_DATA = os.path.join(ROOT, "api", "data")
QA_DIR = os.path.join(ROOT, "_qa")
os.makedirs(API_DATA, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)


def stage_composites(aois, epochs):
    print("== Stage 1: composites ==")
    for aoi in aois:
        for ek in epochs:
            for attempt in range(3):
                t0 = time.time()
                try:
                    res = stac_io.build_composite(aoi, ek, max_items=12,
                                                  min_scenes=3)
                    status = "ok" if res else "SKIP/FAIL"
                    print(f"  {aoi.id}/{ek}: {status} in {time.time()-t0:.0f}s")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  {aoi.id}/{ek}: attempt {attempt+1} error: "
                          f"{type(exc).__name__}: {exc}")
                    time.sleep(20 * (attempt + 1))
            else:
                print(f"  {aoi.id}/{ek}: FAILED after retries, continuing")


def stage_labels(aois):
    print("== Stage 2: labels ==")
    for aoi in aois:
        t0 = time.time()
        res = labels_mod.build_labels(aoi)
        n = int((res["labels"] != -1).sum())
        print(f"  {aoi.id}: {n} labelled cells in {time.time()-t0:.0f}s")


def stage_train(aois):
    print("== Stage 3: train/eval ==")
    return model_mod.train_eval(aois)


def stage_predict(aois):
    print("== Stage 4: predict 2019/2026 ==")
    rf = model_mod.load_model()
    for aoi in aois:
        for ek in (EPOCH_BEFORE, EPOCH_AFTER):
            pred = model_mod.predict_epoch(aoi, ek, rf)
            if pred is None:
                print(f"  {aoi.id}/{ek}: missing features, skipped")
            else:
                from collections import Counter
                c = Counter(pred[pred >= 0].tolist())
                dist = {lib_cls_name(i): int(v) for i, v in sorted(c.items())}
                print(f"  {aoi.id}/{ek}: {dist}")


def lib_cls_name(i):
    from lib.aois import CLASSES
    return CLASSES[i] if 0 <= i < len(CLASSES) else str(i)


def stage_products(aois):
    print("== Stage 5: change products ==")
    manifest = []
    for aoi in aois:
        tr = change_mod.transition_map(aoi)
        if tr is None:
            print(f"  {aoi.id}: missing predictions - skipped")
            continue
        aoi_dir = os.path.join(API_DATA, aoi.id)
        os.makedirs(aoi_dir, exist_ok=True)
        png = os.path.join(QA_DIR, f"{aoi.id}_transition.png")
        change_mod.transition_image(aoi, png, tr)
        summary = {"aoi": aoi.id, "name": aoi.name, "focus": aoi.focus,
                   "story": aoi.story, "bbox": list(aoi.bbox),
                   "epochs": [EPOCH_BEFORE, EPOCH_AFTER],
                   "products": []}
        for pkey in PRODUCTS:
            gj = change_mod.patches_geojson(aoi, pkey, tr)
            if gj is None:
                continue
            with open(os.path.join(aoi_dir, f"{pkey}.geojson"), "w") as fh:
                json.dump(gj, fh)
            summary["products"].append({
                "key": pkey,
                "title": PRODUCTS[pkey].title,
                "color": PRODUCTS[pkey].color,
                "description": PRODUCTS[pkey].description,
                "total_ha": gj["properties"]["total_ha"],
                "n_patches": gj["properties"]["n_patches"],
            })
            print(f"  {aoi.id}/{pkey}: {gj['properties']['total_ha']} ha,"
                  f" {gj['properties']['n_patches']} patches")
        with open(os.path.join(aoi_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        manifest.append(summary)
    with open(os.path.join(API_DATA, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  manifest: {len(manifest)} AOIs -> api/data/manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None, help="single AOI id (default: all)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="use cached composites only")
    args = ap.parse_args()

    aois = [AOI_BY_ID[args.aoi]] if args.aoi else AOIS
    epochs = [EPOCH_BEFORE, EPOCH_TRAIN, EPOCH_AFTER]

    t_all = time.time()
    if not args.skip_fetch:
        stage_composites(aois, epochs)
    stage_labels(aois)
    m = stage_train(aois)
    stage_predict(aois)
    stage_products(aois)
    print(f"== done in {time.time()-t_all:.0f}s ==")


if __name__ == "__main__":
    main()
