"""Regression: e84-legacy offset bug must never come back.

element84 legacy `sentinel-2-l2a` advertises offset -0.1 but stores
UNshifted DN; official `sentinel-2-c1-l2a` stores shifted DN and the
offset is correct. _band_specs must apply the advertised offset ONLY
for c1 items.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.stac_io import _band_specs


def _item(collection, offset):
    assets = {}
    for b in ("green", "red", "nir", "swir16", "scl"):
        assets[b] = {
            "href": f"https://example/{b}.tif",
            "raster:bands": [{"nodata": 0, "scale": 0.0001,
                              "offset": offset}],
        }
    return {"collection": collection, "assets": assets,
            "properties": {}}


def test_legacy_offset_ignored():
    specs = _band_specs(_item("sentinel-2-l2a", -0.1))
    assert specs is not None
    for b in ("green", "red", "nir", "swir16", "scl"):
        assert specs[b]["offset"] == 0.0, "legacy e84 offset must be zeroed"
    assert specs["green"]["scale"] == 0.0001


def test_c1_offset_applied():
    specs = _band_specs(_item("sentinel-2-c1-l2a", -0.1))
    assert specs is not None
    for b in ("green", "red", "nir", "swir16", "scl"):
        assert specs[b]["offset"] == -0.1, "c1 offset must be applied"


def test_zero_offset_stays_zero():
    specs = _band_specs(_item("sentinel-2-l2a", 0.0))
    assert specs["green"]["offset"] == 0.0
    specs = _band_specs(_item("sentinel-2-c1-l2a", 0.0))
    assert specs["green"]["offset"] == 0.0
