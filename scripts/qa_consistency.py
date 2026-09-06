"""QA: post-fix epoch radiometry consistency (median green per AOI/epoch)."""
import numpy as np

print(f"{'AOI':<14} {'2019':>7} {'2021':>7} {'2026':>7}")
for aoi in ("ncr-gurugram", "papum-pare", "jharia"):
    row = []
    for ek in ("2019", "2021", "2026"):
        z = np.load(f"data/derived/{aoi}_{ek}.npz")
        f = z["features"]
        valid = np.isfinite(f).all(axis=2)
        row.append(np.nanmedian(f[..., 0][valid]))
    print(f"{aoi:<14} {row[0]:>7.4f} {row[1]:>7.4f} {row[2]:>7.4f}")
