"""QA: NDVI/class-feature distributions 2019 vs 2026 per AOI.

Checks whether epoch-to-epoch class flips ride on a systematic radiometric
shift (bug) or on real spatial structure (change). If NDVI distributions
overlap heavily but the forest class grows only where spatially clustered,
the signal is real/landscape-scale, not an offset artifact.
"""
import numpy as np

for aoi in ("ncr-gurugram", "papum-pare", "jharia"):
    print(f"== {aoi} ==")
    feats = {}
    for ek in ("2019", "2026"):
        z = np.load(f"data/derived/{aoi}_{ek}.npz")
        f = z["features"]
        valid = np.isfinite(f).all(axis=2)
        ndvi = f[..., 4][valid]
        mndwi = f[..., 6][valid]
        feats[ek] = (f, valid)
        print(f"  {ek}: NDVI p25={np.percentile(ndvi,25):.3f} "
              f"p50={np.percentile(ndvi,50):.3f} p75={np.percentile(ndvi,75):.3f} "
              f"| MNDWI p50={np.percentile(mndwi,50):.3f} "
              f"| frac NDVI>0.55: {(ndvi>0.55).mean():.3f}")
    # spatial clustering of forest gain (jharia): forest cells 2026 not 2019
    p19 = np.load(f"data/derived/{aoi}_2019_pred.npz")["pred"]
    p26 = np.load(f"data/derived/{aoi}_2026_pred.npz")["pred"]
    both = (p19 != -1) & (p26 != -1)
    gain = both & (p19 != 0) & (p26 == 0)  # became forest
    loss = both & (p19 == 0) & (p26 != 0)
    # neighbor agreement: fraction of gain cells with >=3 gain neighbors
    g = gain.astype("uint8")
    nb = np.zeros_like(g)
    nb[1:, :] += g[:-1, :]; nb[:-1, :] += g[1:, :]
    nb[:, 1:] += g[:, :-1]; nb[:, :-1] += g[:, 1:]
    core = ((g == 1) & (nb >= 3)).sum()
    print(f"  forest gain cells={gain.sum()} loss cells={loss.sum()} "
          f"| clustered(>=3 nbrs): {core}/{gain.sum() if gain.sum() else 1} "
          f"= {core/max(gain.sum(),1):.2f}")
