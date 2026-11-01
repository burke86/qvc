"""Per-band intrinsic variability from SDSS + PS1 + ZTF: maximum-likelihood excess amplitude, using every point.

For each object and band, in fractional flux, y_i ~ N(mu_s, sigma^2 + e_i^2): one intrinsic amplitude sigma shared by
all surveys, a mean mu_s per survey (absorbs zero-point offsets), and each point's own error e_i (Almaini et al. 2000;
Vaughan et al. 2003). sigma is found on a fine grid with the mu_s profiled out. Each survey is first clipped at 5
robust sigma about its median. Outputs per object-band:
- sigma, sigma_err (from the curvature of the profile likelihood) and det = sigma / sigma_err (detection significance;
  more data never lowers it in expectation, unlike a median-error S/N);
- e_typ = (mean e^-2)^(-1/2), the inverse-variance typical error, and snr = sigma / e_typ (per-point S/N, the
  quantity to compare with SDSS-RM).
Each survey table is read once (only the needed objects and columns).
Run from the qvc repo root: python experiments/blr_lags/pipeline/line_snr.py [min_frac=0.25]   (writes line_snr.csv)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

S = Path(__file__).resolve().parent
D = "data/S82"
BANDS = dict(enumerate("ugriz"))  # filterID -> band in all three tables (PS1 y = 5 is dropped)
SIGMA_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 3.0, 400)])


def ml_amplitude(y, e, group, subgroup):
    """ML intrinsic amplitude per group; subgroup (nested in group) gets its own mean. Returns a DataFrame by group."""
    g = pd.factorize(group)[0]
    gs = pd.factorize(pd.Series(group).astype(str) + "|" + pd.Series(subgroup).astype(str))[0]
    G = g.max() + 1
    ll = np.empty((G, SIGMA_GRID.size))
    for k, s in enumerate(SIGMA_GRID):
        w = 1.0 / (s ** 2 + e ** 2)
        mu = np.bincount(gs, w * y) / np.bincount(gs, w)
        r = y - mu[gs]
        ll[:, k] = -0.5 * np.bincount(g, r ** 2 * w - np.log(w), minlength=G)
    k = np.argmax(ll, axis=1)
    kk = np.clip(k, 1, SIGMA_GRID.size - 2)
    x0, x1, x2 = SIGMA_GRID[kk - 1], SIGMA_GRID[kk], SIGMA_GRID[kk + 1]
    f0, f1, f2 = (ll[np.arange(G), kk + d] for d in (-1, 0, 1))
    # quadratic through the three grid points around the peak: vertex and curvature
    a = ((f2 - f1) / (x2 - x1) - (f1 - f0) / (x1 - x0)) / (x2 - x0)
    b = (f1 - f0) / (x1 - x0) - a * (x1 + x0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where((k > 0) & (a < 0), -b / (2 * a), SIGMA_GRID[k])
        err = np.where(a < 0, 1.0 / np.sqrt(-2 * a), np.nan)
    sig = np.where(k == 0, 0.0, sig)
    e_typ = 1.0 / np.sqrt(np.bincount(g, e ** -2) / np.bincount(g))
    out = pd.DataFrame(dict(n=np.bincount(g), sigma=sig, sigma_err=err, e_typ=e_typ,
                            det=sig / err, snr=sig / e_typ, dlogL=ll.max(axis=1) - ll[:, 0]))
    out.index = pd.Series(group).iloc[np.unique(g, return_index=True)[1]].to_numpy()
    return out


def clip_fractional(lc, key):
    """Fractional flux about each key-group median magnitude, keeping points within 5 robust sigma."""
    lc = lc.copy()
    lc["dm"] = lc.mag - lc.groupby(key).mag.transform("median")
    mad = lc.groupby(key).dm.transform(lambda v: 1.4826 * np.median(np.abs(v)))
    lc = lc[np.abs(lc.dm) <= 5 * mad + 1e-9]
    lc["fl"] = 10 ** (-0.4 * lc.dm) - 1
    lc["fe"] = 0.4 * np.log(10) * lc.err * (1 + lc.fl)
    return lc


if __name__ == "__main__":
    MIN_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    fr = pd.read_csv(S / "chi20_line_fractions.csv", dtype={"objectId": str})
    ids = sorted(fr[fr.good & (fr.frac > MIN_FRAC)].objectId.unique())
    cat = pd.read_parquet(f"{D}/Catalog.parquet", columns=["objectId", "ps1objID"])
    cat["objectId"] = cat.objectId.astype(str)
    cat = cat[cat.objectId.isin(ids)].drop_duplicates("objectId")
    ps1_map = {int(p): o for o, p in zip(cat.objectId, cat.ps1objID) if pd.notna(p)}  # same int cast as qvcfit

    parts = []
    t = pq.read_table(f"{D}/dr16s82_sdssLCRaw.parquet", columns=["objectId", "psMag", "psMagErr_p3", "filterID"],
                      filters=[("objectId", "in", ids), ("filterID", "in", list(BANDS))]).to_pandas()
    parts.append(pd.DataFrame(dict(objectId=t.objectId.astype(str), f=t.filterID, mag=t.psMag, err=t.psMagErr_p3, survey="sdss")))
    for name, mcol, ecol in [("dr16s82_ps1LCRaw", "psfMag", "psfMagErr_p3"), ("dr16s82_ZuberLCRaw", "mag", "magerr_p3")]:
        t = pq.read_table(f"{D}/{name}.parquet", columns=["ps1objID", mcol, ecol, "filterID"],
                          filters=[("ps1objID", "in", list(ps1_map)), ("filterID", "in", list(BANDS))]).to_pandas()
        parts.append(pd.DataFrame(dict(objectId=t.ps1objID.map(ps1_map), f=t.filterID, mag=t[mcol], err=t[ecol],
                                       survey="ps1" if "ps1" in name else "ztf")))
    del t
    lc = pd.concat(parts, ignore_index=True)
    del parts
    lc = lc[np.isfinite(lc.mag) & (lc.err > 0) & (lc.err < 1)]
    lc["band"] = lc.f.map(BANDS)
    lc = clip_fractional(lc, ["objectId", "band", "survey"])

    key = lc.objectId + "|" + lc.band
    out = ml_amplitude(lc.fl.to_numpy(), lc.fe.to_numpy(), key.to_numpy(), lc.survey.to_numpy())
    out[["objectId", "band"]] = out.index.to_series().str.split("|", expand=True).to_numpy()
    for s in ("sdss", "ps1", "ztf"):
        out[f"n_{s}"] = lc[lc.survey == s].groupby(key[lc.survey == s]).size().reindex(out.index, fill_value=0).to_numpy()
    cols = ["objectId", "band", "n", "n_sdss", "n_ps1", "n_ztf", "sigma", "sigma_err", "e_typ", "snr", "det", "dlogL"]
    out[cols].to_csv(S / "line_snr.csv", index=False)
    print(f"{len(ids)} objects, {len(lc)} points -> {len(out)} object-bands; median per-point S/N by band:",
          out.groupby("band").snr.median().round(2).to_dict())
