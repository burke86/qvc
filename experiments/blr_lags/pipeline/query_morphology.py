"""Extendedness flags for a run's selection: SDSS DR16 PhotoObj (by position) and PS1 DR2 mean photometry (cone search).

extended = SDSS type == 3 (galaxy) or psfMag_r - cModelMag_r > 0.145 (the standard SDSS star/galaxy separator).
The same test in the line band is not used: for faint high-z objects (e.g. Mg II at z ~ 2) the i/z PSF - cModel is noise.
PS1 is matched by position because Catalog.parquet stores ps1objID as a float, which drops its last digits.

usage (from the qvc repo root): python experiments/blr_lags/pipeline/query_morphology.py <run_dir> [--no-ps1]
writes <run_dir>/morphology.csv
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, astropy.units as u
from astropy.coordinates import SkyCoord

EXT_DPSF = 0.145
RADIUS_ARCSEC = 2.0
run = Path(sys.argv[1])
ids = set(pd.read_csv(run / "selection.csv", dtype={"objectId": str}).objectId)
cat = pd.read_parquet("data/S82/Catalog.parquet", columns=["objectId", "RA", "DEC", "Z_SYS", "qg_fragal"])
cat["objectId"] = cat.objectId.astype(str)
t = cat[cat.objectId.isin(ids)].reset_index(drop=True)


def nearest(q, ra, dec, rcol, dcol):
    sep = 3600 * np.hypot((q[rcol] - ra) * np.cos(np.radians(dec)), q[dcol] - dec)
    return (q.assign(sep=sep)[sep < RADIUS_ARCSEC]).sort_values("sep")


def sdss():
    from astroquery.sdss import SDSS
    fields = ["objID", "ra", "dec", "type", "mode"] + [f"{p}_{b}" for p in ("psfMag", "cModelMag") for b in "griz"]
    raw = pd.concat([SDSS.query_crossid(SkyCoord(c.RA.values * u.deg, c.DEC.values * u.deg), photoobj_fields=fields,
                                        radius=RADIUS_ARCSEC * u.arcsec, data_release=16).to_pandas()
                     for c in (t.iloc[k:k + 300] for k in range(0, len(t), 300))])
    rows = []
    for r in t.itertuples():
        m = nearest(raw, r.RA, r.DEC, "ra", "dec")
        if len(m) == 0:
            rows.append(dict(objectId=r.objectId)); continue
        m = m.sort_values(["mode", "sep"]).iloc[0]  # mode 1 = primary
        rows.append(dict(objectId=r.objectId, sdss_objID=m.objID, sdss_sep=m.sep, sdss_type=m.type,
                         **{f"sdss_dpsf_{b}": m[f"psfMag_{b}"] - m[f"cModelMag_{b}"] for b in "griz"}, sdss_psf_r=m.psfMag_r))
    return pd.DataFrame(rows)


def ps1():
    from astroquery.mast import Catalogs
    cols = ["objID", "nDetections"] + [f"{b}Mean{k}Mag" for b in "griz" for k in ("PSF", "Kron")]

    def one(r):
        for attempt in range(4):
            try:
                q = Catalogs.query_region(f"{r.RA} {r.DEC}", radius=RADIUS_ARCSEC / 3600, catalog="Panstarrs",
                                          data_release="dr2", table="mean").to_pandas()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            return dict(objectId=r.objectId, ps1_status="error")
        m = nearest(q, r.RA, r.DEC, "raMean", "decMean")
        if len(m) == 0:
            return dict(objectId=r.objectId, ps1_status="none")
        m = m.iloc[0]
        return dict(objectId=r.objectId, ps1_status="ok", ps1_sep=m.sep, **{f"ps1_{c}": m[c] for c in cols},
                    **{f"ps1_dkron_{b}": m[f"{b}MeanPSFMag"] - m[f"{b}MeanKronMag"] for b in "griz"})

    with ThreadPoolExecutor(2) as ex:  # MAST drops requests at higher concurrency
        return pd.DataFrame(list(ex.map(one, t.itertuples())))


out = t.merge(sdss(), on="objectId", how="left")
out["extended"] = ((out.sdss_type == 3) | (out.sdss_dpsf_r > EXT_DPSF)).where(out.sdss_objID.notna())
if "--no-ps1" not in sys.argv:
    out = out.merge(ps1(), on="objectId", how="left")
out.to_csv(run / "morphology.csv", index=False)
print(f"{len(out)} objects | SDSS matched {out.sdss_objID.notna().sum()} | extended {int(out.extended.sum())}")
