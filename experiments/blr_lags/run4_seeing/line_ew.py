"""Extract DR16Q (Wu & Shen 2022) broad-line properties for the run4 selection, matched on PLATE/MJD/FIBERID."""
from pathlib import Path
import numpy as np, pandas as pd
from astropy.io import fits
S = str(Path(__file__).resolve().parent)
sel = pd.read_csv(f"{S}/selection.csv")
cat = pd.read_parquet("data/S82/Catalog.parquet", columns=["objectId", "PLATE", "MJD", "FIBERID"])
cat["objectId"] = cat.objectId.astype(str)
sel["objectId"] = sel.objectId.astype(str)
sel = sel.merge(cat, on="objectId", how="left")
d = fits.open("data/dr16q_prop_May01_2024.fits", memmap=True)[1].data
key = pd.DataFrame(dict(PLATE=d["PLATE"], MJD=d["MJD"], FIBERID=d["FIBERID"])).reset_index()
sel = sel.merge(key, on=["PLATE", "MJD", "FIBERID"], how="left")
COL = dict(Halpha="HALPHA_BR", Hbeta="HBETA_BR", MgII="MGII_BR", CIV="CIV")
rows = []
for _, r in sel.iterrows():
    arr = np.asarray(d[COL[r.line]][int(r["index"])], float) if np.isfinite(r["index"]) else np.full(6, np.nan)
    rows.append(arr)
arr = np.array(rows)
for i, n in enumerate(["peak_wave", "centroid_wave", "flux", "logL_line", "fwhm", "ew_rest"]):
    sel[n] = arr[:, i]
sel.to_csv(f"{S}/line_props.csv", index=False)
print(sel.groupby("line")[["peak_wave", "fwhm", "ew_rest"]].median())
print("unmatched:", sel["index"].isna().sum())
