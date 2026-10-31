"""Extract DR16Q (Wu & Shen 2022) broad-line properties and quality flags for a selection, matched on PLATE/MJD/FIBERID."""
from pathlib import Path
import numpy as np, pandas as pd
from dr16q_lines import LINE_COL as COL, line_quality, load_dr16q, match_dr16q
S = str(Path(__file__).resolve().parent)
sel = pd.read_csv(f"{S}/selection.csv")
cat = pd.read_parquet("data/S82/Catalog.parquet", columns=["objectId", "PLATE", "MJD", "FIBERID", "Z_SYS"])
cat["objectId"] = cat.objectId.astype(str)
sel["objectId"] = sel.objectId.astype(str)
sel = sel.merge(cat, on="objectId", how="left")
d = load_dr16q()
key = pd.DataFrame(dict(PLATE=d["PLATE"], MJD=d["MJD"], FIBERID=d["FIBERID"])).reset_index()
sel = sel.merge(key, on=["PLATE", "MJD", "FIBERID"], how="left")
# quality flags are computed over every S82 quasar with a DR16Q match (sets the EW outlier threshold), then joined
full = match_dr16q(cat, d).drop_duplicates("objectId")
qual = pd.concat([line_quality(d, full["index"].to_numpy(), full.Z_SYS.to_numpy(), full.PLATE.to_numpy(), line)
                  .assign(objectId=full.objectId.to_numpy(), line=line) for line in COL])
sel = sel.merge(qual, on=["objectId", "line"], how="left")
sel["good"] = sel["good"].fillna(False).astype(bool)
rows = []
for _, r in sel.iterrows():
    arr = np.asarray(d[COL[r.line]][int(r["index"])], float) if np.isfinite(r["index"]) else np.full(6, np.nan)
    rows.append(arr)
arr = np.array(rows)
for i, n in enumerate(["peak_wave", "centroid_wave", "flux", "logL_line", "fwhm", "ew_rest"]):
    sel[n] = arr[:, i]
sel.to_csv(f"{S}/line_props.csv", index=False)
print(sel.groupby("line")[["peak_wave", "fwhm", "ew_rest"]].median())
print("unmatched:", sel["index"].isna().sum(), "| failing line_quality:", (~sel.good).sum())
