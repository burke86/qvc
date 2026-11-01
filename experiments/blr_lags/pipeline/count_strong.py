"""Count chi2_g > 20 Stripe 82 quasars with a good-quality broad line contributing > 50% (or 20%) of an SDSS band's flux."""
from pathlib import Path
import numpy as np, pandas as pd
from dr16q_lines import LINE_COL, MAX_SANE_FRAC, band_fractions, line_quality, load_dr16q, match_dr16q

S = str(Path(__file__).resolve().parent)
chi = pd.read_csv("data/lc_chisq.csv", usecols=["object_id", "chi_sq_g"]).rename(columns={"object_id": "objectId"})
cat = pd.read_parquet("data/S82/Catalog.parquet", columns=["objectId", "PLATE", "MJD", "FIBERID", "Z_SYS"])
cat["objectId"] = cat.objectId.astype(str); chi["objectId"] = chi.objectId.astype(str)
d = load_dr16q()
# quality stats (EW outlier threshold) use every S82 quasar with a DR16Q match, not just the chi2 cut
df = match_dr16q(cat, d).drop_duplicates("objectId").merge(chi, on="objectId", how="left")
idx, z, plate = df["index"].to_numpy(), df.Z_SYS.to_numpy(), df.PLATE.to_numpy()
rows = []
for line in LINE_COL:
    q = line_quality(d, idx, z, plate, line)
    for b, frac in band_fractions(d, idx, z, line).items():
        rows.append(pd.DataFrame(dict(objectId=df.objectId, chi_sq_g=df.chi_sq_g, line=line, band=b, frac=frac)).join(q))
fr = pd.concat(rows, ignore_index=True)
fr = fr[fr.chi_sq_g > 20].drop(columns="chi_sq_g")
fr.to_csv(f"{S}/chi20_line_fractions.csv", index=False)
good = fr[fr.good]
worst = good.frac.max()
assert worst <= MAX_SANE_FRAC, f"good-quality line with in-band fraction {worst:.3g}: check line_quality() before using these"
print(f"chi2_g > 20 with DR16Q match: {fr.objectId.nunique()}")
flags = ["covered", "full_window", "measured", "good_fit", "not_outlier"]
print("objects failing each check (line x object):")
print(fr.drop_duplicates(["objectId", "line"]).groupby("line")[flags].apply(lambda x: (~x).sum()).to_string())
for thr in (0.20, 0.50):
    s = good[good.frac > thr]
    tab = s.pivot_table(index="line", columns="band", values="objectId", aggfunc="count", fill_value=0).reindex(columns=list("ugriz"), fill_value=0)
    tab["any band (unique objects)"] = s.groupby("line").objectId.nunique()
    print(f"\n> {thr:.0%} of band flux (counts per line x band):")
    print(tab.reindex(list(LINE_COL)).fillna(0).astype(int).to_string())
    print("unique objects with any line in any band:", s.objectId.nunique())
