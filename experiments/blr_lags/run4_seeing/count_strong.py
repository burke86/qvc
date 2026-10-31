"""Count chi2_g > 20 Stripe 82 quasars with a broad line contributing > 10% (or 20%) of an SDSS band's flux."""
from pathlib import Path
import numpy as np, pandas as pd, speclite.filters
from astropy.io import fits

S = str(Path(__file__).resolve().parent)
C_KMS = 2.998e5
chi = pd.read_csv("data/lc_chisq.csv", usecols=["object_id", "chi_sq_g"]).rename(columns={"object_id": "objectId"})
cat = pd.read_parquet("data/S82/Catalog.parquet", columns=["objectId", "PLATE", "MJD", "FIBERID", "Z_SYS"])
cat["objectId"] = cat.objectId.astype(str); chi["objectId"] = chi.objectId.astype(str)
df = cat.merge(chi, on="objectId")
df = df[df.chi_sq_g > 20].copy()
d = fits.open("data/dr16q_prop_May01_2024.fits", memmap=True)[1].data
key = pd.DataFrame(dict(PLATE=d["PLATE"], MJD=d["MJD"], FIBERID=d["FIBERID"])).reset_index()
df = df.merge(key, on=["PLATE", "MJD", "FIBERID"], how="inner")
idx = df["index"].to_numpy()
filt = {b: speclite.filters.load_filter(f"sdss2010-{b}") for b in "ugriz"}
norm = {b: np.trapezoid(f(f.wavelength), f.wavelength) for b, f in filt.items()}
z = df.Z_SYS.to_numpy()
rows = []
for line, col in [("Halpha", "HALPHA_BR"), ("Hbeta", "HBETA_BR"), ("MgII", "MGII_BR"), ("CIV", "CIV")]:
    a = np.asarray(d[col][idx], float)
    peak, fwhm, ew = a[:, 0], a[:, 4], a[:, 5]
    ok = np.isfinite(peak) & np.isfinite(fwhm) & np.isfinite(ew) & (ew > 0) & (peak > 0) & (fwhm > 0)
    lam_obs = peak * (1 + z)
    sig = lam_obs * fwhm / C_KMS / 2.3548
    u = np.linspace(-5, 5, 201)
    prof = np.exp(-0.5 * u ** 2)
    for b, f in filt.items():
        lam = lam_obs[:, None] + sig[:, None] * u[None, :]
        T = np.where(ok[:, None], f(np.nan_to_num(lam, nan=1.0).clip(1, 2e4)), 0.0)
        frac = np.where(ok, ew * (1 + z) * np.trapezoid(T * prof, u, axis=1) / np.trapezoid(prof, u) / norm[b], np.nan)
        rows.append(pd.DataFrame(dict(objectId=df.objectId, line=line, band=b, frac=frac)))
fr = pd.concat(rows)
fr.to_csv(f"{S}/chi20_line_fractions.csv", index=False)
print(f"chi2_g > 20 with DR16Q match: {len(df)}")
for thr in (0.10, 0.20):
    s = fr[fr.frac > thr]
    tab = s.pivot_table(index="line", columns="band", values="objectId", aggfunc="count", fill_value=0).reindex(columns=list("ugriz"), fill_value=0)
    tab["any band (unique objects)"] = s.groupby("line").objectId.nunique()
    print(f"\n> {thr:.0%} of band flux (counts per line x band):")
    print(tab.reindex(["Halpha", "Hbeta", "MgII", "CIV"]).fillna(0).astype(int).to_string())
    print("unique objects with any line in any band:", s.objectId.nunique())
