"""R-L summary of prm_lag_fit.py results: rest-frame Hbeta lag vs L5100, against Bentz et al. (2013).

Extended objects (morphology/morphology.csv from query_morphology.py: noise-aware psfMag - cModelMag cut)
are dropped from the statistics and the plot; host light dilutes the variability and biases the lag.

With --min-snr X, only objects with line-band per-point variability S/N > X (line_snr.py) are kept; the plot is then
prm_RL_snr<X>.pdf.

usage: python prm_analyze.py <run_dir> [--min-snr X]   (reads <run_dir>/out/*.json; writes summary.csv and prm_RL.pdf)
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from jaxsedfit.mplstyle import style_path

R = Path(sys.argv[1])
MIN_SNR = float(sys.argv[sys.argv.index("--min-snr") + 1]) if "--min-snr" in sys.argv else None
MORPH = Path(__file__).resolve().parents[1] / "morphology" / "morphology.csv"
REF_A, REF_B = 1.527, 0.533  # Bentz et al. 2013: log tau = 1.527 + 0.533 (log L5100 - 44)
rows = []
for f in glob.glob(str(R / "out/*.json")):
    j = json.load(open(f))
    lo, med, hi = j["q_post_rest"]
    rows.append(dict(objectId=j["objectId"], z=j["z"], band=j["line_band"], frac=j["frac"], logL=j["logL5100"],
                     lag=med, lo=lo, hi=hi, width=np.log10(hi / lo) / 2, multimodal=j["multimodal"], edge=j["edge"]))
d = pd.DataFrame(rows)
d["ref"] = 10 ** (REF_A + REF_B * (d.logL - 44))
d["dex"] = np.log10(d.lag / d.ref)
d["clean"] = ~d.multimodal & ~d.edge
morph = pd.read_csv(MORPH, dtype={"objectId": str})[["objectId", "extended"]]
stats = pd.read_csv(Path(__file__).resolve().parent / "line_snr.csv", dtype={"objectId": str})
stats = stats[["objectId", "band", "snr"]].rename(columns={"snr": "snr_line"})
d = d.merge(morph, on="objectId", how="left").merge(stats, on=["objectId", "band"], how="left")
d.to_csv(R / "summary.csv", index=False)
print(f"{len(d)} fits: {int((d.extended == True).sum())} extended (dropped), {int(d.extended.isna().sum())} without SDSS morphology (kept)")
d = d[d.extended != True]
no_L = ~(d.logL > 40)  # DR16Q LOGL5100 missing (stored as 0)
if no_L.any():
    print(f"{int(no_L.sum())} without L5100 (dropped): {', '.join(d.objectId[no_L])}")
d = d[~no_L]
if MIN_SNR is not None:
    print(f"line-band S/N > {MIN_SNR:g}: {int((d.snr_line > MIN_SNR).sum())} of {len(d)} kept")
    d = d[d.snr_line > MIN_SNR]
nmad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 2 else np.nan
for lab, s in [("all", d), ("single peak", d[d.clean]), ("single peak, half-width < 0.3 dex", d[d.clean & (d.width < 0.3)])]:
    rho = spearmanr(s.logL, np.log10(s.lag)) if len(s) > 3 else (np.nan, np.nan)
    print(f"{lab:36s} N={len(s):3d}  offset {s.dex.median():+.2f} dex  NMAD {nmad(s.dex):.2f}  "
          f"within 2x of R-L {np.mean(np.abs(s.dex) < np.log10(2)):.0%}  rho(L, lag) {rho[0]:+.2f} (p={rho[1]:.2g})")

plt.style.use(style_path())
fig, ax = plt.subplots(figsize=(5.5, 4.6))
L = np.linspace(d.logL.min() - 0.2, d.logL.max() + 0.2, 50)
ax.plot(L, 10 ** (REF_A + REF_B * (L - 44)), "k-", lw=1, label="Bentz et al. 2013")
ax.fill_between(L, 10 ** (REF_A + REF_B * (L - 44) - 0.2), 10 ** (REF_A + REF_B * (L - 44) + 0.2), color="0.9", lw=0)
for m, kw in [(d.clean, dict(fmt="o", color="C0", label="single peak")),
              (~d.clean, dict(fmt="x", color="C3", label="multimodal/edge"))]:
    s = d[m]
    ax.errorbar(s.logL, s.lag, yerr=[s.lag - s.lo, s.hi - s.lag], ms=4, lw=0.6, alpha=0.8, **kw)
ax.set(yscale="log", ylim=(1.0, 1500), xlabel=r"log $\lambda L_{5100}$ (erg s$^{-1}$)", ylabel="rest-frame Hβ lag (days)")
ax.text(0.04, 0.95, f"S82 photometric Hβ, a > 0.25, point sources" + (f", line-band S/N > {MIN_SNR:g}" if MIN_SNR else "") + f"\nN = {len(d)}", transform=ax.transAxes, va="top", fontsize=9)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = R / ("prm_RL.pdf" if MIN_SNR is None else f"prm_RL_snr{MIN_SNR:g}.pdf")
fig.savefig(out)
print("plot:", out)
