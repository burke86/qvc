"""Line-band per-point variability S/N (pipeline/line_snr.py) by band, for every good line with a > 0.25.

One panel per SDSS band; each line in the band where its a is largest, stacked by line. The SDSS-RM test bands (a = 1)
are shown for reference in every panel.
usage (from the qvc repo root): python experiments/blr_lags/figures/line_snr_bands.py [sdss_rm_dir]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jaxsedfit.mplstyle import style_path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from line_snr import ml_amplitude  # noqa: E402

RM = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "research" / "sdss_rm_repo"
LINES = [("Halpha", "Hα"), ("Hbeta", "Hβ"), ("MgII", "Mg II"), ("CIV", "C IV")]
fr = pd.read_csv(HERE.parent / "pipeline" / "chi20_line_fractions.csv", dtype={"objectId": str})
snr = pd.read_csv(HERE.parent / "pipeline" / "line_snr.csv", dtype={"objectId": str})
best = fr[fr.good & (fr.frac > 0.25)].sort_values("frac").groupby(["objectId", "line"]).tail(1)
best = best.merge(snr[["objectId", "band", "snr"]], on=["objectId", "band"])

rmids = sorted({int(l.split()[0]) for l in open(RM / "jobs.txt")})
y, e, grp = [], [], []
for r in rmids:
    d = np.loadtxt(RM / "synthetic_hbeta_prm" / f"rm{r:03d}" / "prm_hb_a1_lc.txt")
    y.append(d[:, 1] / np.median(d[:, 1]) - 1); e.append(d[:, 2] / np.median(d[:, 1])); grp += [r] * len(d)
rm = ml_amplitude(np.concatenate(y), np.concatenate(e), np.array(grp), np.zeros(len(grp), int)).snr

plt.style.use(style_path())
bins = np.geomspace(0.1, 30, 30)
fig, axes = plt.subplots(1, 5, figsize=(17, 4.3), sharex=True)
for ax, band in zip(axes, "ugriz"):
    s = best[best.band == band]
    vals = [s[s.line == l].snr.clip(bins[0], bins[-1]) for l, _ in LINES]
    ax.hist(vals, bins=bins, stacked=True, color=[f"C{i}" for i in range(4)], label=[lab for _, lab in LINES])
    ax.hist(rm.clip(bins[0], bins[-1]), bins=bins, histtype="step", lw=1.6, color="k", label="SDSS-RM test band (a = 1)")
    ax.axvline(3, color="0.4", ls="--", lw=1.2, label="cut (3)")
    ax.set(xscale="log", yscale="log", ylim=(0.8, None), xlabel="per-point variability S/N")
    ax.text(0.04, 0.95, f"{band} band\nN = {len(s)}, S/N > 3: {int((s.snr > 3).sum())}", transform=ax.transAxes,
            va="top", fontsize=9)
axes[0].set_ylabel("number of quasars")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=9)
fig.savefig(HERE / "line_snr_bands.pdf")
fig.savefig(HERE / "line_snr_bands.png", dpi=150)  # for the README; GitHub does not render PDFs inline
print(HERE / "line_snr_bands.pdf")
