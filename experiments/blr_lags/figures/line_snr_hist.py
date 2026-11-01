"""Line-band variability (ML excess amplitude, pipeline/line_snr.py): S82 Hbeta a > 0.25 vs the SDSS-RM test bands.

SDSS-RM: the synthetic photometric bands of burke86/sdss_rm (Hbeta spectra + same-night photometric continuum) for
the 37 quasars of the lag-recovery test, one survey (the band is already intercalibrated).
usage (from the qvc repo root): python experiments/blr_lags/figures/line_snr_hist.py [sdss_rm_dir]
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
sel = pd.read_csv(HERE.parent / "pipeline" / "prm_selection.csv", dtype={"objectId": str})
snr = pd.read_csv(HERE.parent / "pipeline" / "line_snr.csv", dtype={"objectId": str})
s82 = sel.merge(snr, on=["objectId", "band"])

rmids = sorted({int(l.split()[0]) for l in open(RM / "jobs.txt")})
rm = {}
for tag, lab in [("a1", "a = 1"), ("a0p25", "a = 0.25")]:
    y, e, grp = [], [], []
    for r in rmids:
        d = np.loadtxt(RM / "synthetic_hbeta_prm" / f"rm{r:03d}" / f"prm_hb_{tag}_lc.txt")
        y.append(d[:, 1] / np.median(d[:, 1]) - 1); e.append(d[:, 2] / np.median(d[:, 1])); grp += [f"rm{r:03d}"] * len(d)
    rm[lab] = ml_amplitude(np.concatenate(y), np.concatenate(e), np.array(grp), np.zeros(len(grp), int))

for lab, x in [("S82 Hβ line band, a > 0.25", s82)] + [(f"SDSS-RM band, {k}", v) for k, v in rm.items()]:
    print(f"{lab:28s} N={len(x):3d}  per-point S/N median {x.snr.median():.2f} (16-84%: {x.snr.quantile(.16):.2f}-"
          f"{x.snr.quantile(.84):.2f})  detection median {x.det.median():.1f}")

plt.style.use(style_path())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, col, xlab, bins in [(axes[0], "snr", "per-point variability S/N (σ / typical error)", np.geomspace(0.1, 30, 30)),
                            (axes[1], "det", "variability detection (σ / δσ)", np.geomspace(0.5, 100, 30))]:
    ax.hist(s82[col].clip(bins[0], bins[-1]), bins=bins, color="0.6", label="S82 Hβ line band, a > 0.25")
    for (k, v), ls in zip(rm.items(), ["-", "--"]):
        ax.hist(v[col].clip(bins[0], bins[-1]), bins=bins, histtype="step", lw=1.8, color="C3", ls=ls, label=f"SDSS-RM test band, {k}")
    ax.set(xscale="log", xlabel=xlab)
axes[0].axvline(3, color="C0", lw=1.5, label="cut (3)")
axes[0].set_ylabel("number of quasars")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "line_snr_hist.pdf")
fig.savefig(HERE / "line_snr_hist.png", dpi=150)  # for the README; GitHub does not render PDFs inline
print(HERE / "line_snr_hist.pdf")
