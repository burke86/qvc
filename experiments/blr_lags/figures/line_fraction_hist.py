"""Histograms of in-band broad-line/continuum flux ratios for chi2_g > 20 Stripe 82 quasars (quality-good lines)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jaxsedfit.mplstyle import style_path

B = Path(__file__).resolve().parents[1]
CUT = 0.5  # strong-line selection: line / continuum flux ratio in the band
fr = pd.read_csv(B / "pipeline/chi20_line_fractions.csv", dtype={"objectId": str})
g = fr[fr.good & fr.frac.notna() & (fr.frac > 0)]
best = g.sort_values("frac").groupby(["objectId", "line"]).tail(1)  # strongest band per object and line
plt.style.use(style_path())
fig, axes = plt.subplots(2, 2, figsize=(10, 8.4), sharex=True)
axes = axes.ravel()
bins = np.logspace(-3, 0.5, 50)
for ax, line, lab in zip(axes, ["Halpha", "Hbeta", "MgII", "CIV"], ["Hα", "Hβ", "Mg II", "C IV"]):
    v = best[best.line == line].frac.values
    ax.hist(v, bins=bins, color="0.55")
    for a, ls in [(0.25, ":"), (0.5, "--"), (0.75, "-."), (1.0, "-")]:
        ax.axvline(a, color="C3", ls=ls, lw=1.3, label=f"SDSS-RM test, a = {a}")
    ax.axvline(CUT, color="C0", lw=4, alpha=0.4, zorder=1, label=f"selection cut ({CUT})")
    ax.set(xscale="log", yscale="log")
    ax.text(0.04, 0.95, f"{lab}\nN = {len(v)}\nmedian {np.median(v):.2f}\n>{CUT}: {np.sum(v > CUT)}",
            transform=ax.transAxes, va="top", fontsize=10)
for ax in axes[2:]:
    ax.set_xlabel("line / continuum flux ratio in band")
for ax in axes[::2]:
    ax.set_ylabel("number of quasars")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)
fig.savefig(Path(__file__).with_suffix(".pdf"))
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)  # for the README; GitHub does not render PDFs inline
print(Path(__file__).with_suffix(".pdf"))
