"""Lag vs luminosity for objects whose broad line contributes > 10% of the band flux (DR16Q EW-based)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr, theilslopes
from jaxsedfit.mplstyle import style_path

S = str(Path(__file__).resolve().parent)
THR = 0.10
nmad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
plt.style.use(style_path())
fig, axes = plt.subplots(1, 4, figsize=(19, 5))
for ax, (line, lkey, name) in zip(axes, [("Halpha", "5100", r"H$\alpha$"), ("Hbeta", "5100", r"H$\beta$"),
                                        ("MgII", "3000", "Mg II"), ("CIV", "3000", "C IV")]):
    df = pd.read_csv(f"{S}/plots/{line}/{line}_profile_results_with_frac.csv", dtype={"objectId": str})
    d = df[df.frac > THR]
    clean = ~d.multimodal & ~d.edge
    yerr = np.log10([d.lag / d.lo, d.hi / d.lag])
    sc = ax.scatter(d.logL[clean], np.log10(d.lag[clean]), c=d.frac[clean], cmap="viridis", vmin=0.1, vmax=0.4, s=40, zorder=3)
    ax.errorbar(d.logL[clean], np.log10(d.lag[clean]), yerr=yerr[:, clean.values], fmt="none", ecolor="0.4", lw=0.8, zorder=2)
    ax.errorbar(d.logL[~clean], np.log10(d.lag[~clean]), yerr=yerr[:, ~clean.values], fmt="x", color="0.6", ms=6, lw=0.6,
                label="multimodal / edge-pinned")
    if df.ref.notna().any():
        b, a = np.polyfit(df.logL, np.log10(df.ref), 1)
        xs = np.linspace(df.logL.min() - 0.1, df.logL.max() + 0.1, 10)
        ax.plot(xs, a + b * xs, "k-", lw=1.2, label="reference R–L")
    txt = [f"N = {len(d)} ({int(clean.sum())} clean)"]
    c = d[clean]
    if len(c) > 3:
        rho, p = spearmanr(c.logL, np.log10(c.lag))
        sl = theilslopes(np.log10(c.lag), c.logL)
        txt.append(rf"clean: $\rho$ = {rho:+.2f} (p = {p:.2f})")
        txt.append(f"slope = {sl[0]:.2f} [{sl[2]:.2f}, {sl[3]:.2f}]")
        if c.dex_from_ref.notna().any():
            txt.append(f"offset {c.dex_from_ref.median():+.2f}, NMAD {nmad(c.dex_from_ref):.2f} dex")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title(f"{name} (> {THR:.0%})")
    ax.set_xlabel(rf"$\log\,\lambda L_{{{lkey}}}$ (erg s$^{{-1}}$)")
    ax.set_ylim(0.3, 3.1)
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel(r"$\log\,\tau_{\rm rest}$ (days)")
fig.colorbar(sc, ax=axes, pad=0.01, label="line/continuum band flux")
fig.savefig(f"{S}/plots/lag_vs_L_frac_gt10.pdf", bbox_inches="tight")
