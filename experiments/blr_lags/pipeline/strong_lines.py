"""Line-to-continuum band-flux fraction from DR16Q broad-line EWs, and R-L statistics for strong-line subsets.

fraction = EW_obs * <T>_line / int T dlam, where <T>_line is the SDSS filter throughput averaged over a
Gaussian line profile with the catalog FWHM (so lines near filter edges are down-weighted properly).
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, speclite.filters
from scipy.stats import spearmanr, theilslopes
from jaxsedfit.mplstyle import style_path

S = str(Path(__file__).resolve().parent)
C_KMS = 2.998e5
props = pd.read_csv(f"{S}/line_props.csv", dtype={"objectId": str})
filt = {b: speclite.filters.load_filter(f"sdss2010-{b}") for b in "ugriz"}


def band_fraction(r):
    f = filt[r.band]
    lam_obs = r.peak_wave * (1 + r.z)
    sig = lam_obs * r.fwhm / C_KMS / 2.3548
    lam = np.linspace(lam_obs - 5 * sig, lam_obs + 5 * sig, 401)
    prof = np.exp(-0.5 * ((lam - lam_obs) / sig) ** 2)
    T_line = np.trapezoid(f(lam) * prof, lam) / np.trapezoid(prof, lam)
    w = f.wavelength
    return r.ew_rest * (1 + r.z) * T_line / np.trapezoid(f(w), w)


props["frac"] = [band_fraction(r) if np.isfinite(r.ew_rest) and r.good else np.nan for r in props.itertuples()]  # see dr16q_lines.line_quality
nmad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
plt.style.use(style_path())
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
out = []
for j, (line, lkey) in enumerate([("Halpha", "L5100"), ("Hbeta", "L5100"), ("MgII", "L3000"), ("CIV", "L3000")]):
    df = pd.read_csv(f"{S}/plots/{line}/{line}_profile_results.csv", dtype={"objectId": str})
    df = df.merge(props.loc[props.line == line, ["objectId", "frac", "ew_rest"]], on="objectId", how="left")
    df.to_csv(f"{S}/plots/{line}/{line}_profile_results_with_frac.csv", index=False)
    clean = ~df.multimodal & ~df.edge
    for thr in [0.0, 0.10, 0.20]:
        for lab, m in [("all", df.frac > thr), ("clean", (df.frac > thr) & clean)]:
            d = df[m]
            rho, p = spearmanr(d.logL, np.log10(d.lag)) if len(d) > 3 else (np.nan, np.nan)
            sl = theilslopes(np.log10(d.lag), d.logL) if len(d) > 3 else (np.nan,) * 4
            off = d.dex_from_ref.dropna()
            out.append(dict(line=line, frac_gt=thr, subset=lab, N=len(d), rho=rho, p=p, slope=sl[0], slope_lo=sl[2], slope_hi=sl[3],
                            offset=off.median() if len(off) else np.nan, nmad=nmad(off) if len(off) > 2 else np.nan))
    ax = axes[0, j]
    ax.hist(df.frac, bins=np.linspace(0, max(0.6, df.frac.max()), 25), color="0.5")
    for thr in (0.1, 0.2):
        ax.axvline(thr, ls="--", color="k", lw=1)
    ax.set_xlabel("line/continuum band flux")
    ax.set_ylabel("N")
    ax.set_title(f"{line} (median {df.frac.median():.2f})")
    ax = axes[1, j]
    yerr = np.log10([df.lag / df.lo, df.hi / df.lag])
    for m, kw in [(df.frac <= 0.1, dict(color="0.75", label="≤10%")), ((df.frac > 0.1) & (df.frac <= 0.2), dict(color="C0", label="10–20%")),
                  (df.frac > 0.2, dict(color="C3", label=">20%"))]:
        mk = m & clean
        ax.errorbar(df.logL[mk], np.log10(df.lag[mk]), yerr=yerr[:, mk], fmt="o", ms=5, lw=0.8, **kw)
        ax.errorbar(df.logL[m & ~clean], np.log10(df.lag[m & ~clean]), yerr=yerr[:, m & ~clean], fmt="x", ms=5, lw=0.5, alpha=0.6, color=kw["color"])
    if df.ref.notna().any():
        xs = np.linspace(df.logL.min(), df.logL.max(), 10)
        ref_a, ref_b = np.polyfit(df.logL, np.log10(df.ref), 1)
        ax.plot(xs, ref_a * xs + ref_b, "k-", lw=1.2, label="reference R–L")
    ax.set_xlabel(rf"$\log\,\lambda L_{{{lkey[1:]}}}$ (erg s$^{{-1}}$)")
    ax.set_ylabel(r"$\log\,\tau_{\rm rest}$ (days)")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{S}/plots/strong_line_selection.pdf")
res = pd.DataFrame(out)
res.to_csv(f"{S}/plots/strong_line_stats.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(2).to_string(index=False))
