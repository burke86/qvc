"""Lag-marginal posteriors (24-point lag profile) vs luminosity for one line.

Usage: python analyze_line.py <run_dir> <line>
"""
import glob, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R, LINE = sys.argv[1], sys.argv[2]
sel = pd.read_csv(f"{R}/selection.csv")
src = sel[sel.line == LINE].set_index("objectId")
REFS = {"Hbeta": (1.527, 0.533, "Hβ R–L, Bentz+13"), "Halpha": (1.527 + np.log10(1.5), 0.533, "Hα ≈ 1.5× Bentz+13 Hβ (approx.)"),
        "MgII": (1.67, 0.30, "Mg II R–L, approx. literature"), "CIV": None}
LUMN = {"Hbeta": "L5100", "Halpha": "L5100", "MgII": "L3000", "CIV": "L3000"}[LINE]
REF = REFS[LINE]
REF_A, REF_B = (REF[0], REF[1]) if REF else (np.nan, np.nan)
rows = []
for fn in sorted(glob.glob(f"{R}/profiles/{LINE}/*_profile.json")):
    d = json.load(open(fn)); oid = int(d["objectId"])
    grid, post = np.asarray(d["lag_rest_grid"]), np.asarray(d["posterior"])
    lo, med, hi = d["q_post"]
    # Posterior mass in each peak's basin: a second peak with >=20% of the mass marks a multimodal lag.
    peaks = sorted(d["peaks_post"])
    mass = []
    if len(peaks) > 1:
        cuts = [np.sqrt(peaks[i] * peaks[i + 1]) for i in range(len(peaks) - 1)]
        edges = [0] + cuts + [np.inf]
        dx = np.gradient(np.log10(grid))
        mass = [float(np.sum((post * dx)[(grid >= edges[i]) & (grid < edges[i + 1])])) for i in range(len(peaks))]
    multimodal = len(mass) > 1 and sorted(mass)[-2] >= 0.2
    kmax = int(np.argmax(post))
    edge = kmax <= 1 or kmax >= len(grid) - 2  # posterior piles up against the grid boundary
    logL = float(src.loc[oid, "logL"])
    rows.append(dict(objectId=oid, z=d["z"], logL=logL, lag=med, lo=lo, hi=hi, map_lag=d["map_lag_rest"],
                     peaks=", ".join(f"{p:.0f}" for p in peaks), peak_mass=", ".join(f"{m:.2f}" for m in mass) or "-",
                     multimodal=multimodal, edge=edge, ref=(10 ** (REF_A + REF_B * (logL - 44)) if REF else np.nan), t=d["t_total"],
                     width_dex=np.log10(hi / lo) / 2))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no results yet")
df["dex_from_ref"] = np.log10(df.lag / df.ref)
pd.set_option("display.width", 220)
show = df.assign(lag_68=df.apply(lambda r: f"{r.lag:.0f} [{r.lo:.0f},{r.hi:.0f}]", axis=1))
print(show[["objectId", "z", "logL", "lag_68", "map_lag", "ref", "dex_from_ref", "width_dex", "peaks", "peak_mass", "multimodal", "edge"]]
      .round(2).sort_values("logL").to_string(index=False))
nmad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
print(f"\nN={len(df)} | median offset {df.dex_from_ref.median():+.2f} dex, NMAD {nmad(df.dex_from_ref):.2f} | "
      f"multimodal lag posteriors: {int(df.multimodal.sum())} | edge-pinned: {int(df.edge.sum())} | median 68% half-width {df.width_dex.median():.2f} dex | median time {df.t.median():.0f}s")
if len(df) > 4:
    r = spearmanr(df.logL, np.log10(df.lag)); u = df[~df.multimodal & ~df.edge]; r2 = spearmanr(u.logL, np.log10(u.lag))
    print(f"Spearman lag vs L: all rho={r.correlation:+.2f} (p={r.pvalue:.2f}); unimodal & not edge-pinned rho={r2.correlation:+.2f} (p={r2.pvalue:.2f}, N={len(u)})")
    if REF:
        print(f"unimodal & not edge-pinned: median offset {u.dex_from_ref.median():+.2f} dex, NMAD {nmad(u.dex_from_ref):.2f}")

INK, MUTED, BLUE = "#0b0b0b", "#8a8985", "#2a78d6"
from jaxsedfit.mplstyle import style_path
plt.style.use(style_path())  # user's preferred style
fig, ax = plt.subplots(figsize=(8, 6))
L = np.linspace(df.logL.min() - 0.2, df.logL.max() + 0.2, 50)
if REF:
    ax.plot(L, REF_A + REF_B * (L - 44), color=INK, lw=1.5, label=REF[2])
    ax.fill_between(L, REF_A + REF_B * (L - 44) - 0.2, REF_A + REF_B * (L - 44) + 0.2, color=MUTED, alpha=0.15, lw=0, label="±0.2 dex")
ax.axhline(2.0, color=MUTED, ls=":", lw=1, label="lag prior center (100 d)")
for sel, mk, lab in ((~df.multimodal & ~df.edge, "o", "lag posterior median, 68%"), (df.multimodal & ~df.edge, "D", "multimodal lag posterior (2nd peak ≥20% mass)"), (df.edge, "x", "edge-pinned (posterior peaks at 3 or 900 d)")):
    flag = mk == "D"
    s = df[sel]
    if len(s):
        ax.errorbar(s.logL, np.log10(s.lag), yerr=[np.log10(s.lag / s.lo), np.log10(s.hi / s.lag)], fmt=mk, ms=6 if not flag else 7,
                    color=(MUTED if mk == "x" else BLUE), mec=INK if flag else (MUTED if mk == "x" else "#fcfcfb"), elinewidth=1.2, label=f"{lab} (n={len(s)})")
ax.set_xlabel(f"log {LUMN} [erg/s]"); ax.set_ylabel(f"log {LINE} BLR lag [rest days]")
ax.set_ylim(0.3, 3.1); ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title(f"{LINE}: S82 light curves, shared_latent_blr, lag-marginal Laplace", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/plots/{LINE}/{LINE}_lag_vs_{LUMN}.pdf")
df.to_csv(f"{R}/plots/{LINE}/{LINE}_profile_results.csv", index=False)
print(f"plot: {R}/plots/{LINE}/{LINE}_lag_vs_{LUMN}.pdf")
