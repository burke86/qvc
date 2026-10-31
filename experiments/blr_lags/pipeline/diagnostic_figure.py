"""Per-source BLR-lag diagnostic figure.

Panels: (left) each band's light curve with the MAP model conditioned on all bands
(mean +/- 1 sigma, from the model's own marginal prediction); (right top) the
lag-marginal Laplace posterior of the line-band lag with the model prior, the
prior-divided posterior, the MAP and the R-L expectation; (right bottom) the SDSS
spectrum over SDSS filter curves with broad-line positions at the source redshift.
Usage: python diagnostic_figure.py <objectId> <line: Halpha|Hbeta|MgII|CIV> <band> <logL> <map_npz> <profile_json> <outdir>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

oid, line, line_band, logL, map_npz, prof_json, outdir = sys.argv[1:8]
logL = float(logL)
REFS = {  # (A, B, luminosity label, legend label); None = no reference relation drawn
    "Hbeta": (1.527, 0.533, "L5100", "Hβ R–L (Bentz+13)"),
    "Halpha": (1.527 + np.log10(1.5), 0.533, "L5100", "Hα ≈ 1.5× Bentz+13 Hβ (approx.)"),
    "MgII": (1.67, 0.30, "L3000", "Mg II R–L (approx.)"),
    "CIV": None,
}
LUM = {"Hbeta": "L5100", "Halpha": "L5100", "MgII": "L3000", "CIV": "L3000"}
REF = REFS[line]
T0 = time.time()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qvcfit  # noqa: E402  (filtered reads, shared model construction)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro import handlers  # noqa: E402
from numpyro.infer import init_to_median  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402

from qvc.light_curve import multiband_model_shared_latent_blr as SL  # noqa: E402
from qvc.light_curve.multiband_dho_core import relative_flux_to_mag_residual  # noqa: E402
from qvc.spectra.build_local_cache import load_sdss_spec_from_cache, sdss_cache_file_path  # noqa: E402

F = qvcfit.F
obj, model, meta = qvcfit.build(oid)
bands, z, B = meta["bands"], meta["z"], meta["B"]
bidx, survey_idx = meta["bidx"], meta["survey_idx"]
info = initialize_model(jax.random.PRNGKey(0), model, init_strategy=init_to_median())
_, unravel = ravel_pytree(info.param_info.z)
x_map = jnp.asarray(np.load(map_npz)["x_map"])
constrained = info.postprocess_fn(unravel(x_map))

# Run the model once at the MAP and capture the exact GP model object and parameters it scores.
captured = {}
_orig_log_prob = SL.SharedLatentDiskBLRRelativeFluxModel.log_prob


def _capture(self, params):
    captured["m"], captured["params"] = self, params
    return _orig_log_prob(self, params)


SL.SharedLatentDiskBLRRelativeFluxModel.log_prob = _capture
handlers.trace(handlers.substitute(handlers.seed(model, 0), data=constrained)).get_trace()
SL.SharedLatentDiskBLRRelativeFluxModel.log_prob = _orig_log_prob
m, params = captured["m"], captured["params"]

# Data in model coordinates: survey offsets removed, converted back to magnitude residuals.
t_sorted, b_sorted = (np.asarray(v) for v in m.X)
inds = jnp.argsort(jnp.asarray(t_sorted))
y_corr_rel = np.asarray(m._observed_y_sorted(params, inds))
t_s, b_s = t_sorted[np.asarray(inds)], b_sorted[np.asarray(inds)]
y_corr_mag = np.asarray(relative_flux_to_mag_residual(y_corr_rel))
# The model stores X sorted by a stable argsort of time, then sorts again (stably) in the likelihood;
# apply the same two permutations so errors and survey labels stay matched to their points.
order_obj = np.argsort(np.asarray(obj["X"][0]), kind="stable")[np.asarray(inds)]
assert np.array_equal(np.asarray(obj["X"][0])[order_obj], t_s) and np.array_equal(bidx[order_obj], b_s)
err_mag = np.asarray(obj["yerr"], dtype=float)[order_obj]
surv_s = survey_idx[order_obj]

# Conditional model on a dense grid per band.
t_lo, t_hi = t_s.min() - 30, t_s.max() + 30
grid = np.linspace(t_lo, t_hi, 2500)
preds = {}
for k in range(B):
    Xp = (jnp.asarray(grid), jnp.full(grid.size, k, dtype=jnp.int32))
    mu, sd = m.prediction_to_display(m.pred(params, Xp))
    preds[k] = (np.asarray(mu), np.asarray(sd))

# Continuum-only conditional model per band: virtual band B+k observes band k's disk (continuum)
# endpoint alone (BLR loading zeroed); virtual band 2B+k observes its BLR endpoint alone. Both are
# linear in the same posterior, so band model = continuum + BLR exactly and the gap between the
# curves is the BLR term's contribution. Real bands are unchanged.
#
# These are computed with marginal_prediction directly, not m.pred: m.pred is eqx.filter_jit-ed and
# was already traced for the real bands above, so a cached trace would silently ignore the patched
# observation model (this produced wrong overlays before). The additivity check below guards it.
from qvc.light_curve.quasisep_prediction import marginal_prediction  # noqa: E402

_orig_obs = SL.SharedLatentDiskBLRQS._observation_model_with_stds


def _obs_with_components(self, X, stds):
    t_, band_ = X
    band_ = jnp.asarray(band_, dtype=jnp.int32)
    k_ = band_ % B
    h = _orig_obs(self, (t_, k_), stds)
    disk_endpoint = 2 + (k_ + 1) * int(self.disk_order) - 1
    blr_endpoint = 2 + B * int(self.disk_order) + (k_ + 1) * int(self.blr_order) - 1
    h_cont = h.at[blr_endpoint].set(0.0)
    h_blr = h.at[disk_endpoint].set(0.0)
    return jnp.where(band_ >= 2 * B, h_blr, jnp.where(band_ >= B, h_cont, h))


SL.SharedLatentDiskBLRQS._observation_model_with_stds = _obs_with_components
mean_fn = jax.vmap(lambda Xi: m.get_mean(m.zero_mean, params, Xi))
gp_c, inds_c = m._build_gp(params)
y_obs = m._observed_y_sorted(params, inds_c)
lag_disk_obs = np.asarray(params["lag_disk"])  # observed-frame disk lag per band
continuum = {}
for k in range(B):
    Xk = (jnp.asarray(grid), jnp.full(grid.size, k, dtype=jnp.int32))
    Xc = (jnp.asarray(grid), jnp.full(grid.size, B + k, dtype=jnp.int32))
    Xb = (jnp.asarray(grid), jnp.full(grid.size, 2 * B + k, dtype=jnp.int32))
    tot, _ = marginal_prediction(gp_c, y_obs, Xk)
    cont, _ = marginal_prediction(gp_c, y_obs, Xc)
    blr, _ = marginal_prediction(gp_c, y_obs, Xb)
    # Every prediction carries a mean function; total = mean + K*a, components = K*a pieces.
    resid = np.max(np.abs(np.asarray((tot - mean_fn(Xk)) - (cont - mean_fn(Xc)) - (blr - mean_fn(Xb)))))
    scale = float(np.max(np.abs(np.asarray(tot - mean_fn(Xk)))))
    assert resid <= 1e-8 * max(scale, 1e-12), f"band {bands[k]}: total != continuum + BLR (max diff {resid:.2e})"
    assert np.allclose(np.asarray(tot), np.asarray(m.pred(params, Xk)[0]), atol=1e-8), "direct vs m.pred mismatch"
    continuum[k] = np.asarray(relative_flux_to_mag_residual(cont - mean_fn(Xc) + mean_fn(Xk)))
SL.SharedLatentDiskBLRQS._observation_model_with_stds = _orig_obs

# Lag posterior from the profile run.
prof = json.load(open(prof_json))
lag_grid = np.asarray(prof["lag_rest_grid"])
post = np.asarray(prof["posterior"])
like = np.asarray(prof["prior_divided"])
prior = np.exp(-0.5 * ((np.log10(lag_grid) - 2.0) / 1.0) ** 2)
prior /= np.sum(prior * np.gradient(np.log10(lag_grid)))
ref_lag = 10 ** (REF[0] + REF[1] * (logL - 44.0)) if REF else None

# SDSS spectrum (pipeline cache layout; download on miss, as fit_spectra.fetch_spectrum_fits does).
spec_note = ""
cache = Path(__file__).resolve().parents[3] / "data" / "spectra_cache"
cache.mkdir(parents=True, exist_ok=True)
plate, mjd, fiber = int(obj["plate"]), int(obj["mjd"]), int(obj["fiberid"])
hdul = load_sdss_spec_from_cache(cache, plate, mjd, fiber)
if hdul is None:
    try:
        from astroquery.sdss import SDSS
        got = SDSS.get_spectra(plate=plate, fiberID=fiber, mjd=mjd, data_release=17)
        if got:
            got[0].writeto(sdss_cache_file_path(cache, plate, mjd, fiber), overwrite=True)
            hdul = load_sdss_spec_from_cache(cache, plate, mjd, fiber)
    except Exception as exc:  # network or archive failure: draw the figure without the spectrum
        spec_note = f"spectrum unavailable ({type(exc).__name__})"
lam = flux = None
if hdul is not None:
    tb = hdul[1].data
    lam = 10 ** np.asarray(tb["loglam"], dtype=float)
    flux = np.asarray(tb["flux"], dtype=float)
    ivar = np.asarray(tb["ivar"], dtype=float)
    flux = np.where(ivar > 0, flux, np.nan)
    hdul.close()
elif not spec_note:
    spec_note = "spectrum unavailable"

# ---- figure ----
lo, med, hi = prof["q_post"]
flags = []
if prof.get("hessian_nonpos") and any(prof["hessian_nonpos"]):
    flags.append("non-PD Hessians")
kmax = int(np.argmax(post))
if kmax <= 1 or kmax >= len(post) - 2:
    flags.append("EDGE-PINNED")
if len(prof["peaks_post"]) > 1:
    flags.append(f"peaks at {', '.join(f'{p:.0f}' for p in prof['peaks_post'])} d")
TITLE = (f"{oid}  z={z:.3f}  log {LUM[line]}={logL:.2f}  |  {line} in {line_band}  |  lag {med:.0f} [{lo:.0f}, {hi:.0f}] d"
         + (f" (R–L {ref_lag:.0f} d)" if REF else "") + (f"  |  {'; '.join(flags)}" if flags else ""))

INK, MUTED = "#0b0b0b", "#8a8985"
SURVEY_COLORS = {0: ("#2a78d6", "SDSS"), 1: ("#eda100", "PS1"), 2: ("#eb6834", "ZTF")}
BAND_COLORS = {"u": "#6250d6", "g": "#1baf7a", "r": "#eb6834", "i": "#e34948", "z": "#52514e"}
from jaxsedfit.mplstyle import style_path  # noqa: E402

plt.style.use(style_path())  # user's preferred style
fig = plt.figure(figsize=(16, 2.2 * B + 1.5))
gs = fig.add_gridspec(B, 2, width_ratios=[2.3, 1], hspace=0.12, wspace=0.18)
for k, band in enumerate(bands):
    ax = fig.add_subplot(gs[k, 0])
    mu, sd = preds[k]
    is_line = band == line_band
    ax.fill_between(grid, mu - sd, mu + sd, color=BAND_COLORS.get(band, MUTED), alpha=0.18, lw=0)
    ax.plot(grid, mu, color=BAND_COLORS.get(band, MUTED), lw=1.0, label="model (disk + BLR responses)" if k == 0 else None)
    ax.plot(grid - lag_disk_obs[k], continuum[k], color=INK, lw=0.7, alpha=0.75,
            label="continuum-only model, shifted to zero lag" if k == 0 else None)
    for s, (c, lab) in SURVEY_COLORS.items():
        sel = (b_s == k) & (surv_s == s)
        if sel.any():
            ax.errorbar(t_s[sel], y_corr_mag[sel], err_mag[sel], fmt="o", ms=2.2, color=c, elinewidth=0.5, alpha=0.8,
                        label=lab if k == 0 else None)
    ax.invert_yaxis()
    ax.set_ylabel(f"{band}  Δmag" + ("  ← line band" if is_line else ""), color=INK if not is_line else "#c4302b")
    if k < B - 1:
        ax.set_xticklabels([])
    if k == 0:
        ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower left")
        ax.set_title(TITLE + "\nlight curves (survey offsets removed); MAP model conditioned on all bands (mean ± 1σ)",
                     loc="left", fontsize=9)
ax.set_xlabel("days since first epoch (observed frame)")

right = gs[:, 1].subgridspec(2, 1, hspace=0.42)
axp = fig.add_subplot(right[0])
axp.plot(np.log10(lag_grid), post, color="#2a78d6", lw=1.8, label="posterior (lag-marginal Laplace)")
axp.plot(np.log10(lag_grid), like, color="#2a78d6", lw=1.0, ls="--", label="prior divided out")
axp.plot(np.log10(lag_grid), prior, color=MUTED, lw=1.0, ls=":", label="model lag prior")
if REF:
    axp.axvline(np.log10(ref_lag), color=INK, lw=1.2, label=f"{REF[3]}: {ref_lag:.0f} d")
axp.axvline(np.log10(prof["map_lag_rest"]), color="#eb6834", lw=1.0, ls="-.", label=f"MAP: {prof['map_lag_rest']:.0f} d")
lo, med, hi = prof["q_post"]
axp.axvspan(np.log10(lo), np.log10(hi), color="#2a78d6", alpha=0.08, lw=0)
axp.set_xlabel(f"log {line} ({line_band}-band) BLR lag [rest days]")
axp.set_ylabel("density per dex")
axp.set_title(f"Lag posterior: {med:.0f} [{lo:.0f}, {hi:.0f}] d", loc="left", fontsize=9)
axp.legend(fontsize=7.5, frameon=False, loc="upper left")

axs = fig.add_subplot(right[1])
import speclite.filters as sf  # noqa: E402

filt = sf.load_filters(*[f"sdss2010-{b}" for b in "ugriz"])
fmax = np.nanpercentile(flux, 99) if flux is not None else 1.0
for fcurve, b in zip(filt, "ugriz"):
    wl = np.asarray(fcurve.wavelength)
    tr = np.asarray(fcurve.response)
    axs.fill_between(wl, 0, tr / tr.max() * fmax * 0.9, color=BAND_COLORS[b], alpha=0.12 if b != line_band else 0.25, lw=0)
    axs.text(wl[np.argmax(tr)], fmax * 0.93, b, color=BAND_COLORS[b], ha="center", fontsize=9)
if flux is not None:
    sm = pd.Series(flux).rolling(5, center=True, min_periods=1).median().to_numpy()
    axs.plot(lam, sm, color=INK, lw=0.6)
else:
    axs.text(0.5, 0.5, spec_note, transform=axs.transAxes, ha="center", color=MUTED)
for name, w in (("Lyα", 1216), ("C IV", 1549), ("C III]", 1909), ("Mg II", 2798), ("Hγ", 4341), ("Hβ", 4861),
                ("[O III]", 5007), ("Hα", 6563)):
    wo = w * (1 + z)
    if 3000 < wo < 10500:
        hl = {"Hα": "Halpha", "Hβ": "Hbeta", "Mg II": "MgII", "C IV": "CIV"}.get(name) == line
        axs.axvline(wo, color="#c4302b" if hl else MUTED, lw=1.2 if hl else 0.7, ls="-" if hl else ":")
        axs.text(wo, fmax * 0.02, name, rotation=90, fontsize=7.5, color="#c4302b" if hl else "#52514e", va="bottom", ha="right")
axs.set_xlim(3000, 10500)
axs.set_ylim(0, fmax)
axs.set_xlabel("observed wavelength [Å]")
axs.set_ylabel("flux [1e-17 erg/s/cm²/Å]")
axs.set_title(f"SDSS spectrum {plate}-{mjd}-{fiber}; SDSS filter curves (ZTF/PS1 similar)", loc="left", fontsize=9)

out = Path(outdir)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{oid}_{line}_diagnostic.pdf", bbox_inches="tight")
print(f"[{oid}] saved {out / f'{oid}_{line}_diagnostic.pdf'} in {time.time() - T0:.0f}s {spec_note}", flush=True)
