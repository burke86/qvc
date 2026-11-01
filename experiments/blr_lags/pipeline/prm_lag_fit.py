"""Broad-line lag profile for one S82 quasar and line: EzTaoX MixedBLRKernel with a prior on the in-band ratio a.

Same model as the SDSS-RM validation (burke86/sdss_rm, rm_lag_profile.py), extended to five bands:
- a DRW driver x(t) normalized to the fractional continuum variability in the line band;
- each continuum band sees (lam_b / lam_line)^(-beta) x(t), so the other bands fix the continuum amplitude in the
  line band through a power law in wavelength;
- the line band sees x(t) / (1 + a) + r (psi * x)(t), with psi an Erlang response; a has a log-normal prior at the
  DR16Q in-band ratio (prm_select.py), width sqrt(0.1^2 + EW error^2) (0.1 for epoch-to-epoch changes in the ratio);
- one mean and one jitter per (band, survey);
- continuum bands whose rest-frame pivot is below LYA_CUT_REST (Lyman-alpha forest) are dropped: forest absorption
  breaks the continuum power law there.
Without the prior on a, a short echo is degenerate with the prompt continuum and the BLR term fits slow structure at
lags near the baseline (see the sdss_rm README). The lag posterior is the profile likelihood over a fixed rest-frame
lag grid (other parameters maximized at each lag) times a log-normal lag prior; no Laplace term.

Requires burke86/EzTaoX (branch mixed-blr-kernel) and the GitHub version of tinygp.
Run from the qvc repo root: python experiments/blr_lags/pipeline/prm_lag_fit.py <objectId> <line> <outdir>
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

oid, line, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
T0 = time.time()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qvcfit  # noqa: E402  (filtered reads, light-curve preparation)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from tinygp import GaussianProcess  # noqa: E402

from eztaox.kernels.quasisep import Exp, MixedBLRKernel  # noqa: E402

F = qvcfit.F
LAG_GRID_REST = np.geomspace(2.0, 900.0, 24)
LAG_PRIOR_CENTER_REST, LAG_PRIOR_SIGMA_DEX = 100.0, 1.0
ORDER = 3
MULTIMODAL_MASS = 0.2
EPOCH_RATIO_SCATTER = 0.1  # ln scatter of a single spectrum's line/continuum ratio about the mean
LYA_CUT_REST = 1300.0  # Angstrom

sel = pd.read_csv(Path(__file__).resolve().parent / "prm_selection.csv", dtype={"objectId": str}).set_index(["objectId", "line"])
row = sel.loc[(oid, line)]
obj, _, meta = qvcfit.build(oid)
_cat = pd.read_parquet(Path(qvcfit.D) / "Catalog.parquet", columns=["objectId", "qg_LogL3000"])
LOGL3000 = dict(zip(_cat.objectId.astype(str), _cat.qg_LogL3000))
bands, z = meta["bands"], meta["z"]
line_band = row.band
bl = bands.index(line_band)
B = len(bands)
onepz = 1.0 + z

t_all = np.asarray(obj["X"][0], float)
band_all = np.asarray(obj["X"][1], np.int32)
y_all = np.asarray(F.mag_residual_to_relative_flux(obj["y"]), float)
e_all = np.asarray(F.magerr_residual_to_relative_fluxerr(obj["y"], obj["yerr"]), float)
surv_all = np.asarray(obj["survey_idx"], np.int32)
dropped = [b_ for i, b_ in enumerate(bands) if i != bl and F.lambda_pivot[b_] / onepz < LYA_CUT_REST]
keep = ~np.isin(band_all, [bands.index(b_) for b_ in dropped])
t_all, band_all, y_all, e_all, surv_all = (a[keep] for a in (t_all, band_all, y_all, e_all, surv_all))
order = np.lexsort((band_all, t_all))
t, band, y, yerr, surv = (jnp.asarray(a[order]) for a in (t_all, band_all, y_all, e_all, surv_all))
keys = [(int(b_), int(s_)) for b_, s_ in zip(np.asarray(band), np.asarray(surv))]
gnames = sorted(set(keys))
grp = jnp.asarray([gnames.index(k) for k in keys])
G = len(gnames)
jitter_loc = np.log([0.3 * np.median(np.asarray(yerr)[np.asarray(grp) == g]) for g in range(G)])
lam = np.array([F.lambda_pivot[b_] for b_ in bands], float)
lam_ratio = jnp.asarray(lam / lam[bl])
is_line = jnp.arange(B) == bl

log_a0 = float(np.log(row.frac))
a_width = float(np.hypot(EPOCH_RATIO_SCATTER, row.frac_lnerr))
norm = jax.scipy.stats.norm.logpdf
names = ["log_tau", "log_sigma", "beta", "log_amp_blr", "log_a"]
names += [f"mean_{bands[b_]}_{obj['survey_names'][s_]}" for b_, s_ in gnames]
names += [f"logjit_{bands[b_]}_{obj['survey_names'][s_]}" for b_, s_ in gnames]


def neg_log_post(theta, lag_obs):
    beta, a = theta[2], jnp.exp(theta[4])
    amp_cont = jnp.where(is_line, 1.0 / (1.0 + a), lam_ratio ** (-beta))
    amp_blr = jnp.where(is_line, jnp.exp(theta[3]), 0.0)
    kernel = MixedBLRKernel(params={"lag": lag_obs, "amp_cont": amp_cont, "amp_blr": amp_blr},
                            kernel=Exp(scale=jnp.exp(theta[0]), sigma=jnp.exp(theta[1])), order=ORDER)
    means, logjit = theta[5:5 + G], theta[5 + G:5 + 2 * G]
    gp = GaussianProcess(kernel, (t, band), diag=yerr ** 2 + jnp.exp(2 * logjit)[grp] + 1e-10, assume_sorted=True)
    lp = gp.log_probability(y - means[grp])
    lp += norm(theta[0], jnp.log(10 ** 2.5 * onepz), 1.2 * jnp.log(10.0))
    lp += norm(theta[1], jnp.log(0.1), jnp.log(10.0))
    lp += norm(beta, 1.0, 1.0)
    lp += norm(theta[3], -1.0, 1.0)
    lp += norm(theta[4], log_a0, a_width)
    lp += jnp.sum(norm(means, 0.0, 0.2))
    lp += jnp.sum(norm(logjit, jnp.asarray(jitter_loc), 0.5))
    return -lp


vg = jax.jit(jax.value_and_grad(neg_log_post))
bounds = ([(np.log(10 * onepz), np.log(1e4 * onepz)), (-8, 3), (-3, 5), (-6, 3), (log_a0 - 5 * a_width, log_a0 + 5 * a_width)]
          + [(-2, 2)] * G + [(-12, 0)] * G)


def fit(lag_obs, starts):
    best = None
    for s in starts:
        def fun(x):
            v, g = vg(jnp.asarray(x), lag_obs)
            return float(v), np.asarray(g)
        r = minimize(fun, s, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 800, "ftol": 1e-11})
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best


def base(tau):
    return np.array([np.log(tau), np.log(max(float(np.std(y_all[band_all == bl])), 0.02)), 1.0, -1.0, log_a0]
                    + [0.0] * G + list(jitter_loc))


U, maps, prev = [], [], None
for lag_rest in LAG_GRID_REST:
    lag_obs = lag_rest * onepz
    r = fit(lag_obs, [base(300.0), base(3000.0)] if prev is None else [prev, base(500.0)])
    prev = r.x
    U.append(float(r.fun)); maps.append(r.x.tolist())
U = np.array(U)
loglag = np.log10(LAG_GRID_REST)
log_prior = -0.5 * ((loglag - np.log10(LAG_PRIOR_CENTER_REST)) / LAG_PRIOR_SIGMA_DEX) ** 2
logp = -U + log_prior
post = np.exp(logp - logp.max()); post /= post.sum()
lik = np.exp(-U - (-U).max()); lik /= lik.sum()


def quantiles(w):
    c = np.cumsum(w) - 0.5 * w
    return [float(10 ** np.interp(q, c, loglag)) for q in (0.16, 0.5, 0.84)]


def peaks(w):
    out = []
    for i in [i for i in range(len(w)) if (i == 0 or w[i] >= w[i - 1]) and (i == len(w) - 1 or w[i] > w[i + 1])]:
        lo = i
        while lo > 0 and w[lo - 1] <= w[lo]:
            lo -= 1
        hi = i
        while hi < len(w) - 1 and w[hi + 1] <= w[hi]:
            hi += 1
        out.append((float(LAG_GRID_REST[i]), float(w[lo:hi + 1].sum())))
    return sorted(out, key=lambda p: -p[1])


pk = peaks(post)
res = dict(objectId=oid, line=line, z=z, line_band=line_band, dropped_bands=dropped, frac=float(row.frac), frac_prior_width=a_width,
           logL5100=float(obj["LOGL5100"]), logL3000=float(LOGL3000.get(oid, np.nan)), bands=bands, n_points=int(t.size), groups=[names[5 + g][5:] for g in range(G)],
           lag_rest_grid=LAG_GRID_REST.tolist(), U=U.tolist(), posterior=post.tolist(), prior_divided=lik.tolist(),
           q_post_rest=quantiles(post), q_prior_divided_rest=quantiles(lik), peaks_post=pk,
           multimodal=bool(len(pk) > 1 and pk[1][1] >= MULTIMODAL_MASS),
           edge=bool(np.argmax(post) in (0, len(post) - 1)), map_theta=maps, theta_names=names, t_total=time.time() - T0)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{oid}_{line}.json").write_text(json.dumps(res))
q = res["q_post_rest"]
print(f"DONE {oid} {line} z={z:.3f} {line_band} a={row.frac:.2f}: rest lag {q[1]:.1f} [{q[0]:.1f},{q[2]:.1f}] d | "
      f"multimodal {res['multimodal']} edge {res['edge']} | {res['t_total']:.0f}s", flush=True)
