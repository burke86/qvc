"""MAP + Laplace fit of one S82 quasar with the default QVC model (shared_latent_blr), seeing terms on.

Input preparation mirrors fit_light_curves.main() (concat_light_curves ->
populate_sdss_fields -> make_lc -> jitter grids -> flux-linearized model with
refinement_iters=1), using the fused fast solver. MAP: 4 starts (init_to_median +
3 unconstrained perturbations), each L-BFGS-B in coordinates whitened by a
finite-difference Hessian at the start. Every start's optimum and lags are kept
so multimodality is visible. Laplace: FD Hessian at the best optimum.
Usage: python laplace_object.py <objectId> <outdir>
"""

from pathlib import Path
import json
import resource
import sys
import time

import numpy as np
import pandas as pd

oid, outdir = sys.argv[1], sys.argv[2]
T_START = time.time()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qvcfit  # noqa: E402  (filtered reads, scratch seeing sidecars, shared model construction)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro.infer import init_to_median  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

obj, model, meta = qvcfit.build(oid, seeing=True)
bands, z, B, bidx = meta["bands"], meta["z"], meta["B"], meta["bidx"]
t_load = time.time() - T_START

info = initialize_model(jax.random.PRNGKey(0), model, init_strategy=init_to_median())
z_ref, unravel = ravel_pytree(info.param_info.z)
jax.clear_caches()
potf = lambda x: info.potential_fn(unravel(x))
vg = jax.jit(jax.value_and_grad(potf))
grad = jax.jit(jax.grad(potf))
post = jax.jit(lambda x: info.postprocess_fn(unravel(x)))
jax.block_until_ready(vg(z_ref))
t_setup = time.time() - T_START - t_load


def f(x):
    v, g = vg(jnp.asarray(x))
    v = float(v)
    g = np.asarray(g, dtype=float)
    if not (np.isfinite(v) and np.all(np.isfinite(g))):
        return 1e10, np.zeros_like(g)
    return v, g


def fd_hess(x):
    h = 1e-4 * np.maximum(1.0, np.abs(x))
    H = np.empty((x.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h[j]
        H[:, j] = (np.asarray(grad(jnp.asarray(x + e))) - np.asarray(grad(jnp.asarray(x - e)))) / (2 * h[j])
    return 0.5 * (H + H.T)


def summary(x):
    p = post(jnp.asarray(x))
    out = {"lag_rest": (np.asarray(p["lag_blr"]) / (1 + z)).tolist()}
    for k in ("log_tau_uv", "log_sigma_uv"):
        if k in p:
            out[k] = float(np.ravel(np.asarray(p[k]))[0])
    if "amp_blr_relflux" in p and "amp_cont_relflux" in p:
        out["blr_fraction"] = (np.asarray(p["amp_blr_relflux"]) / np.asarray(p["amp_cont_relflux"])).tolist()
    return out


rng = np.random.default_rng(1)
starts = [np.asarray(z_ref)] + [np.asarray(z_ref) + rng.normal(0, 1.0, z_ref.size) for _ in range(3)]
optima = []
for i, x0 in enumerate(starts):
    t0 = time.time()
    w, V = np.linalg.eigh(fd_hess(x0))
    Lm = V / np.sqrt(np.maximum(np.abs(w), 1e-3))
    fu = lambda u: (lambda v, g: (v, Lm.T @ g))(*f(x0 + Lm @ u))
    r = minimize(fu, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=3000, maxcor=20, gtol=1e-6, ftol=1e-12))
    x = x0 + Lm @ r.x
    optima.append(dict(start=i, U=float(r.fun), nfev=int(r.nfev), seconds=time.time() - t0, x=x, **summary(x)))
    print(f"[{oid}] start {i}: U={r.fun:.3f} nfev={r.nfev} {time.time() - t0:.0f}s "
          f"lags={np.round(optima[-1]['lag_rest'], 1).tolist()}", flush=True)

finite = [o for o in optima if o["U"] < 1e9]
best = min(finite, key=lambda o: o["U"])
H = fd_hess(best["x"])
evals, evecs = np.linalg.eigh(H)
cov = (evecs / np.maximum(evals, 1e-6)) @ evecs.T
draws = np.random.default_rng(0).multivariate_normal(best["x"], cov, size=400)
S = [summary(d) for d in draws]
lag = np.array([s["lag_rest"] for s in S])
frac = np.array([s["blr_fraction"] for s in S]) if "blr_fraction" in S[0] else None
tau = np.array([s.get("log_tau_uv", np.nan) for s in S])

# Distinct modes: optima whose r-band-independent lag vectors differ by > 0.1 dex in any band.
modes = []
for o in sorted(finite, key=lambda o: o["U"]):
    ll = np.log10(np.asarray(o["lag_rest"]))
    if all(np.max(np.abs(ll - np.log10(np.asarray(m["lag_rest"])))) > 0.1 for m in modes):
        modes.append(o)

row = dict(
    objectId=oid, z=z, bands=bands, n_points=int(len(obj["y"])),
    n_per_band={b: int(np.sum(np.asarray(bidx) == i)) for i, b in enumerate(bands)},
    U_best=best["U"], hessian_min_eig=float(evals[0]), hessian_nonpos=int(np.sum(evals <= 0)),
    lag_rest_q16=np.quantile(lag, 0.16, axis=0).tolist(), lag_rest_q50=np.quantile(lag, 0.5, axis=0).tolist(),
    lag_rest_q84=np.quantile(lag, 0.84, axis=0).tolist(), lag_rest_map=best["lag_rest"],
    blr_fraction_q50=(np.quantile(frac, 0.5, axis=0).tolist() if frac is not None else None),
    log_tau_uv_q16=float(np.nanquantile(tau, 0.16)), log_tau_uv_q50=float(np.nanquantile(tau, 0.5)),
    log_tau_uv_q84=float(np.nanquantile(tau, 0.84)),
    n_modes=len(modes), seeing_groups=meta["seeing_groups"], n_points_with_seeing=meta["n_points_with_seeing"],
    modes=[dict(U=m["U"], dU=m["U"] - best["U"], lag_rest=m["lag_rest"], blr_fraction=m.get("blr_fraction"),
                log_tau_uv=m.get("log_tau_uv")) for m in modes],
    starts=[dict(U=o["U"], nfev=o["nfev"], seconds=o["seconds"], lag_rest=o["lag_rest"]) for o in optima],
    t_load=t_load, t_setup=t_setup, t_total=time.time() - T_START,
    peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
)
with open(f"{outdir}/{oid}.json", "w") as fh:
    json.dump(row, fh)
np.savez(f"{outdir}/{oid}_laplace.npz", x_map=best["x"], cov=cov, lag_draws=lag,
         frac_draws=(frac if frac is not None else np.array([])), log_tau_uv_draws=tau)
print(f"[{oid}] DONE z={z:.3f} N={row['n_points']} modes={len(modes)} total {row['t_total']:.0f}s "
      f"peak {row['peak_rss_gb']:.2f} GB", flush=True)
