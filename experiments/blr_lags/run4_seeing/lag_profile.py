"""Lag-marginal Laplace (INLA-style) posterior for one band's BLR lag.

For each rest-frame lag on a log grid, the line-band lag is held fixed and every
other parameter is re-optimized (warm-started along the grid, two passes) and
integrated out with a Laplace approximation:

    log p(L) ~ -U(y_L, L) + log|du_b/dL| - 0.5 log det H_y(y_L)

where L is the log observed-frame lag, y the remaining unconstrained parameters,
u_b the unconstrained coordinate of delta_log_lag_blr_raw[b] (the model sets
lag_blr[b] = exp(delta_b + log lag0)), and H_y the Hessian of U in y.
Usage: python lag_profile.py <objectId> <band> <map_npz> <outdir>
"""

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

oid, line_band, map_npz, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
T0 = time.time()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qvcfit  # noqa: E402  (filtered reads, scratch seeing sidecars, shared model construction)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro.distributions import constraints  # noqa: E402
from numpyro.distributions.transforms import biject_to  # noqa: E402
from numpyro.infer import init_to_median  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

F = qvcfit.F
obj, model, meta = qvcfit.build(oid, seeing=True)
bands, z, B = meta["bands"], meta["z"], meta["B"]
b = bands.index(line_band)
info = initialize_model(jax.random.PRNGKey(0), model, init_strategy=init_to_median())
z_ref, unravel = ravel_pytree(info.param_info.z)
jax.clear_caches()

# Flat indices of the pinned delta_b and of lag0 (ravel_pytree orders dict keys).
offsets, start = {}, 0
for key in sorted(info.param_info.z):
    size = int(np.size(info.param_info.z[key]))
    offsets[key] = (start, size)
    start += size
i_delta = offsets["delta_log_lag_blr_raw"][0] + b
i_lag0 = offsets["lag0"][0]
n = z_ref.size
free = np.array([i for i in range(n) if i != i_delta])

one_plus_z = 1.0 + z
LAG0_SUPPORT = biject_to(constraints.interval(0.0, F.LAG0_HIGH * one_plus_z))
LOG_LO = float(F.LOG_LAG_BLR_LOW + np.log(one_plus_z))
LOG_HI = float(F.LOG_LAG_BLR_HIGH + np.log(one_plus_z))


def u_delta(y_full, L):
    """Unconstrained delta_b that makes the band's observed log lag equal L."""
    log_lag0 = jnp.log(LAG0_SUPPORT(y_full[i_lag0]))
    t = biject_to(constraints.interval(LOG_LO - log_lag0, LOG_HI - log_lag0))
    return t.inv(L - log_lag0)


def full_from(y, L):
    y_full = jnp.zeros(n).at[free].set(y)
    return y_full.at[i_delta].set(u_delta(y_full, L))


pot = lambda x: info.potential_fn(unravel(x))
prof = jax.jit(lambda y, L: pot(full_from(y, L)))
prof_vg = jax.jit(jax.value_and_grad(lambda y, L: pot(full_from(y, L))))
prof_grad = jax.jit(jax.grad(lambda y, L: pot(full_from(y, L))))
log_jac = jax.jit(lambda y, L: jnp.log(jnp.abs(jax.grad(lambda LL: u_delta(jnp.zeros(n).at[free].set(y), LL))(L))))
post = jax.jit(lambda x: info.postprocess_fn(unravel(x)))

# ---- self-check at the saved MAP ----
x_map = np.load(map_npz)["x_map"]
assert x_map.size == n, (x_map.size, n)
p_map = post(jnp.asarray(x_map))
L_map = float(np.log(np.asarray(p_map["lag_blr"])[b]))
recon = np.asarray(full_from(jnp.asarray(x_map[free]), L_map))
U_map = float(pot(jnp.asarray(x_map)))
print(f"[{oid}] self-check: max|recon - x_map| = {np.max(np.abs(recon - x_map)):.2e}; "
      f"U_map {U_map:.4f} vs profile {float(prof(jnp.asarray(x_map[free]), L_map)):.4f}; "
      f"MAP rest lag {np.exp(L_map) / one_plus_z:.1f} d", flush=True)


def fd_hess(y, L):
    h = 1e-4 * np.maximum(1.0, np.abs(y))
    H = np.empty((y.size, y.size))
    for j in range(y.size):
        e = np.zeros_like(y)
        e[j] = h[j]
        H[:, j] = (np.asarray(prof_grad(jnp.asarray(y + e), L)) - np.asarray(prof_grad(jnp.asarray(y - e), L))) / (2 * h[j])
    return 0.5 * (H + H.T)


w, V = np.linalg.eigh(fd_hess(x_map[free], L_map))
Lw = V / np.sqrt(np.maximum(np.abs(w), 1e-3))  # whitening from the MAP Hessian


def optimize(y0, L):
    def fu(u):
        v, g = prof_vg(jnp.asarray(y0 + Lw @ u), L)
        v, g = float(v), np.asarray(g, dtype=float)
        if not (np.isfinite(v) and np.all(np.isfinite(g))):
            return 1e10, np.zeros_like(u)
        return v, Lw.T @ g
    r = minimize(fu, np.zeros(y0.size), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=1500, maxcor=20, gtol=1e-6, ftol=1e-12))
    return y0 + Lw @ r.x, float(r.fun), int(r.nfev)


lag_rest = np.exp(np.linspace(np.log(3.0), np.log(900.0), 24))  # 0.1 dex steps
Lgrid = np.log(lag_rest * one_plus_z)
k0 = int(np.argmin(np.abs(Lgrid - L_map)))
best = [None] * len(Lgrid)  # (U, y)
nfev_total = 0


def visit(k, y0):
    global nfev_total
    y, U, nf = optimize(y0, Lgrid[k])
    nfev_total += nf
    if best[k] is None or U < best[k][0]:
        best[k] = (U, y)


# Pass 1: outward from the MAP lag. Pass 2: back inward from each end.
visit(k0, x_map[free])
for k in range(k0 + 1, len(Lgrid)):
    visit(k, best[k - 1][1])
for k in range(k0 - 1, -1, -1):
    visit(k, best[k + 1][1])
t_pass1 = time.time() - T0
for k in range(len(Lgrid) - 2, -1, -1):
    visit(k, best[k + 1][1])
for k in range(1, len(Lgrid)):
    visit(k, best[k - 1][1])
print(f"[{oid}] two optimization passes done: {nfev_total} evals, {time.time() - T0:.0f}s", flush=True)

U = np.array([best[k][0] for k in range(len(Lgrid))])
logdet = np.empty(len(Lgrid))
nonpos = np.zeros(len(Lgrid), dtype=int)
logjac = np.empty(len(Lgrid))
DU_SKIP = 15.0  # profile points this far above the minimum carry negligible posterior mass
skip = U - U.min() > DU_SKIP
for k in range(len(Lgrid)):
    if skip[k]:
        logdet[k], logjac[k] = 0.0, 0.0
        continue
    ev = np.linalg.eigvalsh(fd_hess(best[k][1], Lgrid[k]))
    nonpos[k] = int(np.sum(ev <= 0))
    logdet[k] = float(np.sum(np.log(np.maximum(ev, 1e-6))))
    logjac[k] = float(log_jac(jnp.asarray(best[k][1]), Lgrid[k]))
logpost_L = np.where(skip, -np.inf, -U + logjac - 0.5 * logdet)  # density per unit L (= ln lag)
# Convert to density per unit log10(rest lag) on the grid (constant Jacobian), normalize.
lp = logpost_L - logpost_L.max()
dx = np.gradient(np.log10(lag_rest))
post_grid = np.exp(lp) / np.sum(np.exp(lp) * dx)
# The model's lag prior: Normal(log10 100, 1 dex) in log10 rest lag (truncated well outside the grid).
prior = np.exp(-0.5 * ((np.log10(lag_rest) - 2.0) / 1.0) ** 2)
like_grid = post_grid / prior
like_grid /= np.sum(like_grid * dx)


def summarize(dens):
    cdf = np.cumsum(dens * dx)
    cdf /= cdf[-1]
    q = np.interp([0.16, 0.5, 0.84], cdf, np.log10(lag_rest))
    peaks = [k for k in range(1, len(dens) - 1) if dens[k] >= dens[k - 1] and dens[k] >= dens[k + 1]
             and dens[k] > 0.05 * dens.max()]
    return (10 ** q).tolist(), [float(lag_rest[k]) for k in peaks]


q_post, peaks_post = summarize(post_grid)
q_like, peaks_like = summarize(like_grid)
row = dict(objectId=oid, band=line_band, z=z, map_lag_rest=float(np.exp(L_map) / one_plus_z),
           lag_rest_grid=lag_rest.tolist(), U=U.tolist(), logdet=logdet.tolist(), logjac=logjac.tolist(),
           hessian_nonpos=nonpos.tolist(), hessian_skipped=skip.tolist(), posterior=post_grid.tolist(), prior_divided=like_grid.tolist(),
           q_post=q_post, peaks_post=peaks_post, q_prior_divided=q_like, peaks_prior_divided=peaks_like,
           nfev=nfev_total, t_total=time.time() - T0)
with open(f"{outdir}/{oid}_{line_band}_profile.json", "w") as fh:
    json.dump(row, fh)
print(f"[{oid}] DONE lag posterior {q_post[1]:.0f} [{q_post[0]:.0f},{q_post[2]:.0f}] d, peaks {np.round(peaks_post).tolist()} | "
      f"prior-divided {q_like[1]:.0f} [{q_like[0]:.0f},{q_like[2]:.0f}] | MAP {np.exp(L_map) / one_plus_z:.0f} d | "
      f"non-PD Hessians {int(np.sum(nonpos > 0))}/{len(Lgrid)} | Hessians skipped {int(skip.sum())} | {row['t_total']:.0f}s", flush=True)
