"""Exactness of the fused single-scan solver against the tinygp path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tinygp import GaussianProcess

from qvc.light_curve.fast_quasisep import fused_log_probability
from qvc.light_curve.multiband_model_shared_latent_blr import SharedLatentDiskBLRQS

jax.config.update("jax_enable_x64", True)

B = 4


def _data(n_epochs, rng, exact_ties):
    t_ep = np.sort(rng.uniform(0.0, 2000.0, n_epochs))
    t = np.repeat(t_ep, B)
    if not exact_ties:
        t = t + np.tile(np.arange(B) * 1e-3, n_epochs)
    band = np.tile(np.arange(B), n_epochs)
    keep = rng.uniform(size=t.size) > 0.35
    t, band = t[keep], band[keep]
    order = np.argsort(t + 1e-9 * band)
    t, band = t[order], band[order]
    y = rng.normal(0.0, 0.1, t.size)
    yerr = np.full(t.size, 0.02)
    return (jnp.asarray(t), jnp.asarray(band, dtype=jnp.int32)), jnp.asarray(y), jnp.asarray(yerr)


def _kernel_params(rng):
    return dict(
        tau_fast=jnp.asarray(rng.uniform(5.0, 30.0, 1)),
        tau_slow=jnp.asarray(rng.uniform(80.0, 300.0, 1)),
        lag_disk=jnp.asarray(rng.uniform(1.0, 10.0, B)),
        lag_blr=jnp.asarray(rng.uniform(20.0, 200.0, B)),
        amp_cont=jnp.asarray(rng.uniform(0.05, 0.2, B)),
        amp_blr=jnp.asarray(rng.uniform(0.01, 0.08, B)),
    )


@pytest.mark.parametrize("exact_ties", [True, False])
@pytest.mark.parametrize("disk_order,blr_order", [(1, 1), (3, 3)])
def test_fused_log_probability_matches_tinygp(exact_ties, disk_order, blr_order):
    rng = np.random.default_rng(7 * blr_order + exact_ties)
    X, y, yerr = _data(120, rng, exact_ties)
    diag = yerr**2
    kp = _kernel_params(rng)

    def kernel(kp):
        return SharedLatentDiskBLRQS(**kp, disk_order=disk_order, blr_order=blr_order)

    def lp_ref(kp):
        gp = GaussianProcess(kernel(kp), X, diag=diag, assume_sorted=True)
        return gp.log_probability(y)

    def lp_fast(kp):
        return fused_log_probability(kernel(kp), X, diag, y, sort_time=X[0])

    v0, v1 = lp_ref(kp), lp_fast(kp)
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-10)

    g0 = jax.grad(lp_ref)(kp)
    g1 = jax.grad(lp_fast)(kp)
    for k in g0:
        np.testing.assert_allclose(
            np.asarray(g1[k]), np.asarray(g0[k]), rtol=5e-7, atol=1e-10
        )
