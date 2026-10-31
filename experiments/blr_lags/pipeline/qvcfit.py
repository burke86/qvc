"""Shared single-object model construction for the BLR lag pipeline.

Mirrors fit_light_curves.main() input preparation (concat_light_curves with the
default outlier rejection -> populate_sdss_fields -> make_lc -> jitter/survey
masks -> flux-linearized shared_latent_blr model, refinement_iters=1, fast
solver). Parquet reads are filtered to the object so a fit never loads a full
catalog.
"""

from pathlib import Path

import numpy as np
import pandas as pd

D = str(Path(__file__).resolve().parents[3] / "data" / "S82")
_STATE = {}
_read_parquet = pd.read_parquet


def _filtered_read_parquet(path, *a, **k):
    p = str(path)
    if p.endswith("dr16s82_sdssLCRaw.parquet"):
        k["filters"] = [("objectId", "==", _STATE["oid"])]
    elif p.endswith("dr16s82_ps1LCRaw.parquet") or p.endswith("dr16s82_ZuberLCRaw.parquet"):
        k["filters"] = [("ps1objID", "==", _STATE["ps1"])]
    return _read_parquet(path, *a, **k)


pd.read_parquet = _filtered_read_parquet
_CATALOG = _read_parquet(f"{D}/Catalog.parquet", columns=["objectId", "ps1objID"])

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from qvc.light_curve import fit_light_curves as F  # noqa: E402
from qvc.light_curve import multiband_generate_lc as G  # noqa: E402

def build(oid):
    """Return (obj, model, info dict) for one object, exactly as the pipeline would build it."""
    oid = str(oid)
    _STATE["oid"] = oid
    _STATE["ps1"] = int(_CATALOG[_CATALOG.objectId.astype(str) == oid].iloc[0].ps1objID)
    obj = G.populate_sdss_fields(G.concat_light_curves(filter_object_ids=[oid]), progress_bar=False)[0]
    obj["psf_constant_flux_n_bands_corrected"] = 0
    obj["psf_constant_flux_corrected"] = False
    obj |= F.make_lc(obj, bands=["u", "g", "r", "i", "z"], verbose=False)
    bands = list(obj["bands"])
    z = float(obj["z"])
    B = len(bands)
    lam_rf = jnp.array([F.lambda_pivot[x] for x in bands], dtype=float) / (1.0 + z)
    bidx = np.asarray(obj["band_idx"])
    survey_idx = np.asarray(obj["survey_idx"], dtype=np.int32)
    _, obj["log_jitter_active_mask"] = F._compute_log_jitter_mean_grid(np.asarray(obj["yerr"]), bidx, survey_idx, B)
    obj["survey_offset_active_mask"] = F._compute_survey_offset_active_mask(bidx, survey_idx, B)
    y_rel, yerr_rel = F._flux_linearized_initial_arrays(obj)
    log_jitter_mean_fit, _ = F._compute_log_jitter_mean_grid(np.asarray(yerr_rel), bidx, survey_idx, B)
    model = F.build_single_object_model_mag_flux_linearized(
        F._flux_linearized_fit_object(obj, y_rel, yerr_rel), lam_rf, log_jitter_mean_fit,
        lam_lya_rf=F.compute_lam_lya_suppression_rf(bands, z),
        use_fast_solver=True,
    )
    return obj, model, dict(bands=bands, z=z, B=B, bidx=bidx, survey_idx=survey_idx)
