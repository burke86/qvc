import numpy as np
import pandas as pd
import pytest

from qvc.hubble import hubble_utils
from qvc.hubble.hubble_model import AgnPivotContext


@pytest.mark.parametrize("prefix", ["fiducial", "restricted"])
def test_wavelength_tex_exports_pooled_fit_and_bootstrap(tmp_path, monkeypatch, prefix):
    frame = pd.DataFrame({"z": [1.0, 4.0]})
    calls = []

    def fit(df, **kwargs):
        assert df is frame
        calls.append(kwargs)
        return {
            "fit_sigma": {
                "d1": -0.6378, "d2": -0.6301,
                "slope_percentiles": np.array([[-0.6431, -0.6326], [-0.6372, -0.6233]]),
            },
            "fit_tau": {
                "d1": 0.0228, "d2": 0.0146,
                "slope_percentiles": np.array([[0.0219, 0.0236], [0.0142, 0.0150]]),
            },
        }

    monkeypatch.setattr(hubble_utils, "fit_sigma_tau_lambda_broken_pl", fit)
    pivot = AgnPivotContext(
        observable_names=("log_sigma_uv", "log_tau_uv_rf"),
        values=(-1.0, np.log10(200)), reference_object_ids=("obj1",),
        z_range=(0.44, 3.16),
    )
    hubble_utils.write_results_tex_variables(
        frame, frame, pd.DataFrame({"IS_CALIBRATOR": [], "zHD": []}),
        (0.44, 3.16), {}, {}, None, str(tmp_path),
        result_prefix=prefix, agn_pivot_context=pivot,
    )
    assert calls == [dict(fit_mode="common_bootstrap", lam_s=2500.0, disk_order=3,
                          bootstrap_replicates=1000, bootstrap_seed=2500)]
    tex = (tmp_path / f"param_results_{prefix}.tex").read_text()
    expected = {
        "SigmaBlue": r"-0.638 \pm 0.005",
        "SigmaRed": r"-0.630 \pm 0.007",
        "TauBlue": r"0.023 \pm 0.001",
        "TauRed": r"0.015 \pm 0.001",
    }
    for suffix, value in expected.items():
        macro = rf"\newcommand{{\result{prefix.capitalize()}Eta{suffix}Lambda}}{{\ensuremath{{{value}}}}}"
        assert tex.count(macro) == 1
    assert "pooled broken power-law fitted slopes" in tex
    assert "16th-84th percentile whole-quasar bootstrap intervals" in tex
    assert "population ranges" not in tex


@pytest.mark.parametrize("value,lower,upper,nd,expected", [
    (0.015, 0.001, 0.001, 3, r"0.015 \pm 0.001"),
    (-0.638, 0.0053, 0.0052, 3, r"-0.638 \pm 0.005"),
    (-0.638, 0.0053, 0.0062, 3, r"-0.638^{+0.006}_{-0.005}"),
    (123000, 5100, 5200, 2, r"(1.23 \pm 0.05) \times 10^{5}"),
    (1.0, None, 0.001, 2, r"1.00^{+0.00}_{-0.00}"),
])
def test_error_notation_uses_displayed_precision(value, lower, upper, nd, expected):
    assert hubble_utils.format_result_errors(
        value, err_lower=lower, err_upper=upper, nd=nd,
        symmetric_if_equal=True,
    ) == expected


def test_other_parameter_errors_keep_asymmetric_notation():
    assert hubble_utils.format_result_errors(
        0.015, err_lower=0.001, err_upper=0.001, nd=3,
    ) == r"0.015^{+0.001}_{-0.001}"
