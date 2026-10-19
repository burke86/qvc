"""Exercise launcher arguments without running fits or copying paper assets."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
XONSH = shutil.which("xonsh")
pytestmark = pytest.mark.skipif(XONSH is None, reason="xonsh is required")


def launch(tmp_path, launcher, overrides=None, args=(), fail_at=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "calls.jsonl"
    stub = f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
log = Path(os.environ["LAUNCHER_TEST_LOG"])
previous = log.read_text().splitlines() if log.exists() else []
with log.open("a") as stream:
    stream.write(json.dumps({{"command": Path(sys.argv[0]).name, "args": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")
sys.exit(17 if len(previous) + 1 == int(os.environ["LAUNCHER_TEST_FAIL_AT"]) else 0)
'''
    for name in ("python", "bash"):
        path = tmp_path / name
        path.write_text(stub)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("QVC_")}
    env.update(PATH=f"{tmp_path}{os.pathsep}{env['PATH']}",
               LAUNCHER_TEST_LOG=str(log), LAUNCHER_TEST_FAIL_AT=str(fail_at))
    catalog = tmp_path / "csc.vot"
    catalog.touch()
    env["QVC_XRAY_CATALOG"] = str(catalog)
    env.update(overrides or {})
    result = subprocess.run([XONSH, "--no-rc", str(ROOT / launcher), *args],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def options(call):
    argv = call["args"]
    parsed = {}
    current = None
    for value in argv[2:-1]:  # omit -m/module and the final input catalog
        if value.startswith("--"):
            current = value
            parsed[current] = []
        else:
            parsed[current].append(value)
    return parsed


@pytest.mark.parametrize("overrides,args", [
    ({}, []),
    ({"QVC_HUBBLE_SPEED": "standard", "QVC_HUBBLE_PREFIX": "paper_test",
      "QVC_HUBBLE_LIGHT_CURVE_H5": "custom_light_curve.h5",
      "QVC_HUBBLE_SPECTRA_H5": "custom_spectra.h5",
      "QVC_CUT_LOG_TAU_UV_RF_MIN": "1.5", "QVC_CUT_NUM_DIVERGENCES_MAX": "none",
      "QVC_CUT_APPARENT_MAG_2500_ERR_MAX": "0.5", "QVC_HUBBLE_CUT_TIER": "1",
      "QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE": "posterior-draws",
      "QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE": "tails",
      "QVC_HUBBLE_COMPLETENESS_LF_MODEL": "shen",
      "QVC_HUBBLE_MAGNITUDE_CONVENTION": "attenuated",
      "QVC_HUBBLE_COMPLETENESS_MAGNITUDE": "dereddened",
      "QVC_HUBBLE_BRIGHT_SUBSAMPLE_MARGIN": "0.15", "NUM_CORES": "2"},
     ["--completeness-mag-bin-width", "0.3", "--completeness-z-bin-width", "0.4",
      "--completeness-smooth-sigma-mag", "0", "--completeness-smooth-sigma-z", "0.5"]),
])
def test_science_profile_and_paper_outputs(tmp_path, overrides, args):
    baseline_result, baseline = launch(tmp_path / "baseline", "run_hubble.xonsh", overrides, args)
    result, calls = launch(tmp_path / "paper", "run_hubble_paper.xonsh", overrides, args)
    assert baseline_result.returncode == 0, baseline_result.stderr
    assert result.returncode == 0, result.stderr
    assert [c["command"] for c in calls] == ["python", "python", "bash"]
    reference = options(baseline[0])
    expected_prefix = overrides.get(
        "QVC_HUBBLE_PREFIX",
        "fiducial",
    )
    expected_profile = {
        "QVC_CUT_SED_REDUCED_CHI2_MAX": "1.3",
        "QVC_CUT_SPECTROSCOPY_REDUCED_CHI2_MAX": "1.1",
        "QVC_CUT_JAXSEDFIT_JOINT_REDUCED_CHI2_MAX": "1.2",
        "QVC_CUT_LOO_CHI2_EFF_MAX": "1.05",
        "QVC_CUT_SPECTRAL_RHAT_MAX": "1.05",
        "QVC_CUT_LIGHT_CURVE_RHAT_MAX": "1.05",
        "QVC_CUT_NUM_DIVERGENCES_MAX": "0",
        "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN": "20",
    }
    expected_samples = (
        ("fiducial", "", "0.44"),
        ("restricted", "", "0.7"),
    )
    for call, (result_prefix, prefix_suffix, z_min) in zip(calls[:2], expected_samples):
        actual = options(call)
        assert actual["--agn-sigma-pivot"] == ["0.2"]
        assert actual["--agn-tau-pivot"] == ["200"]
        assert actual["--cosmo_models"] == ["FlatLambdaCDM", "FlatwCDM", "Flatw0waCDM"]
        assert actual["--run"] == [overrides.get("QVC_HUBBLE_RUN_MODE", "full")]
        assert actual["--z_range"] == [z_min, "3.16"]
        assert actual["--result_prefix"] == [result_prefix]
        prefix = expected_prefix if result_prefix == "fiducial" else overrides.get("QVC_HUBBLE_RESTRICTED_PREFIX", "restricted")
        assert actual["--prefix"] == [prefix]
        assert call["env"]["PREFIX"] == prefix
        assert actual["--speed"] == [overrides.get("QVC_HUBBLE_SPEED", "standard")]
        assert "--plot-completeness" in actual
        assert actual["--bright-subsample-margin"] == [
            overrides.get("QVC_HUBBLE_BRIGHT_SUBSAMPLE_MARGIN", "0.2")
        ]
        assert "--resume" in actual
        assert "--skip-debiased-residual-plot" in actual
        assert not ({"--resume-if-exists", "--minimal-plots", "--skip_plots"} & actual.keys())
        exceptions = {"--sdss-target-selection", "--agn-sigma-pivot", "--agn-tau-pivot", "--spectra_fit_csv", "--cosmo_models", "--run", "--speed", "--z_range", "--result_prefix",
                      "--prefix", "--plot-completeness", "--bright-subsample-margin",
                      "--resume", "--resume-if-exists", "--skip-debiased-residual-plot",
                      "--minimal-plots", "--early-de-guard"}
        assert {k: v for k, v in actual.items() if k not in exceptions} == {
            k: v for k, v in reference.items() if k not in exceptions}
        assert call["args"][-1] == overrides.get(
            "QVC_HUBBLE_LIGHT_CURVE_H5",
            "results/data/light_curve_data.h5",
        )
        assert actual["--spectra_fit_csv"] == [overrides.get(
            "QVC_HUBBLE_SPECTRA_H5",
            "results/data/spectra_data.h5",
        )]
        for key, value in baseline[0]["env"].items():
            if key in {"NUM_CORES", "PYTHONPATH", "MPLCONFIGDIR", "QT_QPA_PLATFORM", "SHEN_PUBTOOLS_PATH"}:
                assert call["env"][key] == value
        for key, default in expected_profile.items():
            assert call["env"][key] == overrides.get(key, default)
    assert calls[2]["args"] == [str(ROOT / "scripts/copy_paper_assets.sh"), "--speed",
                                 overrides.get("QVC_HUBBLE_SPEED", "standard"),
                                 "--fiducial-dir", f"plots/hubble/{expected_prefix}",
                                 "--restricted-dir", "plots/hubble/" + overrides.get("QVC_HUBBLE_RESTRICTED_PREFIX", "restricted")]


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_failure_stops_launcher(tmp_path, fail_at):
    result, calls = launch(tmp_path, "run_hubble_paper.xonsh", fail_at=fail_at)
    assert result.returncode != 0
    assert len(calls) == fail_at


def test_exploratory_mode_skips_paper_asset_copy(tmp_path):
    result, calls = launch(
        tmp_path,
        "run_hubble_paper.xonsh",
        {
            "QVC_HUBBLE_RUN_MODE": "single",
            "QVC_HUBBLE_SKIP_PAPER_ASSET_COPY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert [call["command"] for call in calls] == ["python", "python"]
    assert all(options(call)["--run"] == ["single"] for call in calls)
    assert "Skipping paper asset copy" in result.stdout


@pytest.mark.parametrize("key", ["QVC_HUBBLE_CUT_TIER", "QVC_HUBBLE_RUN_MODE",
                                  "QVC_HUBBLE_LIGHT_CURVE_UNCERTAINTY_MODE",
                                  "QVC_HUBBLE_COMPLETENESS_MAGNITUDE_SUPPORT_MODE",
                                  "QVC_HUBBLE_BRIGHT_SUBSAMPLE_MARGIN"])
def test_invalid_setting_fails_before_launch(tmp_path, key):
    result, calls = launch(tmp_path, "run_hubble_paper.xonsh", {key: "invalid"})
    assert result.returncode != 0
    assert calls == []


def test_missing_csc_copies_remaining_paper_assets(tmp_path):
    result, calls = launch(tmp_path, "run_hubble_paper.xonsh", {
        "QVC_XRAY_CATALOG": str(tmp_path / "missing-csc.vot"),
    })
    assert result.returncode == 0, result.stderr
    assert calls[-1]["command"] == "bash"
    assert calls[-1]["args"][:2] == [str(ROOT / "scripts/copy_paper_assets.sh"), "--skip-xray"]
