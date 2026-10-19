"""Exercise both entrypoints in an empty workspace without downloads or inference."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_demo(tmp_path, command="hubble", *, docker=False, missing=False, setup_fail=False, raw=False):
    work = tmp_path / "mounted work"
    work.mkdir()
    if raw:
        folder = work / "data/S82"
        folder.mkdir(parents=True)
        for name in ("Catalog.parquet", "dr16s82_sdssLCRaw.parquet", "dr16s82_ps1LCRaw.parquet", "dr16s82_ZuberLCRaw.parquet"):
            (folder / name).touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps({{"args": args, "cwd": os.getcwd(), "xray": os.environ.get("QVC_XRAY_CATALOG")}}) + "\\n")
if args[:2] == ["-m", "qvc.setup_data"]:
    if os.environ.get("SETUP_FAIL") == "1":
        sys.exit(19)
    from qvc.setup_data import PAPER_STEPS
    for step in PAPER_STEPS:
        if step["type"] == "download" and not step["filename"].endswith(".zip"):
            path = Path(step["folder"]) / step["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    for sample in ("fiducial", "restricted"):
        for model in ("FlatLambdaCDM", "FlatwCDM", "Flatw0waCDM"):
            for stage in ("joint", "sna"):
                if os.environ.get("MISSING_CHECKPOINT") == "1" and sample == "restricted" and model == "FlatwCDM" and stage == "sna":
                    continue
                path = Path("results/hubble_posteriors") / sample / f"{{model}}_{{stage}}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
elif args[:2] == ["-m", "xonsh"]:
    os.execv(sys.executable, [sys.executable, *args])
elif args[:2] not in (["-m", "qvc.hubble.hubble_fit"], ["-m", "qvc.spectra.fit_spectra"]):
    sys.exit(20)
''')
    stub.chmod(0o755)
    copy_stub = bin_dir / "bash"
    copy_stub.write_text(f'''#!{sys.executable}
import json, os, sys
if not sys.argv[1].endswith("copy_paper_assets.sh"):
    os.execv("/bin/bash", ["/bin/bash", *sys.argv[1:]])
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps({{"args": sys.argv[1:], "cwd": os.getcwd()}}) + "\\n")
''')
    copy_stub.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("QVC_")}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", QVC_CODE_DIR=str(ROOT),
               QVC_WORKDIR=str(work), CALL_LOG=str(log),
               XONSH_HISTORY_BACKEND="dummy", MISSING_CHECKPOINT=str(int(missing)),
               SETUP_FAIL=str(int(setup_fail)))
    script = "docker_entrypoint.sh" if docker else "run_demo.sh"
    args = ["/bin/bash", str(ROOT / "scripts" / script)]
    if command is not None:
        args.append(command)
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls, work


@pytest.mark.parametrize("docker,command", [(False, "hubble"), (False, None), (True, "hubble"), (True, None)])
def test_published_resume_in_separate_workspace(tmp_path, docker, command):
    result, calls, work = run_demo(tmp_path, command, docker=docker)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0]["args"] == ["-m", "qvc.setup_data", "--paper"]
    assert len(calls) == 5  # setup, xonsh, two strict resumes, manuscript copy
    assert calls[-1]["args"] == [str(ROOT / "scripts/copy_paper_assets.sh"),
                                 "--skip-xray", "--speed", "standard",
                                 "--fiducial-dir", "plots/hubble/fiducial",
                                 "--restricted-dir", "plots/hubble/restricted"]
    assert calls[-1]["cwd"] == str(work)
    assert "Skipping paper asset copy" not in result.stdout
    for call, sample in zip(calls[2:], ("fiducial", "restricted")):
        args = call["args"]
        assert args[:3] == ["-m", "qvc.hubble.hubble_fit", "--resume"]
        assert args[args.index("--result_prefix") + 1] == sample
        assert args[args.index("--prefix") + 1] == sample
        assert args[args.index("--spectra_fit_csv") + 1] == "results/data/spectra_data.h5"
        assert args[-1] == "results/data/light_curve_data.h5"
        assert call["cwd"] == str(work)
        assert call["xray"] == str(work / "results/diagnostics/chandra_precut_sep13/csc21_region_download.vot")


@pytest.mark.parametrize("kwargs", [{"missing": True}, {"setup_fail": True}])
def test_incomplete_setup_never_launches_inference(tmp_path, kwargs):
    result, calls, _ = run_demo(tmp_path, **kwargs)
    assert result.returncode != 0
    assert len(calls) == 1


@pytest.mark.parametrize("command", ["all", "light-curve", "spectra"])
def test_raw_fitting_inputs_are_required_before_downloads(tmp_path, command):
    result, calls, _ = run_demo(tmp_path, command)
    assert result.returncode != 0
    assert "missing raw fitting input" in result.stdout
    assert calls == []


def test_spectrum_refit_uses_object_id_not_published_summary(tmp_path):
    result, calls, _ = run_demo(tmp_path, "spectra", raw=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0]["args"] == ["-m", "qvc.setup_data"]
    args = calls[1]["args"]
    assert args[:2] == ["-m", "qvc.spectra.fit_spectra"]
    assert "--fpath-in" not in args
    assert args[args.index("--filter_object_id") + 1] == "1452887"
    assert "results/data/spectra/demo.csv" in args
