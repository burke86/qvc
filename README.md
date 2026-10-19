# QVC: Demo and Reproducibility Guide

This repository provides end-to-end tooling for:

1. Multi-band AGN light-curve fitting
2. Spectral fitting with JAXQSOFit
3. Hubble-diagram fitting and figure generation

The quick-start workflow downloads the published Zenodo summaries and recreates the Hubble runs with saved posteriors. Optional light-curve and spectrum refitting requires additional raw S82 inputs (see below).

The resulting figures are found under `docker-workdir/plots/` when run with Docker, or under `plots/` for a local run.

Note: For speed, the light curve fitting and spectra fitting will run with a minimal number of warmup and sampling steps, and the produced plots may have minor deviations from the published figures. The published figures were run with longer warmup and sampling steps in Yale's HPC Clusters.

## Quick Start

```bash
git clone https://github.com/dutra/qvc.git
cd qvc
```

### Docker
For convenience, we include a self-contained Docker (https://www.docker.com) container.

```bash
docker build -t qvc-demo .
mkdir -p "$(pwd)/docker-workdir"
```

Run the published Hubble replication workflow:

```bash
docker run --rm \
  -v "$(pwd)/docker-workdir:/work/qvc-demo" \
  qvc-demo hubble
```

Available commands:

```text
all
setup
light-curve
spectra
hubble
```

The `hubble` command runs `setup_data --paper` automatically inside `docker-workdir`, then strictly resumes the published checkpoints. It generates the Wang completeness mock locally and skips copying unrelated manuscript assets. Results and plots remain in the mounted directory. The image defaults to `hubble` when no command is supplied.

`all`, `light-curve`, and `spectra` additionally require the raw S82 files described below. Place them in `docker-workdir/data/S82/` before using those commands. Full `setup` also downloads the fitting dependencies and dustmaps into the mounted directory.

## Local Install

You may also install the package using pip.
First, create and activate a Python environment. We recommend Conda (https://www.anaconda.com/download).

```bash
conda create -n jaxcpu -c conda-forge python=3.12.11 pip
conda activate jaxcpu
pip install -e .
pip install --force-reinstall --no-deps "jaxsedfit @ git+https://github.com/burke86/jaxsedfit.git"
```

JAXSEDFit follows the upstream default branch without a commit pin. The final
command refreshes it even when upstream has not changed its package version;
repeat it when updating an existing environment. Catalog manifests record the
installed revision for provenance.

Run the published Hubble replication workflow:

```bash
bash scripts/run_demo.sh hubble
```

With the raw S82 inputs installed, you can also run the fitting stages (or `all`):

```bash
bash scripts/run_demo.sh setup
bash scripts/run_demo.sh light-curve
bash scripts/run_demo.sh spectra
bash scripts/run_demo.sh hubble
```

Alternatively, you may call each command directly. Note again that these commands will run a minimal number of warmup and sampling steps for speed; the produced plots may have minor deviations from the published figures. The published figures were run with longer warmup and sampling steps in Yale's HPC Clusters.

All plots and results will be generated under `plots/` and `results/`, respectively.

The paper launcher (`run_hubble_paper.xonsh`) fixes both the fiducial and
restricted samples at `--agn-sigma-pivot 0.2 --agn-tau-pivot 200` (magnitudes and
rest-frame days). Other launchers retain sample-median pivots. Supply both options
together with finite, positive values; they cannot be combined with
`--disable-pivot-rounding`. Checkpoint resumes require matching pivot values and
rules. The paper launcher resumes the published `fiducial` and `restricted`
checkpoints; missing checkpoints fail rather than starting a new fit.

### Setup

Downloads the cached spectrum, DR16Q catalog, Hubble inputs, legacy completeness mock, DSPS data, Pantheon+ files, CSC catalog, and dustmaps. Raw S82 photometry is not included in this setup.

Expected output: downloaded inputs under `data/`, `results/data/`, `results/cosmo/`, `results/hubble_posteriors/`, and dustmaps under `results/dustmaps/`.

```bash
python -m qvc.setup_data
```

For the published paper's Hubble resume only:

```bash
python -m qvc.setup_data --paper
QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1 python -m xonsh --no-rc run_hubble_paper.xonsh
```

This downloads the checksum-verified inputs from
[Zenodo record 23021880](https://zenodo.org/records/23021880):
`results/data/light_curve_data.h5`, `results/data/spectra_data.h5`, and the
12 checkpoints under `results/hubble_posteriors/{fiducial,restricted}/`.
The paper-only setup also downloads the external Pantheon+ data and covariance
and queries the official CSC 2.1 TAP service for the paper's X-ray region
(-1.251 to +1.251 degrees declination, all right ascensions). It validates the
response and produces
`results/diagnostics/chandra_precut_sep13/csc21_region_download.vot`, the CSC
catalog read by the paper launcher. Existing valid catalogs are reused.
CSC network failures (including HTTP errors and timeouts) produce a warning;
setup continues and retries CSC on its next invocation. When the catalog is
missing, Hubble fitting continues with the same sample and fit settings,
but omits CSC matching and X-ray-dependent diagnostics. Setup reports that CSC
is unavailable. Invalid catalog responses and failures downloading required
inputs still stop setup. Source matching, when available, uses the catalog's
positional-error ellipses during the Hubble analysis.
The launcher regenerates the Wang completeness mock; raw fitting catalogs,
per-object plot archives, and `Table_7.csv` are not required for this resume.
The command above skips the full manuscript asset-copy step, which needs assets outside this download. Omit `QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1` only when those assets are available.

### Raw inputs for optional refitting

The Zenodo HDF5 files contain fit summaries, not individual photometric epochs.
Both refitting stages require these additional files under `data/S82/`:

- `Catalog.parquet`
- `dr16s82_sdssLCRaw.parquet`
- `dr16s82_ps1LCRaw.parquet`
- `dr16s82_ZuberLCRaw.parquet`

`setup_data` currently has no download source for these files. Supply them from
the original S82 dataset before running the refitting commands; the demo runner
checks for them before downloading fitting dependencies. The published Hubble
workflow above does not require them. Run `python -m qvc.setup_data` before
calling the individual fitting modules directly.

### Light Curve Fitting

Fits the demo AGN light curve for object `1452887` with reduced SVI/NUTS sampling and writes the local light-curve result files and plots.

Expected output: results under `results/data/demo/` and figures under `plots/multiband/demo/`.

```bash
PREFIX=demo SUFFIX=demo python -m qvc.light_curve.fit_light_curves \
  --filter_object_id 1452887 \
  --svi_steps 100 \
  --nwarm 100 \
  --nsamp 100 \
  --nchains 1 \
  --progress \
  --plot \
  --fit_method "svi+nuts" \
  --corner_plot_mode "full"
```

The experimental shared-latent response model is available through a
dedicated entry point:

```bash
PREFIX=latent_demo SUFFIX=latent_demo \
python -m qvc.light_curve.fit_light_curves_shared_latent_blr \
  --filter_object_id 1452887 \
  --nwarm 500 \
  --nsamp 250 \
  --nchains 2 \
  --plot
```

It uses one unit-RMS DHO driver for every band. Each observed continuum is a
narrow Erlang-convolved disk response whose centroid follows a shared
`lambda^(4/3)` law, while each band's generic delayed component has its own
lag and unit-RMS Erlang response. The same model can be selected from the main
entry point with `--model_variant shared_latent_blr`. `--disk_order` controls
the disk-response width (default 3); `--erlang_order` controls the delayed
response width (default 3). No emission-line, Fe, or diffuse-continuum
wavelength template is imposed. The shared driver is wavelength-independent,
so this variant has no `eta_tau`; `--eta_prior_profile modified` changes only
its `eta_sigma` prior.

For this variant, `log_tau_uv_rf` is the exact rest-frame integral-correlation
timescale of a synthetic continuum-only disk response evaluated at 2500 A.
It excludes every delayed/BLR component. `log_tau_driver_rf` is the intrinsic
shared-driver integral timescale, `log10((tau_fast + tau_slow)/(1 + z))`, with
the individual pole diagnostics retained as `log_tau_driver_fast_rf` and
`log_tau_driver_slow_rf`.

For output compatibility with the other light-curve models,
`log_tau_band_<band>_RF` is the rest-frame effective timescale of that band's
complete disk-plus-delayed light curve. It is the exact integral of the
normalized autocorrelation function, not a fitted response centroid. The same
value is also written as `log_tau_effective_<band>_RF`.

Seeing-dependent photometric contamination can be enabled with
`--enable_seeing_dependence` for any GP choice, including
`shared_latent_blr` and the CARMA(2,1) parameterization. For each band/survey
combination with at least three usable epoch-level PSF-FWHM measurements and
nonzero seeing range, this fits both a centered mean-flux slope and a
seeing-dependent extra-scatter slope. It is disabled by default.

### Spectra Fitting

Fits the matching cached SDSS spectrum with `qvc.spectra.fit_spectra`, saving the output CSV, fit figures, and MCMC diagnostic plots.

Expected output: the fit CSV at `results/data/spectra/demo.csv`, result artifacts under `results/spectra/demo/`, and figures under `plots/spectra/demo/`.

```bash
python -m qvc.spectra.fit_spectra \
  --mode fit \
  "results/data/spectra/demo.csv" \
  --cache-dir "data/spectra_cache_all" \
  --output-dir "results/spectra/demo" \
  --fig-dir "plots/spectra/demo" \
  --verbose \
  --save-fig \
  --nuts-warmup 100 \
  --nuts-samples 100 \
  --nuts-chains 1 \
  --filter_object_id 1452887 \
  --plot_mcmc_diagnostics \
  --nproc 1
```

An experimental SED-first companion jointly fits a saved broadband-SED table
and the SDSS spectrum with JAXSEDFit. It keeps the native `fit_spectra` sample
matching and spectrum cache, always replaces saved SDSS points with the QVC
light-curve PSF `ugriz` means (including `z` even when it is excluded from the
variability fit), and uses JAXSEDFit's `jaxqsofit` joint spectral backend:
The summary CSV includes `m_2500_dereddened` from the intrinsic AGN disk plus
separate `a_2500_galaxy` and `a_2500_internal` attenuation estimates.

```bash
python -m qvc.spectra.fit_spectra_jaxsedfit_joint \
  --mode fit \
  "results/data/spectra/demo_joint.csv" \
  --cache-dir "data/spectra_cache_all" \
  --sed-photometry-path "/path/to/saved_sed_photometry.parquet" \
  --output-dir "results/jaxsedfit_joint/demo" \
  --fig-dir "plots/jaxsedfit_joint/demo" \
  --dsps-ssp-fn "data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5" \
  --filter_object_id 1452887 \
  --progress \
  --nproc 1
```

### Hubble Diagram Fitting

Resumes the saved fiducial and restricted Hubble posterior checkpoints and regenerates the comparison outputs and plots.

`--resume` remains strict and fails when a requested checkpoint is absent. Use
`--resume-if-exists` with the same optional per-model H5 paths to resume files
that exist and start fresh fits for files that do not. For `run_hubble.xonsh`,
set `QVC_HUBBLE_RESUME_IF_EXISTS=true` to check the default checkpoint paths, or
set it to an explicit checkpoint path; do not combine it with
`QVC_HUBBLE_RESUME`.

Expected output: results under `results/` and figures under `plots/hubble/fiducial/` and `plots/hubble/restricted/`.

Both Docker and local execution use `run_hubble_paper.xonsh`, with the same
Zenodo input names, cut settings, fixed pivots, Wang completeness model, and
`fiducial`/`restricted` checkpoint directories. The default magnitude convention
is `dereddened`; the completeness magnitude is `attenuated`. Preserve these
settings when reproducing the published checkpoints.

```bash
bash scripts/run_demo.sh hubble
```

For an already downloaded dataset, the equivalent direct command is:

```bash
QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1 python -m xonsh --no-rc run_hubble_paper.xonsh
```

The launcher reads `results/data/light_curve_data.h5` and
`results/data/spectra_data.h5`. The latter is the versioned minimal spectral
HDF5 schema accepted by the Hubble loader; a legacy spectral CSV is not needed.
It uses redshift ranges 0.44–3.16 (fiducial) and 0.7–3.16 (restricted).


## System Notes

- Tested with Python `3.12.11`.
- Tested on an AMD x86_64 desktop, MacBook Air M1, and Yale HPC.
- All required Python packages and versions are listed under `pyproject.toml` and will be installed automatically with the `pip install -e .` command.
- Typical local install time is on the order of 10 minutes and less than 1 hour on a normal desktop computer, depending on hardware and internet speed.
- Local demo runtime is on the order of a few hours on a normal computer.
- Full-scale production runs were executed on Yale HPC and required over `100,000` CPU-hours.

## Optional Pubtools Build

The published Hubble launcher generates its Wang completeness mock using the Python implementation and does not need `pubtools`. Full setup also retains the legacy Shen mock at `results/data/mock_completeness_catalog_fresh.h5`.

Only build `pubtools` if you want to regenerate that mock catalog:

```bash
git clone https://bitbucket.org/ShenXuejian/quasarlf.git
export SHEN_PUBTOOLS_PATH=<directory of quasarlf>/quasarlf/pubtools
```

Follow the `pubtools/clib` instructions in that repository to compile the shared libraries.

## Useful Scripts

`scripts/run_demo.sh hubble` runs paper setup and Hubble reproduction. Its `all` command additionally runs light-curve and spectrum fitting when raw inputs are present.

`scripts/docker_entrypoint.sh` is the Docker entrypoint and accepts `all`, `setup`, `light-curve`, `spectra`, and `hubble`.

`scripts/create_master_input_list.py` exports a master CSV of S82 light curves joined to DR17 spectroscopy metadata.

`scripts/copy_paper_assets.sh` copies generated manuscript assets into paper-output locations.

## Appendix Data

To fetch appendix datasets:

```bash
python -m qvc.setup_data --appendix
```

The Stone, MacLeod, and same-length comparison plots were generated using Yale's HPC by running the `hpc_scripts/sfitlc.py` script.

## HPC Scripts
The `hpc_scripts` folder contains Slurm/Yale-HPC-oriented helpers. Treat these as templates: partitions, paths, Conda environments, and account-specific settings may need local edits.

- `hpc_scripts/smanage.py` inspects, cancels, holds, or resumes Slurm jobs by job-name glob.
  ```bash
  python hpc_scripts/smanage.py status "multiband_*"
  ```
- `hpc_scripts/sfitlc.py` generates and submits multiband light-curve Slurm arrays for `chisq`, `stone`, `macleod`, and `samelength` samples, including merge jobs.
  ```bash
  python hpc_scripts/sfitlc.py --fit stone --N 1
  ```
- `hpc_scripts/sfitspectra.xsh` generates and submits spectra-fitting Slurm arrays from a chi-square selected object list, with optional exclusion filtering.
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh \
    --chisq-csv data/jul14_master_input_file_chisqgt20_nested_N8000.csv \
    --description nested_N8000
  ```
  Add `--fit-bal` to enable JAXSEDFit's built-in N V, Si IV, and C IV BAL
  absorption components in the default `fit_spectra_jaxsedfit_joint.py`
  backend. It can be combined with `--resume` for posterior bundles produced
  by a BAL-enabled run; the saved JAXSEDFit configuration is validated before
  reuse.
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh \
    --chisq-csv data/jul14_master_input_file_chisqgt20_nested_N8000.csv \
    --description nested_N8000_BAL \
    --fit-bal
  ```
  To regenerate catalogs and plots without any fresh inference, add
  `--resume OLD_RUN_NAME --resume-only`. Saved bundles are read from
  `results/data/jaxqsofit/OLD_RUN_NAME/all/`; outputs go into a new run.
  Individual failures (including missing bundles) produce `fit_ok=False`
  catalog rows and processing continues. A chunk with any successful objects
  exits successfully; an entirely failed chunk writes its failure catalog
  before exiting nonzero. Inspect `fit_ok` and `error_message` even for
  completed Slurm tasks.

  To retry only tasks whose latest attempt ended unsuccessfully, pass the full
  original job name:
  ```bash
  xonsh hpc_scripts/sfitspectra.xsh \
    --retry aug03_1853_f3ea5a6_svi4000_N2000_PRS103to107
  ```
  Retry mode requires the original generated
  `hpc_scripts/submit/jaxqsofit/submit_<job>.sbatch` script and matching
  `<job>_object_ids.txt` manifest. It searches Slurm accounting records from
  the last seven days, preserves the original task and chunk numbers, and
  applies the scheduler resources currently configured in `sfitspectra.xsh`.
  Every object in an unsuccessful chunk is rerun; completed chunks are left
  untouched.
- `hpc_scripts/shubble_grid.xsh` generates and submits Hubble grid sweeps over `N` and `zmax`.
  ```bash
  xonsh hpc_scripts/shubble_grid.xsh
  ```
