#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-hubble}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

QVC_CODE_DIR="${QVC_CODE_DIR:-${REPO_ROOT}}"
QVC_WORKDIR="${QVC_WORKDIR:-${REPO_ROOT}}"
QVC_DATA_DIR="${QVC_DATA_DIR:-${QVC_WORKDIR}/data}"
QVC_RESULT_DIR="${QVC_RESULT_DIR:-${QVC_WORKDIR}/results}"
QVC_PLOTS_DIR="${QVC_PLOTS_DIR:-${QVC_WORKDIR}/plots}"
QVC_DUSTMAPS_DIR="${QVC_DUSTMAPS_DIR:-${QVC_WORKDIR}/results/dustmaps}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export NUM_CORES="${NUM_CORES:-4}"
export QVC_CODE_DIR
export QVC_WORKDIR
export QVC_DATA_DIR
export QVC_RESULT_DIR
export QVC_PLOTS_DIR
export QVC_DUSTMAPS_DIR
export PYTHONPATH="${QVC_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DEMO_PREFIX="${DEMO_PREFIX:-demo}"
DEMO_OBJECT_ID="${DEMO_OBJECT_ID:-1452887}"

HUBBLE_LC_H5="${QVC_WORKDIR}/results/data/light_curve_data.h5"
HUBBLE_SPECTRA_H5="${QVC_WORKDIR}/results/data/spectra_data.h5"

log() {
  printf '[qvc-demo] %s\n' "$*"
}

ensure_workdirs() {
  mkdir -p \
    "${QVC_WORKDIR}" \
    "${QVC_DATA_DIR}" \
    "${QVC_RESULT_DIR}" \
    "${QVC_PLOTS_DIR}" \
    "${QVC_DUSTMAPS_DIR}"
}

run_setup() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"
  log "running qvc.setup_data in ${QVC_WORKDIR}"
  python -m qvc.setup_data "$@"
}

require_raw_light_curves() {
  local name
  for name in Catalog.parquet dr16s82_sdssLCRaw.parquet dr16s82_ps1LCRaw.parquet dr16s82_ZuberLCRaw.parquet; do
    if [[ ! -f "${QVC_DATA_DIR}/S82/${name}" ]]; then
      log "missing raw fitting input: ${QVC_DATA_DIR}/S82/${name}"
      log "Zenodo contains Hubble summaries, not raw S82 photometry. See README; use 'hubble' for the published-data workflow."
      exit 1
    fi
  done
}

run_light_curve() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  mkdir -p "${QVC_WORKDIR}/results/data/${DEMO_PREFIX}" "${QVC_WORKDIR}/plots/multiband/${DEMO_PREFIX}"
  export PREFIX="${DEMO_PREFIX}"
  export SUFFIX="${DEMO_PREFIX}"

  log "running light-curve fit for object ${DEMO_OBJECT_ID}"
  python -m qvc.light_curve.fit_light_curves \
    --filter_object_id "${DEMO_OBJECT_ID}" \
    --svi_steps 100 \
    --nwarm 100 \
    --nsamp 100 \
    --nchains 1 \
    --progress \
    --plot \
    --fit_method "svi+nuts" \
    --corner_plot_mode "full"
}

run_spectra() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  mkdir -p "results/data/spectra" "results/spectra/${DEMO_PREFIX}" "plots/spectra/${DEMO_PREFIX}"

  log "running spectra fit for object ${DEMO_OBJECT_ID}"
  python -m qvc.spectra.fit_spectra \
    --mode fit \
    "results/data/spectra/${DEMO_PREFIX}.csv" \
    --cache-dir "data/spectra_cache_all" \
    --output-dir "results/spectra/${DEMO_PREFIX}" \
    --fig-dir "plots/spectra/${DEMO_PREFIX}" \
    --verbose \
    --save-fig \
    --nuts-warmup 100 \
    --nuts-samples 100 \
    --nuts-chains 1 \
    --filter_object_id "${DEMO_OBJECT_ID}" \
    --plot_mcmc_diagnostics \
    --nproc 1
}

run_hubble() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  local path sample model stage
  for path in "${HUBBLE_LC_H5}" "${HUBBLE_SPECTRA_H5}"; do
    [[ -f "${path}" ]] || { log "missing Hubble input: ${path}"; exit 1; }
  done
  for sample in fiducial restricted; do
    for model in FlatLambdaCDM FlatwCDM Flatw0waCDM; do
      for stage in joint sna; do
        path="${QVC_WORKDIR}/results/hubble_posteriors/${sample}/${model}_${stage}.h5"
        [[ -f "${path}" ]] || { log "missing resume checkpoint: ${path}"; exit 1; }
      done
    done
  done

  # Resolve inputs in the mounted workspace, not the installed source tree.
  export QVC_XRAY_CATALOG="${QVC_XRAY_CATALOG:-${QVC_WORKDIR}/results/diagnostics/chandra_precut_sep13/csc21_region_download.vot}"
  export QVC_HUBBLE_SKIP_PAPER_ASSET_COPY=1
  log "resuming the published fiducial and restricted Hubble checkpoints"
  python -m xonsh --no-rc "${QVC_CODE_DIR}/run_hubble_paper.xonsh"
}

case "${COMMAND}" in
  setup)
    run_setup
    ;;
  light-curve)
    require_raw_light_curves
    run_setup
    run_light_curve
    ;;
  spectra)
    require_raw_light_curves
    run_setup
    run_spectra
    ;;
  hubble)
    run_setup --paper
    run_hubble
    ;;
  all)
    require_raw_light_curves
    run_setup
    run_light_curve
    run_spectra
    run_hubble
    ;;
  *)
    log "unknown command: ${COMMAND}"
    log "expected one of: all, setup, light-curve, spectra, hubble"
    exit 2
    ;;
esac
