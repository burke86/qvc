# Photometric BLR lags from Stripe 82 light curves: getting started

This project asks whether broadband light curves alone (no spectroscopic monitoring) can measure broad-line region (BLR) lags and recover the radius–luminosity (R–L) relation. We fit multiband SDSS + PS1 + ZTF light curves of Stripe 82 quasars with QVC's `shared_latent_blr` Gaussian-process model. In that model each band's flux is continuum plus a delayed, smoothed copy of that continuum (the BLR echo). The lag is a free parameter per band.

This guide covers the data, the model and why it carries lag information, what we have found so far, and the next tasks.

---

## 1. Where things stand (October 2026)

We ran MAP + Laplace fits on the 60 objects with the highest variability S/N per line (Hα, Hβ, Mg II, C IV; 238 object–line pairs). That run also fitted seeing-dependent nuisance terms, which we have since dropped from the pipeline; the earlier 30-object Hβ and Mg II runs without them also showed no clear trend. Results are in `run4_seeing/plots/` (local only, see Section 2).

| Line | N | Median offset from reference R–L | Scatter (NMAD) | Lag–L correlation |
|---|---|---|---|---|
| Hα | 58 | +0.31 dex | 0.47 dex | none (ρ = −0.07) |
| Hβ | 60 | −0.10 dex | 0.46 dex | none (ρ = −0.15) |
| Mg II | 60 | −0.30 dex | 0.29 dex | none (ρ = +0.10) |
| C IV | 60 | no reference | — | none (ρ = −0.01) |

**Key lesson: selecting by variability alone fails because most lines are too weak in the band.** In that sample the broad line was typically only 6–7% of the band flux (16% for Hα). When we restrict to objects whose line is more than 10% of the band flux, things improve:

- Hβ above 10% (10 clean objects): Spearman ρ = +0.66 (p = 0.04), slope 0.52, compared with 0.53 for Bentz et al. (2013). Scatter is 0.15 dex.
- Hα above 20% (18 clean objects): offset +0.11 dex from the reference, scatter 0.22 dex.

See `run4_seeing/plots/lag_vs_L_frac_gt10.pdf` and `run4_seeing/plots/strong_line_selection.pdf`. These samples are small, so the next step is to **re-select on line strength** (Section 6).

---

## 2. Setup

```bash
git clone https://github.com/burke86/qvc.git && cd qvc
git checkout prm   # shared_latent_blr only, fast likelihood, DR16Q column cache
pip install -e .
pip install "equinox==0.12.2"                      # eztaox breaks with equinox 0.13.x (BoundMethod import error)
pip install jaxns speclite astroquery              # nested sampling, filter curves, SDSS spectra
```

- If you see `ImportError: cannot import name 'BoundMethod'`, equinox is too new.
- We run on CPU with `jax_enable_x64`.
- About tinygp: released tinygp 0.3.0 assumes reversible kernels in its default quasisep matrix methods. That is wrong for our causal response chains. QVC's kernel overrides those methods, and we checked it against a dense covariance to about 1e-10, so stock tinygp is fine for this model. The upstream fix is on the branch `fix/nonreversible-quasisep-upstream` in Colin's `tinygp_dev`.
- Plots use the group style: `plt.style.use(jaxsedfit.mplstyle.style_path())`. Save figures as PDF with no suptitle.

**Memory and time:** across 221 fits, each process peaked at 2.3 GB (median; 2.9 GB max), and one MAP + Laplace fit took about 2.5 minutes. Run at most 3 at a time on a laptop. An earlier run with 6 parallel processes, each loading the full catalog, crashed a laptop. The scripts here read the parquet files filtered to one object, so they never load a full catalog.

**Data and results are not in git.** The repo ignores data files (`*.parquet`, `*.csv`, `*.fits`, `*.pdf`). Get these from Colin and place them at the paths below:

- `data/` (light curves, catalogs, `lc_chisq.csv`, DR16Q FITS);
- `experiments/blr_lags/run4_seeing/` (results of the first run: `selection.csv`, `fits/`, `profiles/`, `plots/`; needed only to look at the earlier figures).

The scripts in `pipeline/` are committed. Rerun `count_strong.py` to rebuild `chi20_line_fractions.csv`.

---

## 3. The data

All paths are relative to the qvc repo root.

| File | Contents |
|---|---|
| `data/S82/Catalog.parquet` | One row per quasar (~38k): `objectId`, `ps1objID`, RA/Dec, `PLATE/MJD/FIBERID`, `Z_SYS`, epoch counts per survey and band, `qg_LogL3000`, `qg_fragal` (host-galaxy fraction from the SED decomposition) |
| `data/S82/dr16s82_sdssLCRaw.parquet` | SDSS Stripe 82 PSF magnitudes, ugriz, 1998–2008 (`objectId`, `mjd`, `psMag`, `psMagErr`, `filterID`) |
| `data/S82/dr16s82_ps1LCRaw.parquet` | Pan-STARRS1 detections, grizy (`ps1objID`, `obsTime`, `psfFlux`, `psfQf`, `psfChiSq`, `infoFlag*`, ...) |
| `data/S82/dr16s82_ZuberLCRaw.parquet` | ZTF PSF light curves, gri (`mag`, `magerr`, `fieldid`, `rcidin`, `flag`) |
| `data/lc_chisq.csv` | Variability reduced χ² per object (`chi_sq_g`, `chi_sq_all`). **The selection uses `chi_sq_g > 20`.** |
| `data/dr16q_prop_May01_2024.fits` | DR16Q spectral properties (Wu & Shen 2022). Line arrays such as `HBETA_BR` have 6 entries: [peak λ, 50% flux centroid λ, flux, log L_line, FWHM (km/s), rest EW (Å)]. Match to the catalog on PLATE/MJD/FIBERID. |

**Important: these are PSF-photometry light curves, not difference imaging.** For a resolved host galaxy, the PSF flux includes some host light, and how much depends on the seeing. Two consequences:

1. The host adds a constant flux that dilutes the AGN variability, more strongly in red bands. That breaks the model's power law for variability amplitude versus wavelength (`eta_sigma`).
2. Epoch-to-epoch seeing changes leak host flux into the light curve as fake variability.

The pipeline models neither. That is why you need to check extendedness (Task B).

**How the light curves are built.** `concat_light_curves` → `populate_sdss_fields` → `make_lc`. Rolling photometric outlier rejection is on by default (±30-day window, at least 8 neighbors). About 20% of points are too isolated to test. Survey offsets and per-survey jitter are fitted.

---

## 4. The model and the BLR–continuum covariance

Code: `src/qvc/light_curve/multiband_model_shared_latent_blr.py` (kernel `SharedLatentDiskBLRQS`), with priors and model assembly in `fit_light_curves.py`. `shared_latent_blr` is the only model; the older per-band DHO and CARMA(2,1) variants were removed.

### 4.1 Components

We model relative flux, r_b(t) = F_b(t)/F̄_b − 1. The magnitude likelihood is linearized in flux (m ≈ m̄ − 2.5 log10(1 + r)). Continuum and line flux add in flux space before the conversion to magnitudes.

- **Driver.** One unit-RMS damped harmonic oscillator x(t), overdamped, with two timescales (`tau_fast`, `tau_slow`). There is one driver per object, not one per band.
- **Continuum in band b.** The driver convolved with a narrow Erlang kernel (order 3) with mean lag τ_disk,b = `lag0` × (λ_b/λ_c)^{4/3}, the thin-disk scaling. The amplitude σ_b follows a power law in wavelength (`log_sigma_center0`, `eta_sigma`).
- **BLR in band b.** The driver convolved with a broad Erlang kernel (order 3) with mean τ_BLR,b (the free per-band lag) and width τ_BLR,b/√3, scaled by A_b = σ_UV × exp(`log_amp_ratio_blr`).

  The BLR term is always a delayed, smoothed copy of the driver. A long lag also means a broad kernel, so the echo is heavily smoothed.

  Lag prior: truncated normal on log rest-frame lag, centered at 100 d, width 1 dex, bounds 0.1–1000 d. Amplitude prior: ln(A/σ_UV) ~ TN(−1, 0.75), capped at 0.

Each Erlang chain is a small linear ODE, so the whole multiband process stays Markovian (26 states for 4 bands). The likelihood then costs O(N) time through tinygp's quasiseparable solver. `use_fast_solver=True` adds a fused scan with a custom adjoint.

Nuisance terms: mean per band, linear trend, survey offsets, and per-survey/per-band jitter.

### 4.2 Where the lag information comes from

Write each band as r_b = c_b + ℓ_b, where c_b = K^disk_b ⊛ x and ℓ_b = K^BLR_b ⊛ x. The GP covariance between band b at time t and band b′ at time t′ is

```
C_bb'(Δt) = (K_b ⋆ K_b' ⋆ k_x)(Δt),   K_b = σ_b K^disk_b + A_b K^BLR_b
          = σ_b σ_b' k_cc + σ_b A_b' k_cℓ + A_b σ_b' k_ℓc + A_b A_b' k_ℓℓ
```

Here k_x is the driver's autocovariance and ⋆ is cross-correlation. The **continuum–BLR cross term k_cℓ** is the driver autocovariance smoothed by both kernels, and it peaks at Δt ≈ τ_BLR,b′ − τ_disk,b. That term carries the lag.

In practice the information comes from comparing bands:

- A band containing a strong line has a covariance with an extra delayed, smoothed shoulder. A line-free band (or the continuum part of the same band) does not.
- The model learns the continuum shape from all bands together through the shared driver. Whatever delayed excess is left in the line band constrains τ_BLR and A_b.

This is the same idea as photometric reverberation mapping (Haas et al. 2011; Chelouche & Daniel 2012; Zu et al. 2016, JAVELIN) but with every band fitted jointly.

### 4.3 What makes a lag measurable

The lag signal scales with **A_b/σ_b, the line's share of the variable flux in that band.** That share tracks the line's equivalent width compared with the filter width. We estimate the band fraction as

```
f_line ≈ EW_rest × (1+z) × ⟨T⟩_line / ∫T dλ
```

where ⟨T⟩_line is the filter throughput averaged over a Gaussian line profile with the catalog FWHM. This handles lines near a filter edge.

Other things that matter:

- **Driver timescale vs. lag.** The driver needs variability on timescales near τ_BLR. τ is only recovered reliably when it is shorter than about baseline/3.
- **Cadence.** SDSS is dense from 1998 to 2008, then there is a gap until PS1, then ZTF from 2018 on. Lags above about 300 d sit in the regime where the seasonal gaps and the inter-survey offsets fight with the echo.
- **No host contamination** (Section 3).

At f_line ≈ 5% (the typical value in our first sample), A_b is degenerate with the continuum shape and the lag posterior mostly shows systematics. Expect multimodality and pinning at the edges of the lag grid.

---

## 5. Inference: MAP + Laplace and the lag profile

Scripts are in `pipeline/`.

- `qvcfit.py`: builds one object's NumPyro model exactly as the pipeline does: outlier rejection, `make_lc`, jitter and survey masks, flux-linearized `shared_latent_blr` with `refinement_iters=1`, fast solver. It reads the parquet files filtered to that object.
- `fit_object.py <objectId> <outdir>`:
  - MAP via SciPy L-BFGS-B in coordinates whitened by a finite-difference (FD) Hessian, from 4 starts (median plus 3 perturbations).
  - Laplace covariance from an FD Hessian of the autodiff gradient. `jax.hessian` returns NaN because the fast solver's `custom_vjp` has no forward mode.
  - Writes `fits/<oid>.json` and `fits/<oid>_laplace.npz`.
- `lag_profile.py <oid> <band> <map_npz> <outdir>`: the lag marginal, which is the actual lag measurement. On a 24-point grid of rest-frame lags from 3 to 900 d, it fixes the lag, re-optimizes everything else from warm starts in two passes, and combines

  ```
  log p(L) ≈ −U(ŷ_L, L) + log|∂u/∂L| − ½ log det H(ŷ_L)
  ```

  This is an INLA-style integrated-Laplace marginal. It catches multimodality that a single Gaussian Laplace misses. A plain Laplace lag posterior is miscalibrated (PSIS k̂ ≈ 0.8). Don't use it for the lag.

  Flags: **multimodal** (a second peak holds at least 20% of the mass) and **edge-pinned** (the argmax is within 2 grid points of an end). "Clean" means neither flag is set.
- `diagnostic_figure.py`: per-object PDF with:
  - each band's light curve with the conditional model;
  - the continuum-only model shifted to zero lag;
  - the lag posterior;
  - the SDSS spectrum over the filter curves.
- `analyze_line.py <run_dir> <line>`: lag vs. L plot, results CSV and `summary.txt`.
- `object_job.sh <oid>` runs fit → profile → figure for each selected line of one object. `driver.sh` runs `job_order.txt` 3 objects at a time.
- `dr16q_lines.py`: DR16Q matching, the line quality mask and in-band line fractions.
- `query_morphology.py`: SDSS and PS1 extendedness flags and the default extended-source cut.
- `line_ew.py`, `strong_lines.py`, `count_strong.py`, `strong10_plot.py`: line band fractions and the strong-line cuts.

**Reference R–L relations** (in `analyze_line.py`):

| Line | Relation | Notes |
|---|---|---|
| Hβ | log τ = 1.527 + 0.533 (log λL5100 − 44) | Bentz et al. 2013 |
| Hα | 1.5 × Hβ | Approximate |
| Mg II | log τ = 1.67 + 0.30 (log λL3000 − 44) | Approximate. Check against a current Mg II RM compilation and update. |
| C IV | none yet | Add one, e.g. from a recent C IV RM compilation. |

**Runtime:** run4 did 238 pairs (221 objects) in about 8.3 h with 3 processes, so roughly 6 min per object–line pair per process including the figure. One likelihood call takes about 9 ms with 54 parameters.

---

## 6. What to do

### Task A: build the strong-line sample (χ²_g > 20, line > 20% of band flux)

`pipeline/count_strong.py` already does the core selection. It writes `chi20_line_fractions.csv` with one row per object × line × band, including the quality flags from `pipeline/dr16q_lines.py` (see "Never trust a DR16Q line without `line_quality()`" in the pitfalls). Use only rows with `good == True`.

Current counts. Of 23,502 objects with χ²_g > 20 matched to DR16Q, **3,094 have a good-quality broad line above 20% of the flux in some band:**

| Line | u | g | r | i | z | Unique objects |
|---|---|---|---|---|---|---|
| Hα | 0 | 0 | 0 | 25 | 140 | 165 |
| Hβ | 0 | 0 | 0 | 20 | 194 | 214 |
| Mg II | 1 | 22 | 124 | 137 | 268 | 552 |
| C IV | 489 | 1637 | 259 | 12 | 1 | 2398 |

Before the quality mask these counts were 4,494 in total (Hβ 515, C IV 3,408). The extra objects were off-spectrum or unconstrained DR16Q line fits.

To do:

1. **No cadence cuts.** We decided against a minimum number of epochs; any cut there (e.g. 50 SDSS epochs) is arbitrary. Instead, tag each object with its line band and its number of epochs in that band, and report results against them.
   - z and u bands are SDSS and PS1 only; there is no ZTF z or u. Expect worse lag constraints there. Keep them, but compare against g/r/i.
   - Drop objects with ZTF rms above 0.6 mag; that indicates bad photometry.
2. **One row per object × line** in `selection.csv` (columns `objectId,z,band,logL,var_snr,line`).
   - Use `LOGL5100` for Hα and Hβ, and `qg_LogL3000` for Mg II and C IV.
   - If a line passes in two bands, keep the band with the larger fraction. Record the fraction.
3. **Order the jobs.** Generate `job_order.txt` round-robin across lines, so partial results cover every line.
4. **Prioritize.** C IV dominates the counts and has no reference relation. Run all of Hα, Hβ and Mg II first (about 930 objects), then a random subset of about 300 C IV objects.

### Task B: check extendedness (do this before interpreting any lag)

`pipeline/query_morphology.py <run_dir>` writes `<run_dir>/morphology.csv` for every object in the run's `selection.csv`:

- **SDSS DR16 PhotoObj**, matched by position within 2″: `type` and `psfMag − cModelMag` in g, r, i, z.
- **PS1 DR2 mean photometry**, by 2″ cone search: `MeanPSFMag − MeanKronMag` in g, r, i, z. Do not join on the catalog's `ps1objID`: `Catalog.parquet` stores it as a float, so its last digits are lost and almost no IDs match.
- **Default cut:** `extended = (sdss_type == 3) | (psfMag_r − cModelMag_r > 0.145)`. Drop extended objects for Hα and Hβ. Do **not** apply the PSF − cModel test in the line band: for faint, high-z objects (Mg II at z ≈ 2, Hβ in z), the i/z values are noise and flag about a third of them spuriously.
- Already in the catalog: `qg_fragal` (SED-based host fraction).

What the cut did (first Hα/Mg II run and the Hβ run, single-peaked lags not at the grid edge):

- **Hα:** 39 of 95 objects are extended. Their lags sit about 0.35 dex above the reference R–L, mostly at low luminosity, consistent with host light. Removing them cut the scatter from about 0.5 to 0.27 dex, with no offset, but the 31 remaining point sources span only log L5100 ≈ 43.8–44.6 and show no significant slope.
- **Hβ:** only 25 of 214 are extended (z ≈ 0.8), and they sit about 0.2 dex high. The Hβ problem is the noisy z-band photometry, not hosts.
- **Mg II:** posteriors are prior-dominated with or without the cut.

Still to do:

- Compare the SDSS flag with PS1 PSF − Kron (above about 0.05 means extended) and the PS1 point-source score (`ps_score`, Tachibana & Miller 2018). The Stripe 82 coadd (Annis et al. 2014) is deeper and better for faint hosts.
- Instead of dropping extended objects, try fitting them with the PSF-fraction correction (`--subtract_psf_constant_flux` needs the spectra-fit HDF5 PSF fractions). Report R–L results both with and without them.

### Task C: Laplace run on the strong-line sample

1. Copy `pipeline/` to a new directory next to it (e.g. `experiments/blr_lags/run5_strong20/`) and add `selection.csv` and `job_order.txt` from Task A. Output folders (`fits/`, `profiles/`, `plots/`, `logs/`) are created on first run.
2. Run it:

   ```bash
   cd experiments/blr_lags/run5_strong20 && sh driver.sh
   ```

   - The script `cd`s to the repo root itself.
   - Set `QVC_PYTHONPATH` only if you need an overlay, such as an equinox 0.12.2 install in a separate folder.
3. **Plan the compute.** About 1,200 pairs (930 Hα/Hβ/Mg II plus about 300 C IV) at 1–6 min each over 3 processes is roughly 7–40 h on a laptop. Use the cluster (`hpc_scripts/` has examples) and run one object per task. Each object is independent.
4. Run `analyze_line.py` per line.
   - Report offset, scatter, the lag–L slope with uncertainty, and the multimodal and edge-pinned fractions.
   - Report all of these against line fraction (20–30%, 30–50%, above 50%) and against band (well-sampled g/r/i vs. u/z).
5. **Spot-check diagnostic figures.** Look at about 20 per line.
   - Does the BLR component look like a delayed, smoothed echo of the continuum in the line band?
   - Is the lag driven by one season or by the survey gaps?

### Task D: verify the detections with nested sampling

The Laplace profile is an approximation. Before claiming a lag, verify it with nested sampling on a subset:

- **Candidates:** the clean detections with a 68% interval narrower than 0.3 dex.
- **Controls:** some multimodal objects.

Notes:

- **The CLI cannot do this yet.** `fit_light_curves.py` rejects `--fit_method ns` for the flux-linearized model, because the pipeline refines the linearization iteratively, and `ns` is also not in the argparse choices.
- **Our model is fixed after one linearization** (`refinement_iters=1`, as built by `qvcfit.build`), so you can sample it directly:

  ```python
  from numpyro.contrib.nested_sampling import NestedSampler
  obj, model, meta = qvcfit.build(oid)
  ns = NestedSampler(model, constructor_kwargs=dict(num_live_points=..., max_samples=...),
                     termination_kwargs=dict(dlogZ=0.1))
  ns.run(jax.random.PRNGKey(0))
  samples = ns.get_samples(jax.random.PRNGKey(1), 4000)   # samples["lag_blr"], etc.
  ```

- **Cost.** There are 34 parameters for a 4-band object, at about 9 ms per likelihood call. Plan for 10⁵–10⁶ likelihood calls per object, which is tens of minutes to hours.
  - Start with 2–3 objects to calibrate `num_live_points` (`align_nested_sampler_num_live_points` in `fit_light_curves.py` gives the default).
  - If it's too slow, fix the nuisance parameters (survey offsets, jitter, means) at their MAP values and sample the roughly 10 physical parameters. Check on one object that this doesn't shift the lag.
- **Compare with the Laplace profile.** Check that the NS lag marginal matches the Laplace-profile posterior in median, interval and number of modes. Compute the Bayes factor between the full model and the model with the BLR amplitude set to zero (the evidence comes free with NS). A real detection should have ln BF ≳ 3.
- **Save results.** Write `ns/<oid>.npz` (samples, log Z, its error) and add a column to the results CSV so the R–L plot can mark NS-confirmed points.

### Task E: write-up figures

For each line:

- lag vs. L with only NS-confirmed points, the reference relation, and a fitted slope with uncertainty;
- the fraction of detections vs. line fraction.

One summary figure should show how the selection (variability only → line fraction → extendedness → NS confirmation) changes the scatter.

---

## 7. Pitfalls we hit already

- **Never trust a DR16Q line without `line_quality()`.** DR16Q reports a fit for every line, even when the line falls off the end of the spectrum (SDSS-I/II: 3,800–9,200 Å; BOSS: 3,600–10,350 Å) or is not detected. Those fits run away to absurd fluxes and EWs, which gave in-band fractions of 10^8 and put the junk at the top of a fraction-sorted queue (e.g. Hβ at z ≈ 1.2, past the BOSS red end). `pipeline/dr16q_lines.line_quality()` requires: line center ±2 FWHM inside the spectrum, at least 90% of the fitting-window pixels (`*_COMP_STAT[1]`), EW S/N ≥ 3, and log EW below the per-line median + 3 robust σ. `count_strong.py` asserts that no good line has a fraction above 3.
- **Rebuild instead of monkeypatching.** `eqx.filter_jit` caches traces, so monkeypatching a method after the first call is silently ignored. This produced wrong "continuum" overlays once. Compute derived predictions with the function directly, and check that total = continuum + BLR.
- **Compare lags between MAP starts, not just U.** Two starts with nearly equal U can sit at very different lags.
- **Lag label switching** between the disk and BLR components is rare (11/240 bands), but check `first_run_diagnostics/lag_roles.csv` and the fitted disk lags when a band's "BLR lag" is less than about 2 d.
- **Edge-pinned lags are non-detections.** Treat them that way, even when the interval looks narrow.
- **Systematics dominate at low line fraction.** A simulation-based S/N metric on cadence and errors did not predict which real objects would give good lags.
- **Keep everything on persistent storage.** Earlier runs lived in a temporary directory.

## 8. File map

```
experiments/blr_lags/
  README.md                 this file
  pipeline/                 committed scripts; copy this folder for each new run
    qvcfit.py fit_object.py lag_profile.py diagnostic_figure.py analyze_line.py
    object_job.sh driver.sh
    dr16q_lines.py            DR16Q matching, line quality mask, in-band line fractions
    query_morphology.py       SDSS/PS1 extendedness flags for a run (morphology.csv)
    line_ew.py strong_lines.py strong10_plot.py count_strong.py
  first_run_diagnostics/    (local) lag_roles.csv, outcome_correlates.csv from the first 30+30 run
  run4_seeing/              (local) results of the 4-line, 60-object run (fitted with seeing terms)
    selection.csv line_props.csv chi20_line_fractions.csv
    fits/ profiles/ plots/<line>/   (per-object diagnostics, R–L plots, summaries)
```
