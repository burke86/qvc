"""DR16Q (Wu & Shen 2022) broad-line properties: catalog matching, quality mask, and in-band line fractions.

DR16Q reports a fit for every line even when it falls off the spectrum or is unconstrained; those fits run away
to absurd fluxes/EWs (e.g. Hbeta at z ~ 1.2, past the BOSS red end). Use line_quality() before trusting any EW.
"""
import numpy as np, pandas as pd, speclite.filters
from astropy.io import fits

C_KMS = 2.998e5
DR16Q = "data/dr16q_prop_May01_2024.fits"
LINE_COL = dict(Halpha="HALPHA_BR", Hbeta="HBETA_BR", MgII="MGII_BR", CIV="CIV")
STAT_COL = dict(Halpha="HA_COMP_STAT", Hbeta="HB_COMP_STAT", MgII="MGII_COMP_STAT", CIV="CIV_COMP_STAT")
BOSS_MIN_PLATE = 3500  # SDSS-I/II plates are below this, BOSS/eBOSS plates above
WAVE_RANGE = dict(sdss=(3800.0, 9200.0), boss=(3600.0, 10350.0))
COVER_FWHM = 2.0  # line center +- this many FWHM must lie inside the spectrum
MIN_NPIX_FRAC = 0.9  # fitting-window pixels relative to the full window
MIN_EW_SNR = 3.0
EW_OUTLIER_SIGMA = 3.0  # robust sigma in log EW above the per-line median
MAX_SANE_FRAC = 3.0
MAX_REDCHI2 = 2.0  # reduced chi2 of the line-complex fit and of the continuum fit (the EW needs both)


def load_dr16q():
    return fits.open(DR16Q, memmap=True)[1].data


def match_dr16q(df, d):
    """Add the DR16Q row ``index`` to df (needs PLATE/MJD/FIBERID); unmatched rows are dropped."""
    key = pd.DataFrame(dict(PLATE=d["PLATE"], MJD=d["MJD"], FIBERID=d["FIBERID"])).reset_index()
    return df.merge(key, on=["PLATE", "MJD", "FIBERID"], how="inner")


def line_quality(d, idx, z, plate, line):
    """Per-object quality flags for one broad line; ``good`` is True only if every check passes."""
    v = np.asarray(d[LINE_COL[line]][idx], float)
    e = np.asarray(d[LINE_COL[line] + "_ERR"][idx], float)
    peak, fwhm, ew, ew_err = v[:, 0], v[:, 4], v[:, 5], e[:, 5]
    boss = np.asarray(plate) >= BOSS_MIN_PLATE
    lo = np.where(boss, WAVE_RANGE["boss"][0], WAVE_RANGE["sdss"][0])
    hi = np.where(boss, WAVE_RANGE["boss"][1], WAVE_RANGE["sdss"][1])
    lam = peak * (1 + z)
    half = COVER_FWHM * fwhm / C_KMS * lam
    stat = np.asarray(d[STAT_COL[line]][idx], float)  # [reduced chi2, npix]
    npix, line_chi2 = stat[:, 1], stat[:, 0]
    conti_chi2 = np.asarray(d["CONTI_STAT"][idx], float)[:, 1]  # [npix, reduced chi2]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = pd.DataFrame(dict(
            finite=np.isfinite(peak) & np.isfinite(fwhm) & np.isfinite(ew) & (peak > 0) & (fwhm > 0) & (ew > 0),
            covered=(lam - half > lo) & (lam + half < hi),
            full_window=npix >= MIN_NPIX_FRAC * np.nanmax(npix),
            measured=ew / ew_err >= MIN_EW_SNR,
            good_fit=(line_chi2 > 0) & (line_chi2 <= MAX_REDCHI2) & (conti_chi2 > 0) & (conti_chi2 <= MAX_REDCHI2),
        ))
        pre = q.all(axis=1).to_numpy()
        lew = np.log10(np.where(pre, ew, np.nan))
    med = np.nanmedian(lew)
    sig = 1.4826 * np.nanmedian(np.abs(lew - med))
    q["not_outlier"] = ~(np.log10(np.where(ew > 0, ew, np.nan)) > med + EW_OUTLIER_SIGMA * sig)
    q["good"] = q.all(axis=1)
    return q


def band_fractions(d, idx, z, line, bands="ugriz"):
    """In-band line/continuum flux fraction EW_obs <T>_line / int T dlam for each band (Gaussian line, catalog FWHM)."""
    filt = {b: speclite.filters.load_filter(f"sdss2010-{b}") for b in bands}
    a = np.asarray(d[LINE_COL[line]][idx], float)
    peak, fwhm, ew = a[:, 0], a[:, 4], a[:, 5]
    ok = np.isfinite(peak) & np.isfinite(fwhm) & np.isfinite(ew) & (ew > 0) & (peak > 0) & (fwhm > 0)
    lam_obs = peak * (1 + z)
    sig = lam_obs * fwhm / C_KMS / 2.3548
    u = np.linspace(-5, 5, 201)
    prof = np.exp(-0.5 * u ** 2)
    out = {}
    for b, f in filt.items():
        lam = lam_obs[:, None] + sig[:, None] * u[None, :]
        T = np.where(ok[:, None], f(np.nan_to_num(lam, nan=1.0).clip(1, 2e4)), 0.0)
        norm = np.trapezoid(f(f.wavelength), f.wavelength)
        out[b] = np.where(ok, ew * (1 + z) * np.trapezoid(T * prof, u, axis=1) / np.trapezoid(prof, u) / norm, np.nan)
    return out
