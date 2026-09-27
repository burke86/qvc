"""Plot spectral component fractions against redshift from a merged catalog."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

plt.style.use(Path(__file__).resolve().parents[1] / "src/qvc/hubble/style.mplstyle")


PANELS = (
    ("f_bc_3000", r"$f_{\rm BC}$"),
    ("f_fe_uv_3000", r"$f_{\rm FeII}$"),
    ("f_host_2500_psf", r"$f_{\rm host,2500\,\AA}^{\rm PSF}$"),
)
Y_MIN = 2e-6
Y_MAX = 5e1
CONTOUR_PROBABILITIES = (0.683, 0.954)


def add_density_contours(ax, z, values):
    """Draw enclosed-probability contours in redshift and log-fraction space."""
    inside = (
        np.isfinite(z) & np.isfinite(values)
        & (values >= Y_MIN) & (values <= Y_MAX)
    )
    if np.count_nonzero(inside) < 20:
        return
    xlim = ax.get_xlim()
    log_edges = np.linspace(np.log10(Y_MIN), np.log10(Y_MAX), 91)
    x_edges = np.linspace(*xlim, 91)
    histogram, _, _ = np.histogram2d(
        z[inside], np.log10(values[inside]), bins=(x_edges, log_edges)
    )
    density = gaussian_filter(histogram, sigma=2.5)
    ranked = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ranked) / ranked.sum()
    thresholds = [
        ranked[np.searchsorted(cumulative, probability)]
        for probability in CONTOUR_PROBABILITIES
    ]
    if not (0 < thresholds[1] < thresholds[0]):
        return
    ax.contour(
        0.5 * (x_edges[:-1] + x_edges[1:]),
        10 ** (0.5 * (log_edges[:-1] + log_edges[1:])),
        density.T,
        levels=[thresholds[1], thresholds[0]],
        colors="black", linestyles=["--", "-"],
        linewidths=1.7, zorder=4,
    )


def plot(z, fractions, rows, output, xlim):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for index, (ax, (name, label)) in enumerate(zip(axes, PANELS)):
        values = fractions[name]
        finite = rows & np.isfinite(z) & np.isfinite(values)
        valid = finite & (values >= Y_MIN)
        below = finite & (values < Y_MIN)
        ax.plot(
            z[valid], values[valid], linestyle="None", marker="o",
            markersize=2.5, alpha=0.28, color="tab:red", rasterized=True,
        )
        if below.any():
            ax.plot(
                z[below], np.full(np.count_nonzero(below), Y_MIN * 3),
                linestyle="None", marker=r"$\downarrow$", markersize=5,
                alpha=0.45, color="tab:red", rasterized=True, zorder=3,
            )
        ax.set_xlim(*xlim)
        add_density_contours(ax, z[rows], values[rows])
        ax.set_xlabel(r"$z$")
        ax.set_ylabel("Component fraction" if index == 0 else "")
        ax.set_yscale("log")
        ax.set_ylim(Y_MIN, Y_MAX)
        handles = [
            Line2D([], [], marker="o", linestyle="None", markersize=5,
                   markerfacecolor="tab:red", markeredgecolor="tab:red",
                   alpha=1, label=label),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=12, frameon=True,
                  facecolor="white", framealpha=0.75, edgecolor="none")
        print(f"{name}: {np.count_nonzero(valid)} points, "
              f"{np.count_nonzero(below)} below-axis arrows / "
              f"{np.count_nonzero(rows)} rows")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


def current_tier2_ids(hubble_data, catalog_path):
    # Match the Sep 27 paper launcher, including grid-dependent redshift bounds.
    os.environ.update({
        "QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH": "0.2",
        "QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH": "0.2",
        "QVC_CUT_SED_REDUCED_CHI2_MAX": "1.3",
        "QVC_CUT_LOO_CHI2_EFF_MAX": "1.05",
        "QVC_CUT_SPECTRAL_RHAT_MAX": "1.05",
        "QVC_CUT_VARIABILITY_CHI_SQ_RED_G_MIN": "20",
        "QVC_CUT_COMPLETENESS_MAG_2500_MIN": "17",
        "QVC_CUT_COMPLETENESS_MAG_2500_MAX": "24",
    })
    from qvc.hubble.hubble_utils import load_agn_data

    with TemporaryDirectory(prefix="qvc-spectral-cuts-") as scratch:
        selected, _, _ = load_agn_data(
            str(hubble_data), spectra_fit_csv=[str(catalog_path)], cut_tier="2",
            magnitude_convention="dereddened",
            completeness_magnitude="attenuated",
            completeness_magnitude_support_mode="hard-cut",
            enforce_completeness_support=True,
            return_completeness_parent=True,
            z_range=(0.7, 3.16), plot_path=scratch, plot_diagnostics=False,
        )
    ids = set(selected["object_id"].astype(str))
    if len(selected) != 5402 or len(ids) != 5402:
        raise ValueError(f"Tier-2 selection differs from the Sep 27 run: {len(selected)} rows")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hubble-data", type=Path,
                        help="Reproduce the Sep 27 tier-2 selection and write pre/post-cut PDFs")
    args = parser.parse_args()

    with h5py.File(args.catalog, "r") as handle:
        catalog = handle["catalog"]
        z = np.asarray(catalog["z"][:], dtype=float)
        fractions = {name: np.asarray(catalog[name][:], dtype=float) for name, _ in PANELS}
        object_ids = catalog["object_id"][:].astype(str)

    finite_z = z[np.isfinite(z)]
    z_min, z_max = np.min(finite_z), np.max(finite_z)
    padding = 0.02 * (z_max - z_min)
    common_xlim = (max(0.0, z_min - padding), z_max + padding)

    if args.hubble_data is None:
        output = args.output or args.catalog.with_name(
            args.catalog.stem + "_spectral_fractions_vs_redshift.pdf"
        )
        plot(z, fractions, np.ones(len(z), dtype=bool), output, common_xlim)
        return

    if args.output is not None:
        parser.error("--output cannot be combined with --hubble-data")
    selected_ids = current_tier2_ids(args.hubble_data, args.catalog)
    selected_rows = np.isin(object_ids, list(selected_ids))
    if np.count_nonzero(selected_rows) != len(selected_ids):
        raise ValueError("Tier-2 object IDs do not match the spectral catalog one-to-one")
    stem = args.catalog.stem + "_spectral_fractions_vs_redshift"
    plot(z, fractions, np.ones(len(z), dtype=bool),
         args.catalog.with_name(stem + "_precut.pdf"), common_xlim)
    plot(z, fractions, selected_rows,
         args.catalog.with_name(stem + "_postcut.pdf"), common_xlim)


if __name__ == "__main__":
    main()
