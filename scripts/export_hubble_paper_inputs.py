#!/usr/bin/env python
"""Build the versioned minimal inputs used by run_hubble_paper.xonsh."""

from argparse import ArgumentParser
from pathlib import Path

from qvc.hubble.minimal_inputs import export_minimal_hubble_inputs


DEFAULT_LIGHT_CURVE_SOURCE = Path(
    "results/data/sep14_1232pm_chisq_slb_faster_etamodified_iters3disk3_"
    "svi10000lr0003w1000s500_specaug31w500s500_83cb31d_chisq.h5"
)
DEFAULT_SPECTRA_SOURCE = Path(
    "results/data/jaxqsofit/sep25_0443pm_spectrafit_24f6f85_"
    "aug31_w500s500_d01c034_rerun.h5"
)
DEFAULT_LIGHT_CURVE_OUTPUT = Path(
    "results/data/hubble_paper_light_curve_minimal_v1.h5"
)
DEFAULT_SPECTRA_OUTPUT = Path(
    "results/data/hubble_paper_spectra_minimal_v1.h5"
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--light-curve-source", type=Path, default=DEFAULT_LIGHT_CURVE_SOURCE)
    parser.add_argument("--spectra-source", type=Path, default=DEFAULT_SPECTRA_SOURCE)
    parser.add_argument("--light-curve-output", type=Path, default=DEFAULT_LIGHT_CURVE_OUTPUT)
    parser.add_argument("--spectra-output", type=Path, default=DEFAULT_SPECTRA_OUTPUT)
    args = parser.parse_args()
    count = export_minimal_hubble_inputs(
        args.light_curve_source,
        args.spectra_source,
        args.light_curve_output,
        args.spectra_output,
    )
    print(f"Wrote {count} matched successful rows to:")
    print(args.light_curve_output)
    print(args.spectra_output)


if __name__ == "__main__":
    main()
