"""Regenerate only the sep22b wavelength diagnostic and its four slope macros."""
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd

from qvc.hubble.sigma_tau_lambda_fit import fit_sigma_tau_lambda_broken_pl, slope_bootstrap_band

RUN = Path('plots/hubble/sep22b_wawider_paper_sed13spec11joint12loo105rhat105_bsm02_2_standard')
H5 = Path('results/data/sep14_1232pm_chisq_slb_faster_etamodified_iters3disk3_svi10000lr0003w1000s500_specaug31w500s500_83cb31d_chisq.h5')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def replace_macros(text, result):
    from qvc.hubble.hubble_utils import format_result_errors

    macros = {}
    for target in ('sigma', 'tau'):
        fit = result[f'fit_{target}']
        for i, side in enumerate(('Blue', 'Red')):
            value = fit[f'd{i + 1}']
            low, high = fit['slope_percentiles'][i]
            if not low <= value <= high:
                raise ValueError('Full-sample slope outside bootstrap interval')
            name = f'resultFiducialEta{target.capitalize()}{side}Lambda'
            # Match the main TeX export's precision and uncertainty floor.
            value_tex = format_result_errors(
                value, err_lower=max(value-low, 0.001),
                err_upper=max(high-value, 0.001), nd=3,
                symmetric_if_equal=True,
            )
            line = '\\newcommand{\\' + name + '}{\\ensuremath{' + value_tex + '}}'
            text, n = re.subn(r'^\\newcommand\{\\' + name + r'\}[^\n]*', lambda m: line, text, flags=re.M)
            if n != 1:
                raise ValueError(f'Expected exactly one macro {name}, found {n}')
            macros[name] = value_tex
    return text, macros


def main():
    from qvc.hubble.hubble_plotting import plot_sigma_tau_vs_lambda_broken_pl_fit

    table = RUN / 'Flatw0waCDM_joint/agn_table_all_fields.csv'
    selected = pd.read_csv(table, dtype={'object_id': str})
    ids = selected.object_id.to_numpy()
    assert len(ids) == len(set(ids)) == 4759
    columns = ['z', 'log_sigma_uv', 'log_tau_uv_rf', 'tau_fast_driver',
               'tau_slow_driver', 'lag0', 'lambda_center_rf']
    columns += [f'{prefix}_{band}' for prefix in ('lag_disk', 'log_sigma_band') for band in 'ugri']
    with h5py.File(H5) as source:
        source_ids = source['object_id'].asstr()[:]
        assert len(set(source_ids)) == len(source_ids)
        lookup = {value: i for i, value in enumerate(source_ids)}
        rows = np.array([lookup[value] for value in ids])
        df = pd.DataFrame({key: source[key][:][rows] for key in columns})
    for key in columns:
        np.testing.assert_allclose(df[key], selected[key], rtol=1e-12, atol=1e-12, equal_nan=True)
    kwargs = dict(bands=tuple('ugri'), fit_mode='common_bootstrap', bootstrap_replicates=1000,
                  bootstrap_seed=2500, disk_order=3, lam_s=2500.)
    result = fit_sigma_tau_lambda_broken_pl(df, include_plot_payload=True, **kwargs)
    assert result['tau_data']['value_definition'] == 'continuum_median_parameter_approximation'
    expected = {'sigma': [-0.6377969782942958, -0.6301113030675505],
                'tau': [0.02283127784700207, 0.014553587625143962]}
    grid = np.unique(np.r_[np.logspace(2.81, 3.89, 600), 2500.])
    arrays = {'object_id': ids.astype(str), 'wavelength_grid': grid}
    fits = {}
    for target in ('sigma', 'tau'):
        fit = result[f'fit_{target}']
        np.testing.assert_allclose([fit['d1'], fit['d2']], expected[target], rtol=1e-9, atol=1e-10)
        assert fit['bootstrap_replicates_valid'] == 1000
        assert len(np.unique(result[f'{target}_data']['group'])) == 4759
        arrays[f'{target}_bootstrap_slopes'] = fit['bootstrap_slopes']
        arrays[f'{target}_curve_percentiles'] = slope_bootstrap_band(fit, grid, lam_s=2500.)
        assert np.all(arrays[f'{target}_curve_percentiles'][:, grid == 2500.] == 0.)
        fits[target] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in fit.items() if k != 'bootstrap_slopes'}
    tex = RUN / 'model_compare/param_results_fiducial.tex'
    original = tex.read_text()
    updated, macros = replace_macros(original, result)
    diag = RUN / 'diagnostics'
    pdf = diag / 'sigma_tau_vs_lambda_broken_pl_fit_postcut.pdf'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for path in (pdf, tex):
        shutil.copy2(path, path.with_name(path.name + '.' + stamp + '.bak'))
    metadata = dict(mode='common_bootstrap', seed=2500, replicates=1000, n_objects=4759,
                    break_angstrom=2500., intercept=0., weighted=False,
                    uncertainty='16th-84th percentiles of whole-quasar bootstrap fitted relations; no measurement-error perturbation',
                    tau_definition=result['tau_data']['value_definition'], disk_order=3,
                    membership_sha256=hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest(),
                    ordered_membership_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                    inputs={str(p.resolve()): sha256(p) for p in (table, H5, Path(__file__), Path('src/qvc/hubble/sigma_tau_lambda_fit.py'), Path('src/qvc/hubble/hubble_plotting.py'))},
                    previous_pdf_sha256=sha256(pdf), fits=fits, macros=macros)
    output = plot_sigma_tau_vs_lambda_broken_pl_fit(df, plot_path=str(RUN), **kwargs)
    assert output is not None and pdf.is_file()
    tex.write_text(updated)
    stem = diag / 'sigma_tau_vs_lambda_broken_pl_fit_postcut_common_bootstrap'
    np.savez_compressed(str(stem)+'.npz', **arrays)
    metadata['pdf_sha256'] = sha256(pdf)
    metadata['tex_sha256'] = sha256(tex)
    Path(str(stem)+'.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'fits': fits, 'macros': macros, 'pdf': str(pdf)}, indent=2))


if __name__ == '__main__':
    main()
