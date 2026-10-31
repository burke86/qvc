import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.chdir(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qvc.light_curve import multiband_generate_lc
from qvc.light_curve.multiband_generate_lc import (
    DR16Q_POPULATE_COLUMNS,
    load_dr16q_populate_columns,
    populate_sdss_fields,
)


def _build_catalog_table():
    return Table(
        {
            "objectId": np.array(["obj_a", "obj_dup", "obj_dup"]),
            "RA": np.array([10.0, 20.0, 21.0]),
            "DEC": np.array([0.0, 1.0, 1.5]),
            "z": np.array([0.5, 1.2, 9.9]),
        }
    )


def _build_dr16q_table():
    return Table(
        {
            "RA": np.array([10.0 + 1.0e-4, 20.0 + 1.0e-4, 80.0]),
            "DEC": np.array([0.0, 1.0, 0.0]),
            "Z_SYS": np.array([0.5, 1.2, 3.0]),
            "Z_SYS_ERR": np.array([0.01, 0.02, 0.03]),
            "SDSS_NAME": np.array(["A", "B", "C"]),
            "EBV": np.array([0.03, 0.04, 0.05]),
            "LOGLBOL": np.array([45.1, 46.2, 47.3]),
            "LOGLBOL_ERR": np.array([0.1, 0.2, 0.3]),
            "LOGL3000": np.array([44.0, 45.0, 46.0]),
            "LOGL3000_ERR": np.array([0.4, 0.5, 0.6]),
            "LOGL5100": np.array([44.1, 45.2, 46.3]),
            "LOGL5100_ERR": np.array([0.01, 0.02, 0.03]),
            "LOGMBH": np.array([8.1, 8.2, 8.3]),
            "LOGMBH_ERR": np.array([0.11, 0.12, 0.13]),
            "LOGLEDD_RATIO": np.array([-0.5, -0.4, -0.3]),
            "LOGLEDD_RATIO_ERR": np.array([0.05, 0.04, 0.03]),
            "PLATE": np.array([1000, 1001, 1002]),
            "MJD": np.array([55000, 55001, 55002]),
            "FIBERID": np.array([10, 11, 12]),
            "SN_MEDIAN_ALL": np.array([15.0, 16.0, 17.0]),
        }
    )


def _mock_table_read(monkeypatch, cat, dr16q):
    calls = []

    monkeypatch.setattr(
        "qvc.light_curve.multiband_generate_lc.resolve_qvc_data_path",
        lambda path: str(path),
    )

    def fake_read(path, *args, **kwargs):
        path = str(path)
        calls.append((path, kwargs.copy()))
        if path.endswith("Catalog.parquet"):
            include_names = kwargs.get("include_names")
            if include_names is None:
                return cat.copy()
            missing = [name for name in include_names if name not in cat.colnames]
            if missing:
                raise KeyError(", ".join(missing))
            return cat[include_names].copy()
        if path.endswith("dr16q_prop_May01_2024.fits"):
            assert kwargs.get("hdu") == 1
            return dr16q.copy()
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr("qvc.light_curve.multiband_generate_lc.Table.read", fake_read)
    # Bypass the on-disk DR16Q column cache so the mocked table is what gets matched.
    monkeypatch.setattr(
        "qvc.light_curve.multiband_generate_lc.load_dr16q_populate_columns",
        lambda path: fake_read(path, hdu=1),
    )
    return calls


def test_populate_sdss_fields_strict_success(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    calls = _mock_table_read(monkeypatch, cat, dr16q)

    objs = [
        {"object_id": "obj_a"},
        {"object_id": "obj_dup"},
    ]

    result = populate_sdss_fields(objs, progress_bar=True)

    assert result is objs
    assert result[0]["ra"] == 10.0
    assert result[0]["dec"] == 0.0
    assert result[0]["z"] == 0.5
    assert result[0]["z_err"] == 0.01
    assert result[0]["sdss_name"] == "A"
    assert result[0]["plate"] == 1000
    assert result[0]["SN_MEDIAN_ALL"] == 15.0
    np.testing.assert_allclose(result[0]["LOGLBOL"], 45.1)
    np.testing.assert_allclose(result[0]["LOGLBOL_ERR"], 0.1)
    np.testing.assert_allclose(result[0]["LOGLBOL_CORRECTED"], np.log10(5.15) + 44.0)
    np.testing.assert_allclose(result[0]["LOGLBOL_CORRECTED_ERR"], 0.4)

    assert result[1]["ra"] == 20.0
    assert result[1]["dec"] == 1.0
    assert result[1]["z"] == 1.2
    assert result[1]["z_err"] == 0.02
    assert result[1]["sdss_name"] == "B"
    assert result[1]["LOGL5100"] == 45.2
    assert result[1]["fiberid"] == 11
    assert result[1]["SN_MEDIAN_ALL"] == 16.0
    np.testing.assert_allclose(result[1]["LOGLBOL"], 46.2)
    np.testing.assert_allclose(result[1]["LOGLBOL_ERR"], 0.2)
    np.testing.assert_allclose(result[1]["LOGLBOL_CORRECTED"], 46.2)
    np.testing.assert_allclose(result[1]["LOGLBOL_CORRECTED_ERR"], 0.2)

    parquet_calls = [kwargs for path, kwargs in calls if path.endswith("Catalog.parquet")]
    assert len(parquet_calls) == 1
    assert parquet_calls[0]["include_names"] == ["objectId", "RA", "DEC", "z"]


def test_populate_sdss_fields_raises_on_missing_catalog_match(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Missing S82 catalog match.*missing"):
        populate_sdss_fields([{"object_id": "obj_a"}, {"object_id": "missing"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_missing_dr16q_match(monkeypatch):
    cat = _build_catalog_table()
    cat.add_row(("obj_far", 40.0, -5.0, 2.0))
    dr16q = _build_dr16q_table()
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Missing DR16Q match within 1 arcsec.*obj_far"):
        populate_sdss_fields([{"object_id": "obj_a"}, {"object_id": "obj_far"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_missing_z_sys_err_column(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    dr16q.remove_column("Z_SYS_ERR")
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"DR16Q table missing required column\(s\): Z_SYS_ERR"):
        populate_sdss_fields([{"object_id": "obj_a"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_redshift_disagreement(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    dr16q["Z_SYS"][1] = 1.5
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Redshift disagreement exceeds tolerance .*obj_dup"):
        populate_sdss_fields([{"object_id": "obj_a"}, {"object_id": "obj_dup"}], progress_bar=False)


def test_populate_sdss_fields_raises_on_nonfinite_z_sys_err(monkeypatch):
    cat = _build_catalog_table()
    dr16q = _build_dr16q_table()
    dr16q["Z_SYS_ERR"][0] = np.nan
    _mock_table_read(monkeypatch, cat, dr16q)

    with pytest.raises(ValueError, match=r"Missing DR16Q Z_SYS_ERR.*obj_a"):
        populate_sdss_fields([{"object_id": "obj_a"}], progress_bar=False)


def test_populate_sdss_fields_falls_back_to_catalog_z_sys(monkeypatch):
    cat = _build_catalog_table()
    cat.rename_column("z", "Z_SYS")
    dr16q = _build_dr16q_table()
    calls = _mock_table_read(monkeypatch, cat, dr16q)

    result = populate_sdss_fields([{"object_id": "obj_a"}], progress_bar=False)

    assert result[0]["z"] == 0.5
    assert result[0]["z_err"] == 0.01

    parquet_calls = [kwargs for path, kwargs in calls if path.endswith("Catalog.parquet")]
    assert len(parquet_calls) == 2
    assert parquet_calls[0]["include_names"] == ["objectId", "RA", "DEC", "z"]
    assert parquet_calls[1]["include_names"] == ["objectId", "RA", "DEC", "Z_SYS"]


def test_stone_redshift_survives_enrichment(monkeypatch):
    _mock_table_read(monkeypatch, _build_catalog_table(), _build_dr16q_table())
    obj = {"object_id": "obj_a", "stone_Z": 0.51}
    result = populate_sdss_fields([obj], preserve_stone_redshift=True)[0]
    assert result['z'] == 0.51
    assert result['catalog_z'] == 0.5


def _write_dr16q_fits(path, table=None):
    table = _build_dr16q_table() if table is None else table
    table["UNUSED_WIDE_COLUMN"] = np.zeros((len(table), 50))
    table.write(path, format="fits", overwrite=True)
    # The reference is what the uncached path returns: Table.read of the FITS.
    return Table.read(path, hdu=1)


def _assert_tables_equal(expected, actual):
    assert actual.colnames == list(DR16Q_POPULATE_COLUMNS)
    for name in DR16Q_POPULATE_COLUMNS:
        expected_values = [np.asarray(expected[name])[i] for i in range(len(expected))]
        actual_values = [np.asarray(actual[name])[i] for i in range(len(actual))]
        assert actual_values == expected_values
        assert [type(v) for v in actual_values] == [type(v) for v in expected_values]


def test_dr16q_column_cache_written_and_reused(tmp_path, monkeypatch):
    fits_path = tmp_path / "dr16q_prop.fits"
    expected = _write_dr16q_fits(fits_path)
    cache_path = tmp_path / "dr16q_prop_populate_columns.parquet"

    first = load_dr16q_populate_columns(fits_path)
    assert cache_path.exists()
    _assert_tables_equal(expected, first)

    def fail_read(*args, **kwargs):
        raise AssertionError("FITS should not be re-read when the cache is current")

    monkeypatch.setattr(multiband_generate_lc.Table, "read", fail_read)
    _assert_tables_equal(expected, load_dr16q_populate_columns(fits_path))


def test_dr16q_column_cache_rebuilt_when_fits_is_newer(tmp_path):
    fits_path = tmp_path / "dr16q_prop.fits"
    _write_dr16q_fits(fits_path)
    load_dr16q_populate_columns(fits_path)
    cache_path = tmp_path / "dr16q_prop_populate_columns.parquet"

    updated = _build_dr16q_table()
    updated["Z_SYS"][0] = 0.75
    updated = _write_dr16q_fits(fits_path, updated)
    newer = cache_path.stat().st_mtime + 10
    os.utime(fits_path, (newer, newer))

    _assert_tables_equal(updated, load_dr16q_populate_columns(fits_path))


def test_dr16q_column_cache_used_without_fits(tmp_path):
    fits_path = tmp_path / "dr16q_prop.fits"
    expected = _write_dr16q_fits(fits_path)
    load_dr16q_populate_columns(fits_path)
    fits_path.unlink()

    _assert_tables_equal(expected, load_dr16q_populate_columns(fits_path))
