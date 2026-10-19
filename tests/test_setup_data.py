import hashlib
import io
from email.message import Message
import zipfile

import pytest

from qvc import setup_data


def test_paper_setup_excludes_raw_data_and_old_mock():
    steps = setup_data.get_steps(setup_data.build_parser().parse_args(["--paper"]))
    assert {step["filename"] for step in steps if step["type"] == "download"} == {
        "light_curve_data.h5", "spectra_data.h5", "hubble_posteriors.zip",
        "Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov",
    }
    assert {step["name"] for step in steps if step["type"] == "command"} == {"fetch_csc"}


def test_checksum_failure_does_not_install_download(tmp_path, monkeypatch):
    downloaded = tmp_path / "downloaded"
    downloaded.write_bytes(b"incorrect data")
    monkeypatch.setattr(setup_data, "download_from_http", lambda *a, **k: (str(downloaded), "data.h5"))
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="MD5 mismatch"):
        setup_data.download_and_extract("https://example.org/data.h5", str(destination), "data.h5", md5="0" * 32)
    assert not (destination / "data.h5").exists()


def test_cached_archive_is_extracted_and_preserved(tmp_path):
    archive = tmp_path / "hubble_posteriors.zip"
    member = "hubble_posteriors/fiducial/FlatLambdaCDM_joint.h5"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"checkpoint")
    checksum = hashlib.md5(archive.read_bytes()).hexdigest()
    setup_data.download_and_extract("https://example.org/posteriors.zip", str(tmp_path), archive.name, md5=checksum)
    assert (tmp_path / member).read_bytes() == b"checkpoint"
    assert archive.exists()


def test_html_preview_is_rejected(monkeypatch):
    response = io.BytesIO(b"<html>Login</html>")
    headers = Message()
    headers["Content-Type"] = "text/html; charset=utf-8"
    response.info = lambda: headers
    monkeypatch.setattr(setup_data.urllib.request, "urlopen", lambda url: response)
    with pytest.raises(ValueError, match="HTML page"):
        setup_data.download_from_http("https://example.org/preview")


def write_csc_response(path, status="OK"):
    from astropy.table import Table
    import xml.etree.ElementTree as ET

    table = Table({name: [1.0] for name in setup_data.CSC_COLUMNS})
    table["name"] = ["2CXO test"]
    table.write(path, format="votable")
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = root.tag.split("}")[0] + "}"
    ET.SubElement(root, namespace + "INFO", name="QUERY_STATUS", value=status)
    tree.write(path)


def test_csc_download_installs_valid_catalog_and_reuses_it(tmp_path, monkeypatch):
    source = tmp_path / "response.vot"
    destination = tmp_path / "results/csc.vot"
    write_csc_response(source)
    monkeypatch.setattr(setup_data, "CSC_CATALOG", destination)
    calls = []

    def download(url, filename):
        calls.append(url)
        return str(source), filename

    monkeypatch.setattr(setup_data, "download_from_http", download)
    setup_data.fetch_csc()
    assert setup_data.validate_csc_catalog(destination) == 1
    assert destination.with_suffix(".adql").read_text().strip() == setup_data.CSC_QUERY
    setup_data.fetch_csc()
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["ERROR", "OVERFLOW"])
def test_csc_failed_or_truncated_response_is_not_installed(tmp_path, monkeypatch, status):
    source = tmp_path / "response.vot"
    destination = tmp_path / "results/csc.vot"
    write_csc_response(source, status)
    monkeypatch.setattr(setup_data, "CSC_CATALOG", destination)
    monkeypatch.setattr(setup_data, "download_from_http", lambda *a, **k: (str(source), "csc.vot"))
    with pytest.raises(ValueError, match="failed or was truncated"):
        setup_data.fetch_csc()
    assert not destination.exists()
    assert not source.exists()
