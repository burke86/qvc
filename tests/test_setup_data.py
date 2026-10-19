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

    def download(url, filename, **kwargs):
        assert kwargs == {"timeout": 60}
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


@pytest.mark.parametrize('argv,step_list', [([], 'DEFAULT_STEPS'), (['--paper'], 'PAPER_STEPS')])
@pytest.mark.parametrize('failure', ['http500', 'connection', 'timeout', 'short_read'])
def test_csc_network_failure_continues_setup_and_retries(tmp_path, monkeypatch, capsys, argv, step_list, failure):
    import urllib.error
    import http.client

    destination = tmp_path / 'csc.vot'
    monkeypatch.setattr(setup_data, 'CSC_CATALOG', destination)
    steps = [{'type': 'command', 'name': 'fetch_csc'},
             {'type': 'download', 'url': 'https://example.org/required', 'folder': str(tmp_path)}]
    monkeypatch.setattr(setup_data, step_list, steps)
    errors = {
        'http500': lambda: urllib.error.HTTPError(setup_data.CSC_ENDPOINT, 500, 'Internal Server Error', {}, None),
        'connection': lambda: urllib.error.URLError('connection failed'),
        'timeout': lambda: TimeoutError('read timed out'),
        'short_read': lambda: http.client.IncompleteRead(b'partial', 100),
    }
    calls = []
    def fail(*args, **kwargs):
        calls.append('csc')
        raise errors[failure]()
    monkeypatch.setattr(setup_data, 'download_from_http', fail)
    monkeypatch.setattr(setup_data, 'download_and_extract', lambda *a, **k: calls.append('required'))
    setup_data.main(argv)
    setup_data.main(argv)
    assert calls == ['csc', 'required', 'csc', 'required']
    assert not destination.exists()
    output = capsys.readouterr().out
    assert setup_data.CSC_ENDPOINT in output
    assert 'Setup completed with CSC unavailable' in output
    assert 'All tasks completed successfully' not in output


def test_required_download_network_failure_remains_fatal(tmp_path, monkeypatch):
    import urllib.error
    monkeypatch.setattr(setup_data, 'PAPER_STEPS', [
        {'type': 'download', 'url': 'https://example.org/required', 'folder': str(tmp_path)},
    ])
    def fail(*args, **kwargs):
        raise urllib.error.URLError('required download failed')
    monkeypatch.setattr(setup_data, 'download_from_http', fail)
    with pytest.raises(urllib.error.URLError, match='required download failed'):
        setup_data.main(['--paper'])


@pytest.mark.parametrize('failure', ['timeout', 'short_read'])
def test_interrupted_http_download_removes_partial_file(tmp_path, monkeypatch, failure):
    import http.client
    headers = Message()
    headers['Content-Type'] = 'application/octet-stream'
    headers['Content-Length'] = '100'
    class Response(io.BytesIO):
        def read(self, size=-1):
            if self.tell() and failure == 'timeout':
                raise TimeoutError('interrupted body')
            return super().read(size)
        def info(self):
            return headers
    response = Response(b'partial')
    options = []
    def open_response(url, **kwargs):
        options.append(kwargs)
        return response
    monkeypatch.setattr(setup_data.urllib.request, 'urlopen', open_response)
    factory = setup_data.tempfile.NamedTemporaryFile
    monkeypatch.setattr(setup_data.tempfile, 'NamedTemporaryFile', lambda **kwargs: factory(dir=tmp_path, **kwargs))
    expected = TimeoutError if failure == 'timeout' else http.client.IncompleteRead
    with pytest.raises(expected):
        setup_data.download_from_http('https://example.org/csc', timeout=60)
    assert options == [{'timeout': 60}]
    assert list(tmp_path.iterdir()) == []


def test_setup_rejects_existing_malformed_csc(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    destination = tmp_path / 'csc.vot'
    destination.write_text('broken XML')
    monkeypatch.setattr(setup_data, 'CSC_CATALOG', destination)
    monkeypatch.setattr(setup_data, 'PAPER_STEPS', [{'type': 'command', 'name': 'fetch_csc'}])
    with pytest.raises(ET.ParseError):
        setup_data.main(['--paper'])
    assert destination.read_text() == 'broken XML'


def test_single_object_setup_downloads_and_reuses_bundle(tmp_path, monkeypatch):
    import shutil
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name in setup_data.LIGHT_CURVE_DEMO_FILES:
            bundle.writestr('data/' + name, name.encode())
    monkeypatch.setattr(setup_data, 'LIGHT_CURVE_DEMO_MD5', hashlib.md5(archive.read_bytes()).hexdigest())
    destination = tmp_path / 'installed'
    monkeypatch.setenv('QVC_DATA_DIR', str(destination))
    calls = []
    def download(url, output, **kwargs):
        calls.append(url)
        return shutil.copyfile(archive, output)
    monkeypatch.setattr(setup_data.gdown, 'download', download)
    setup_data.main(['--light-curve'])
    setup_data.main(['--light-curve'])
    assert calls == [setup_data.LIGHT_CURVE_DEMO_URL]
    for name in setup_data.LIGHT_CURVE_DEMO_FILES:
        assert (destination / name).read_bytes() == name.encode()


def test_single_object_setup_preserves_partial_existing_catalogs(tmp_path, monkeypatch):
    monkeypatch.setenv('QVC_DATA_DIR', str(tmp_path))
    existing = tmp_path / 'dr16q_prop_May01_2024.fits'
    existing.write_bytes(b'original full catalog')
    monkeypatch.setattr(setup_data.gdown, 'download', lambda *a, **k: pytest.fail('must not download'))
    with pytest.raises(FileExistsError, match='avoid mixing catalogs'):
        setup_data.main(['--light-curve'])
    assert existing.read_bytes() == b'original full catalog'


def test_single_object_setup_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv('QVC_DATA_DIR', str(tmp_path / 'installed'))
    def download(url, output, **kwargs):
        from pathlib import Path
        Path(output).write_bytes(b'bad archive')
        return output
    monkeypatch.setattr(setup_data.gdown, 'download', download)
    with pytest.raises(ValueError, match='MD5 mismatch'):
        setup_data.main(['--light-curve'])
    assert not (tmp_path / 'installed').exists()


def test_spectra_setup_selects_only_required_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv('QVC_DATA_DIR', str(tmp_path))
    steps = setup_data.get_steps(setup_data.build_parser().parse_args(['--spectra']))
    assert [s['name'] for s in steps if s['type'] == 'command'] == [
        'fetch_light_curve_demo', 'fetch_spectrum_demo', 'install_demo_sed_photometry', 'fetch_dustmaps']
    downloads = [s for s in steps if s['type'] == 'download']
    assert len(downloads) == 1
    assert downloads[0]['filename'] == 'ssp_data_continuum_fsps_v3.2_lgmet_age.h5'
    assert downloads[0]['folder'] == str(tmp_path)


@pytest.mark.parametrize("existing", ["none", "spectrum", "sed", "conflicting_spectrum"])
def test_spectrum_bundle_installs_exact_cache_and_reuses_it(tmp_path, monkeypatch, existing):
    import shutil
    archive = tmp_path / 'source.zip'
    member, sed = setup_data.SPECTRUM_DEMO_FILES
    contents = {member: b'single spectrum', sed: b'single object SED'}
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, content in contents.items():
            bundle.writestr('data/' + name, content)
    monkeypatch.setattr(setup_data, 'SPECTRUM_DEMO_MD5', hashlib.md5(archive.read_bytes()).hexdigest())
    destination = tmp_path / 'installed'
    monkeypatch.setenv('QVC_DATA_DIR', str(destination))
    calls = []
    def download(url, output, **kwargs):
        calls.append(url)
        return shutil.copyfile(archive, output)
    monkeypatch.setattr(setup_data.gdown, 'download', download)
    if existing != 'none':
        name = sed if existing == 'sed' else member
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'user data' if existing == 'conflicting_spectrum' else contents[name])
        before = target.stat().st_mtime_ns
    if existing == 'conflicting_spectrum':
        with pytest.raises(FileExistsError, match='refusing to overwrite'):
            setup_data.fetch_spectrum_demo()
        assert target.read_bytes() == b'user data'
        assert not (destination / sed).exists()
        return
    setup_data.fetch_spectrum_demo()
    setup_data.fetch_spectrum_demo()
    if existing != 'none':
        assert target.stat().st_mtime_ns == before
    assert calls == [setup_data.SPECTRUM_DEMO_URL]
    for name, content in contents.items():
        assert (destination / name).read_bytes() == content
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(contents)


def test_demo_sed_resource_contains_only_measured_target_rows(tmp_path, monkeypatch):
    import pandas as pd
    import json
    monkeypatch.setenv('QVC_DATA_DIR', str(tmp_path))
    setup_data.install_demo_sed_photometry()
    setup_data.install_demo_sed_photometry()
    path = tmp_path / 'demo_J233151.91+003258.3_sed.csv'
    frame = pd.read_csv(path)
    assert set(frame.object_id) == {1448770}
    assert len(frame) == 10
    assert frame.filter_name.nunique() == 10
    assert (frame.flux_err_mjy > 0).all()
    provenance = json.loads(path.with_suffix('.provenance.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['csv_sha256']
    path.write_text('existing user data')
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        setup_data.install_demo_sed_photometry()
