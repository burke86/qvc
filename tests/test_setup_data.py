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
