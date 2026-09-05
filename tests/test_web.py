"""Tests for the local web interface.

The UI is a convenience layer, but two of its properties are not: it must refuse to bind
anywhere except loopback, and it must never serve a file from outside the directories it
owns. Both are one-line mistakes away from being wrong, and neither fails loudly, so they
are tested here rather than trusted.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from sigil.config import Settings
from sigil.web import SigilHandler, _parse_multipart, serve


def _settings() -> Settings:
    return Settings(_env_file=None, serpapi_key="k", contract_address="0x" + "ab" * 20)


class TestLocalhostOnly:
    """Publishing a face-search interface would let anyone submit anyone's face."""

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "::"])
    def test_refuses_every_non_loopback_address(self, host):
        with pytest.raises(ValueError, match="localhost-only"):
            serve(host, 0, settings=_settings(), open_browser=False)

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_accepts_loopback(self, host, monkeypatch):
        # Stop at the bind so the test does not start a server it has to tear down.
        started: list[str] = []

        def fake_server(address, handler):
            started.append(address[0])
            raise KeyboardInterrupt

        monkeypatch.setattr("sigil.web.ThreadingHTTPServer", fake_server)
        with pytest.raises(KeyboardInterrupt):
            serve(host, 0, settings=_settings(), open_browser=False)
        assert started == [host]


class TestSafeJoin:
    def test_accepts_a_path_inside_the_base(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        assert SigilHandler._safe_join(tmp_path, "a.jpg") == (tmp_path / "a.jpg").resolve()

    @pytest.mark.parametrize(
        "parts",
        [
            ("..",),
            ("..", "..", "etc", "passwd"),
            ("nested", "..", "..", "outside.txt"),
            ("/etc/passwd",),
        ],
    )
    def test_refuses_anything_that_escapes_the_base(self, tmp_path, parts):
        assert SigilHandler._safe_join(tmp_path / "base", *parts) is None


class TestMultipart:
    def _body(self, boundary: str, filename: str, payload: bytes) -> bytes:
        return (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n"
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )

    def test_extracts_the_filename_and_the_bytes(self):
        fields = _parse_multipart(
            self._body("xyz", "face.jpg", b"\xff\xd8body"),
            "multipart/form-data; boundary=xyz",
        )
        assert fields["image"] == ("face.jpg", b"\xff\xd8body")

    def test_returns_nothing_without_a_boundary(self):
        assert _parse_multipart(b"anything", "application/json") == {}

    def test_tolerates_a_quoted_boundary(self):
        fields = _parse_multipart(
            self._body("xyz", "f.png", b"data"), 'multipart/form-data; boundary="xyz"'
        )
        assert fields["image"][0] == "f.png"


@pytest.fixture
def server(tmp_path, monkeypatch, jpeg_bytes):
    """A live handler on an ephemeral port, with the data directories inside tmp_path."""

    monkeypatch.chdir(tmp_path)
    samples = tmp_path / "data" / "samples"
    samples.mkdir(parents=True)
    (samples / "public_figure.jpg").write_bytes(jpeg_bytes)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "benchmark.json").write_text(json.dumps({"roc_auc": 0.99555}))

    # These are module-level relative paths, resolved against the working directory.
    monkeypatch.setattr("sigil.web.SAMPLES_DIR", samples)
    monkeypatch.setattr("sigil.web.BUNDLES_DIR", tmp_path / "data" / "bundles")
    monkeypatch.setattr("sigil.web.UPLOAD_DIR", tmp_path / "data" / "outputs" / "uploads")
    monkeypatch.setattr("sigil.web.BENCHMARK_PATH", tmp_path / "docs" / "benchmark.json")

    SigilHandler.settings = _settings()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), SigilHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


class TestRoutes:
    def test_serves_the_application_shell(self, server):
        status, body, mime = _get(f"{server}/")
        assert status == 200
        assert b"<title>" in body
        assert mime == "text/html"

    def test_config_reports_presence_and_never_a_credential(self, server):
        status, body, _ = _get(f"{server}/api/config")
        payload = json.loads(body)
        assert status == 200
        assert payload["serpapi_configured"] is True
        assert "serpapi_key" not in body.decode()
        assert "k" not in payload.values()

    def test_lists_samples_from_the_samples_directory(self, server):
        payload = json.loads(_get(f"{server}/api/samples")[1])
        assert payload["samples"] == [
            {
                "name": "public figure",
                "file": "public_figure.jpg",
                "url": "/samples/public_figure.jpg",
            }
        ]

    def test_serves_a_sample_image(self, server, jpeg_bytes):
        status, body, mime = _get(f"{server}/samples/public_figure.jpg")
        assert (status, body, mime) == (200, jpeg_bytes, "image/jpeg")

    def test_benchmark_strip_is_read_from_the_committed_report(self, server):
        # The landing page must not be able to state a number the benchmark never produced.
        payload = json.loads(_get(f"{server}/api/benchmark")[1])
        assert payload["roc_auc"] == 0.99555

    def test_unknown_paths_are_404_json(self, server):
        status, body, mime = _get(f"{server}/nope")
        assert status == 404
        assert json.loads(body)["error"] == "not found"
        assert mime == "application/json"

    def test_unknown_run_id_is_404(self, server):
        assert _get(f"{server}/api/run/does-not-exist")[0] == 404

    def test_runs_listing_starts_empty_or_valid(self, server):
        payload = json.loads(_get(f"{server}/api/runs")[1])
        assert isinstance(payload["runs"], list)

    def test_sets_hardening_headers(self, server):
        with urllib.request.urlopen(f"{server}/api/config", timeout=10) as response:
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]

    @pytest.mark.parametrize(
        "path",
        [
            "/media/../../../etc/passwd",
            "/samples/../../../etc/passwd",
            "/uploads/../../../etc/passwd",
        ],
    )
    def test_refuses_directory_traversal(self, server, path):
        status, body, _ = _get(server + path)
        assert status in (400, 404)
        assert b"root:" not in body

    def test_missing_media_is_404_not_a_traceback(self, server):
        status, body, _ = _get(f"{server}/media/nope/media/thumbs/1.jpg")
        assert status == 404
        assert json.loads(body)["error"] == "not found"

    def test_run_sample_rejects_an_unknown_file(self, server):
        request = urllib.request.Request(f"{server}/api/run-sample?file=absent.jpg", method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    def test_cross_origin_page_cannot_start_a_run(self, server):
        request = urllib.request.Request(
            f"{server}/api/run-sample?file=public_figure.jpg",
            method="POST",
            headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        assert caught.value.code == 403
        assert json.loads(caught.value.read())["error"] == "cross-origin requests are not allowed"

    def test_run_sample_refuses_a_path_outside_the_samples_directory(self, server, tmp_path):
        (tmp_path / "secret.jpg").write_bytes(b"not a sample")
        request = urllib.request.Request(
            f"{server}/api/run-sample?file=../../secret.jpg", method="POST"
        )
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    def test_upload_rejects_an_empty_body(self, server):
        request = urllib.request.Request(
            f"{server}/api/run", data=b"", method="POST", headers={"Content-Length": "0"}
        )
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 413")
        except urllib.error.HTTPError as exc:
            assert exc.code == 413

    def test_upload_rejects_a_body_without_an_image_field(self, server):
        body = b'--b\r\nContent-Disposition: form-data; name="other"\r\n\r\nx\r\n--b--\r\n'
        request = urllib.request.Request(
            f"{server}/api/run",
            data=body,
            method="POST",
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as exc:
            assert json.loads(exc.read())["error"] == "no image field in the request"

    def test_upload_rejects_an_unsupported_extension(self, server):
        body = (
            b'--b\r\nContent-Disposition: form-data; name="image"; filename="payload.svg"'
            b"\r\n\r\n<svg/>\r\n--b--\r\n"
        )
        request = urllib.request.Request(
            f"{server}/api/run",
            data=body,
            method="POST",
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as exc:
            assert "unsupported image type" in json.loads(exc.read())["error"]

    def test_post_to_an_unknown_path_is_404(self, server):
        request = urllib.request.Request(f"{server}/api/nope", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


class TestVerifierPage:
    def test_serves_the_standalone_verifier(self, server):
        status, body, mime = _get(f"{server}/verify")
        assert status == 200
        assert mime == "text/html"
        assert b"VERIFY_SELECTOR" in body

    def test_csp_allows_the_verifiers_default_sepolia_rpc(self, server):
        with urllib.request.urlopen(f"{server}/verify", timeout=10) as response:
            policy = response.headers["Content-Security-Policy"]
        assert "https://ethereum-sepolia-rpc.publicnode.com" in policy


class TestRunState:
    def test_serializes_without_leaking_internals(self):
        from sigil.web import RunState

        payload = RunState(run_id="r1").to_json()
        assert set(payload) == {
            "run_id",
            "status",
            "stage",
            "started_at",
            "upload",
            "result",
            "error",
        }


def test_static_assets_exist_on_disk():
    """A missing shell would only show up as a blank page during the demo."""

    static = Path(__file__).resolve().parent.parent / "sigil" / "static"
    assert (static / "app.html").is_file()


def test_results_ui_separates_distinct_photos_from_source_image_reposts():
    """A provider hint must not become a repeated exact-photo card badge."""

    source = (Path(__file__).resolve().parent.parent / "sigil" / "static" / "app.html").read_text(
        encoding="utf-8"
    )

    assert "Only the submitted image was rediscovered" in source
    assert 'Show ${plural(reposts.length,"source-image repost")}' in source
    assert "EXACT PHOTO" not in source
    # The credit is one sentence split across two links, so assert the rendered wording
    # and both destinations rather than a single string that markup can break.
    assert "Made by Team" in source
    assert ">Deploy For Good</a>" in source
    assert "https://aravarun.in" in source
    assert ">Hacker House Goa</a>" in source
    assert "https://hhgoa.com" in source
    assert "Task 3</footer>" in source
