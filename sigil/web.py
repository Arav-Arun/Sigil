"""A local web interface for Sigil.

Deliberately **localhost only**. Hosting a face-search engine publicly means anyone can
upload anyone's face, which is exactly the use this project's responsible-use statement
rules out. Binding to 127.0.0.1 keeps the visual workflow while leaving the tool in the
hands of the person who installed it.

Built on the standard library so the UI adds no dependency to the pipeline: the parts
that matter for the submission are the engine, the evidence, and the chain, and a web
framework in the dependency tree for a convenience layer would be the wrong trade.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from sigil import __version__
from sigil.config import DATA_DIR, ROOT_DIR, SAMPLES_DIR, Settings, get_settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
BUNDLES_DIR = DATA_DIR / "bundles"
UPLOAD_DIR = DATA_DIR / "outputs" / "uploads"
BENCHMARK_PATH = ROOT_DIR / "docs" / "benchmark.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass
class RunState:
    """In-memory state for one submitted run, polled by the browser."""

    run_id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    result: dict[str, Any] | None = None
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "stage": self.stage,
            "started_at": self.started_at.isoformat(),
            "result": self.result,
            "error": self.error,
        }


RUNS: dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()

STAGE_LABELS = {
    "warm_up": "Loading models",
    "face": "Detecting and encoding the face",
    "search": "Searching the live web",
    "acquire": "Fetching candidate media",
    "verify": "Verifying identity",
    "expand": "Looking for more posts of the same person",
    "evidence": "Building the evidence root",
    "report": "Rendering the report",
    "anchor": "Anchoring on-chain",
}


def _header_value(raw: str) -> str:
    """First line of a header parameter, unquoted."""

    return raw.split("\r\n")[0].strip().strip('"')


def _parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str, bytes]]:
    """Minimal multipart/form-data parser for a single small upload."""

    marker = "boundary="
    if marker not in content_type:
        return {}
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    delimiter = f"--{boundary}".encode()

    fields: dict[str, tuple[str, bytes]] = {}
    for part in body.split(delimiter):
        if not part.strip(b"-\r\n"):
            continue
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue
        headers = header_blob.decode("utf-8", "replace")
        name = filename = ""
        for token in headers.split(";"):
            token = token.strip()
            # Splitting on ";" leaves the last parameter glued to the header line that
            # follows it, so `filename="face.jpg"\r\nContent-Type: image/jpeg`. The line
            # break has to be cut *before* the quotes are stripped, or the closing quote
            # survives and the extension check rejects every real upload.
            if token.startswith("name="):
                name = _header_value(token[5:])
            elif token.startswith("filename="):
                filename = _header_value(token[9:])
        if name:
            fields[name] = (filename, payload.rstrip(b"\r\n"))
    return fields


def _start_run(
    image_path: Path,
    settings: Settings,
    *,
    skip_chain: bool,
    no_cache: bool,
    largest: bool,
    name_hint: str = "",
) -> str:
    """Kick off a pipeline run on a worker thread and return its id."""

    from sigil.imaging import sha256_file

    started_at = datetime.now(UTC)
    run_id = f"{started_at:%Y%m%dT%H%M%S%fZ}-{sha256_file(image_path)[:8]}"
    state = RunState(run_id=run_id, started_at=started_at)
    with RUNS_LOCK:
        RUNS[run_id] = state

    def worker() -> None:
        from sigil.run import run_pipeline

        def on_stage(name: str) -> None:
            state.stage = STAGE_LABELS.get(name, name)

        state.status = "running"
        try:
            result = run_pipeline(
                image_path,
                settings=settings,
                face_index=None,
                select_largest=largest,
                no_cache=no_cache,
                use_cache=not no_cache,
                skip_chain=skip_chain,
                name_hint=name_hint,
                run_id=run_id,
                started_at=started_at,
                on_stage=on_stage,
            )
            state.result = result.to_json()
            state.status = "done"
            state.stage = ""
        except Exception as exc:  # a UI must not die on a pipeline error
            logger.exception("run %s failed", run_id)
            state.status = "error"
            state.error = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=worker, daemon=True).start()
    return run_id


class SigilHandler(BaseHTTPRequestHandler):
    server_version = f"Sigil/{__version__}"
    settings: Settings

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------------

    def _run_sample(self, route: Any) -> None:
        """Start a run on an image already in data/samples/, with no upload."""

        name = parse_qs(route.query).get("file", [""])[0]
        target = self._safe_join(SAMPLES_DIR, name) if name else None
        if not target or not target.is_file():
            self._json(404, {"error": f"no such sample: {name}"})
            return

        def flag(key: str) -> bool:
            return parse_qs(route.query).get(key, ["0"])[0] in ("1", "true")

        def text(key: str) -> str:
            value: str = parse_qs(route.query).get(key, [""])[0]
            return value.strip()[:80]

        try:
            run_id = _start_run(
                target,
                self.settings,
                skip_chain=flag("skip_chain"),
                no_cache=flag("no_cache"),
                largest=flag("largest"),
                name_hint=text("name"),
            )
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(202, {"run_id": run_id, "upload": f"/samples/{target.name}"})

    def _send(self, code: int, body: bytes, content_type: str, *, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        # This server is localhost-only and never embedded; lock it down anyway.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data: https:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload, default=str).encode(), "application/json")

    def _file(self, path: Path, *, cache: bool = False) -> None:
        if not path.is_file():
            self._json(404, {"error": "not found"})
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), mime, cache=cache)

    @staticmethod
    def _safe_join(base: Path, *parts: str) -> Path | None:
        """Resolve a path and refuse anything that escapes the base directory."""

        candidate = base.joinpath(*parts).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            return None
        return candidate

    def _same_origin_post(self) -> bool:
        """Reject browser requests made by an unrelated web page.

        The service binds to localhost, but a public page can still submit a form to a
        localhost endpoint. That must not be enough to start a paid search or write a
        chain transaction. Non-browser clients commonly omit both headers and remain
        usable for local automation.
        """

        if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
            "0.0.0.0",
        }:
            return False
        try:
            origin_port = parsed.port or 80
        except ValueError:
            return False
        server_addr = cast(tuple[Any, ...], self.server.server_address)
        server_port = int(server_addr[1])
        return origin_port == server_port

    # -- routes -------------------------------------------------------------------

    def do_GET(self) -> None:
        route = urlparse(self.path)
        path = route.path

        if path in ("/", "/index.html"):
            self._file(STATIC_DIR / "app.html")
            return
        if path == "/verify":
            self._file(Path(__file__).resolve().parent.parent / "verify.html")
            return
        if path == "/api/config":
            summary = self.settings.redacted_summary()
            self._json(200, {"version": __version__, **summary})
            return
        if path == "/api/samples":
            # Any image dropped into data/samples/ becomes a clickable tile, so adding a
            # demo case needs no code change.
            samples = []
            if SAMPLES_DIR.is_dir():
                for f in sorted(SAMPLES_DIR.iterdir()):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        samples.append(
                            {
                                "name": f.stem.replace("_", " ").replace("-", " "),
                                "file": f.name,
                                "url": f"/samples/{f.name}",
                            }
                        )
            self._json(200, {"samples": samples})
            return
        if path.startswith("/samples/"):
            target = self._safe_join(SAMPLES_DIR, path.rsplit("/", 1)[-1])
            self._file(target, cache=True) if target else self._json(400, {"error": "bad path"})
            return
        if path == "/api/benchmark":
            # The landing strip shows measured numbers, read from the committed report
            # rather than typed into the page, so the two cannot drift apart.
            if BENCHMARK_PATH.is_file():
                self._file(BENCHMARK_PATH)
            else:
                self._json(200, {})
            return
        if path == "/api/runs":
            with RUNS_LOCK:
                self._json(200, {"runs": [s.to_json() for s in RUNS.values()]})
            return
        if path.startswith("/api/run/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                state = RUNS.get(run_id)
            self._json(200, state.to_json()) if state else self._json(404, {"error": "unknown run"})
            return
        if path.startswith("/media/"):
            parts = [p for p in path[len("/media/") :].split("/") if p and p != ".."]
            target = self._safe_join(BUNDLES_DIR, *parts)
            self._file(target, cache=True) if target else self._json(400, {"error": "bad path"})
            return
        if path.startswith("/uploads/"):
            name = path.rsplit("/", 1)[-1]
            target = self._safe_join(UPLOAD_DIR, name)
            self._file(target, cache=True) if target else self._json(400, {"error": "bad path"})
            return

        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._same_origin_post():
            self._json(403, {"error": "cross-origin requests are not allowed"})
            return
        route = urlparse(self.path)
        if route.path == "/api/run-sample":
            self._run_sample(route)
            return
        if route.path != "/api/run":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._json(413, {"error": f"upload must be 1..{MAX_UPLOAD_BYTES} bytes"})
            return

        body = self.rfile.read(length)
        fields = _parse_multipart(body, self.headers.get("Content-Type", ""))
        if "image" not in fields:
            self._json(400, {"error": "no image field in the request"})
            return

        filename, payload = fields["image"]
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            self._json(400, {"error": f"unsupported image type {suffix}"})
            return

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        upload_path = UPLOAD_DIR / f"{stamp}{suffix}"
        upload_path.write_bytes(payload)

        query = parse_qs(urlparse(self.path).query)

        def flag(name: str) -> bool:
            return query.get(name, ["0"])[0] in ("1", "true")

        def text(name: str) -> str:
            value: str = query.get(name, [""])[0]
            return value.strip()[:80]

        try:
            run_id = _start_run(
                upload_path,
                self.settings,
                skip_chain=flag("skip_chain"),
                no_cache=flag("no_cache"),
                largest=flag("largest"),
                name_hint=text("name"),
            )
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self._json(202, {"run_id": run_id, "upload": f"/uploads/{upload_path.name}"})


def serve(
    host: str = "127.0.0.1",
    port: int = 8420,
    *,
    settings: Settings | None = None,
    open_browser: bool = True,
) -> None:
    """Run the local UI until interrupted."""

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"refusing to bind to {host}: Sigil's web UI is localhost-only by design. "
            "Publishing a face-search interface would let anyone submit anyone's face."
        )

    SigilHandler.settings = settings or get_settings()
    server = ThreadingHTTPServer((host, port), SigilHandler)
    url = f"http://{host}:{port}/"
    print(f"Sigil UI on {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


__all__ = ["RunState", "SigilHandler", "serve"]
