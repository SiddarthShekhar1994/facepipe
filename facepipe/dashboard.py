"""The dashboard: live feed with labels, the enrolled list, enroll from the webcam. `facepipe serve`.

The only module that imports an HTTP server. One worker thread owns the
camera and both models and publishes the latest frame; request threads
only ever read it. Enrolling the face on screen reuses the embedding the
pipeline already computed for it, so no model runs off the worker thread.
"""

import html
import tempfile
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, quote, urlparse

import cv2

from facepipe.config import Config
from facepipe.draw import draw_results
from facepipe.pipeline import STAGES, Pipeline
from facepipe.sources import make_source
from facepipe.timing import StageTimer
from facepipe.types import FaceResult, Frame, Identity


@dataclass(frozen=True)
class Snapshot:
    frame: Frame  # raw, undrawn: what enrollment stores
    results: list[FaceResult]
    jpeg: bytes  # drawn and encoded: what the stream sends
    seq: int


class Worker(threading.Thread):
    """Capture -> pipeline -> draw -> encode, forever; publishes the latest Snapshot."""

    def __init__(self, cfg: Config):
        super().__init__(daemon=True)
        self._cfg = cfg
        self.timer = StageTimer(("read", *STAGES, "encode"))
        self.pipeline = Pipeline(cfg, self.timer)
        self._cond = threading.Condition()
        self._snapshot: Snapshot | None = None
        self._stop = threading.Event()
        self._enroll_lock = threading.Lock()

    def run(self) -> None:
        with make_source(self._cfg.source) as source:
            last_report = perf_counter()
            seq = 0
            while not self._stop.is_set():
                with self.timer.stage("read"):
                    frame = source.read()
                if frame is None:
                    break
                results = self.pipeline.process(frame)
                with self.timer.stage("encode"):
                    drawn = frame.copy()
                    draw_results(drawn, results)
                    _, jpeg = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.timer.end_frame()
                seq += 1
                with self._cond:
                    self._snapshot = Snapshot(frame, results, jpeg.tobytes(), seq)
                    self._cond.notify_all()
                if perf_counter() - last_report >= 1.0:
                    print(self.timer.window())
                    last_report = perf_counter()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> Snapshot | None:
        with self._cond:
            return self._snapshot

    def wait_after(self, seq: int, timeout: float) -> Snapshot | None:
        """The first snapshot newer than `seq`, or None on timeout."""
        with self._cond:
            if self._cond.wait_for(lambda: self._snapshot is not None and self._snapshot.seq > seq, timeout):
                return self._snapshot
            return None

    def enroll_current(self, name: str) -> str:
        """Enroll the one face on screen under `name`. Returns a message for the page."""
        snap = self.latest()
        if snap is None:
            return "no frame from the camera yet"
        if len(snap.results) == 0:
            return "no face in view"
        if len(snap.results) > 1:
            return f"{len(snap.results)} faces in view; enroll one person at a time"
        face = snap.results[0]
        with self._enroll_lock, tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "webcam.jpg"
            cv2.imwrite(str(path), snap.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            try:
                self.pipeline.store.add(Identity(name, face.embedding[None, :], [str(path)]))
            except ValueError as e:
                return str(e)
            identities = self.pipeline.reload_gallery()
        count = next(i.embeddings.shape[0] for i in identities if i.name == name)
        return f"enrolled {name}: {count} image(s) on file"


class Handler(BaseHTTPRequestHandler):
    worker: Worker  # set by serve()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._page()
        elif path == "/stream":
            self._stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/enroll":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        name = form.get("name", [""])[0].strip()
        message = self.worker.enroll_current(name) if name else "a name is required"
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/?msg=" + quote(message))
        self.end_headers()

    def _page(self) -> None:
        message = parse_qs(urlparse(self.path).query).get("msg", [""])[0]
        identities = self.worker.pipeline.store.identities()
        rows = "".join(
            f"<tr><td>{html.escape(i.name)}</td><td>{i.embeddings.shape[0]}</td></tr>" for i in identities
        ) or "<tr><td colspan=2>nobody enrolled yet</td></tr>"
        body = PAGE.replace("{message}", html.escape(message)).replace("{rows}", rows)
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self) -> None:
        """MJPEG: one multipart response that never ends, a JPEG part per new frame."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = 0
        try:
            while self.worker.is_alive():
                snap = self.worker.wait_after(seq, timeout=1.0)
                if snap is None:
                    continue
                seq = snap.seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(snap.jpeg))
                self.wfile.write(snap.jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser navigated away

    def log_message(self, format: str, *args) -> None:
        pass  # the console is for the timing line


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>facepipe</title>
<style>body { font-family: sans-serif; margin: 2rem; } td, th { padding: 0.2rem 1rem 0.2rem 0; text-align: left; }</style>
</head>
<body>
<h1>facepipe</h1>
<img src="/stream" alt="live feed" width="640" height="480">
<h2>Enroll the face on screen</h2>
<form method="post" action="/enroll">
  <input name="name" placeholder="name (letters, digits, _ -)" required>
  <button type="submit">Enroll</button>
</form>
<p><b>{message}</b></p>
<h2>Enrolled</h2>
<table>
  <tr><th>Name</th><th>Images</th></tr>
  {rows}
</table>
</body>
</html>
"""


def serve(cfg: Config, host: str, port: int) -> None:
    worker = Worker(cfg)
    t0 = perf_counter()
    identities = worker.pipeline.load()  # on this thread, so a bad model path fails before the server starts
    print(f"models loaded in {(perf_counter() - t0) * 1000:.0f}ms; gallery: {len(identities)} people")
    worker.start()
    Handler.worker = worker
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"dashboard: http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        server.server_close()
        print("summary:", worker.timer.summary())
