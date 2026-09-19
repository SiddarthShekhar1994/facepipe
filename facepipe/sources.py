"""FrameSource implementations and the factory that picks one from config."""

from pathlib import Path

import cv2

from facepipe.config import SourceConfig
from facepipe.interfaces import FrameSource
from facepipe.types import Frame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class WebcamSource(FrameSource):
    """Frames from a local camera via OpenCV's VideoCapture.

    `width`/`height` are a request; the driver picks the nearest mode it
    supports, so `open()` reports what was actually negotiated.
    """

    def __init__(self, device: int, width: int, height: int):
        self._device = device
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open webcam {self._device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        got = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"webcam {self._device}: {got[0]}x{got[1]} via {cap.getBackendName()}")
        self._cap = cap

    def read(self) -> Frame | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource(FrameSource):
    """Frames from a video file, decoded as fast as they are asked for; None at the end."""

    def __init__(self, path: str):
        self._path = path
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self._path}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"video {self._path}: {w}x{h}, {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
        self._cap = cap

    def read(self) -> Frame | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageDirectorySource(FrameSource):
    """Every image in a directory, in sorted name order; None after the last one."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._files: list[Path] = []
        self._next = 0

    def open(self) -> None:
        self._files = sorted(p for p in self._path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self._files:
            raise RuntimeError(f"no images in {self._path}")
        print(f"images {self._path}: {len(self._files)} files")
        self._next = 0

    def read(self) -> Frame | None:
        while self._next < len(self._files):
            path = self._files[self._next]
            self._next += 1
            frame = cv2.imread(str(path))
            if frame is not None:
                return frame
            print(f"  {path.name}: unreadable, skipped")
        return None

    def close(self) -> None:
        self._files = []


def make_source(cfg: SourceConfig) -> FrameSource:
    if cfg.kind == "webcam":
        return WebcamSource(cfg.device, cfg.width, cfg.height)
    if cfg.kind == "video":
        return VideoFileSource(cfg.path)
    if cfg.kind == "images":
        return ImageDirectorySource(cfg.path)
    raise ValueError(f"source.kind must be webcam, video or images, got {cfg.kind!r}")
