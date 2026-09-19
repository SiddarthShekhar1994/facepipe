"""FrameSource implementations."""

import cv2

from facepipe.interfaces import FrameSource
from facepipe.types import Frame


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
