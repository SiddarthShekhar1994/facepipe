"""The live loop: capture, show, time. `facepipe run`."""

from time import perf_counter

import cv2

from facepipe.config import Config
from facepipe.sources import WebcamSource
from facepipe.timing import StageTimer

WINDOW = "facepipe"


def run(cfg: Config, max_frames: int | None = None) -> None:
    timer = StageTimer(("read", "display"))
    with WebcamSource(cfg.source.device, cfg.source.width, cfg.source.height) as source:
        last_report = perf_counter()
        while max_frames is None or timer.frames < max_frames:
            with timer.stage("read"):
                frame = source.read()
            if frame is None:
                break
            with timer.stage("display"):
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
            timer.end_frame()
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if perf_counter() - last_report >= 1.0:
                print(timer.window())
                last_report = perf_counter()
    cv2.destroyAllWindows()
    print("summary:", timer.summary())
