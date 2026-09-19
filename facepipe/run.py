"""The live loop: capture, detect, draw, show, time. `facepipe run`."""

from time import perf_counter

import cv2

from facepipe.config import Config
from facepipe.scrfd import ScrfdDetector
from facepipe.sources import WebcamSource
from facepipe.timing import StageTimer
from facepipe.types import Detection, Frame

WINDOW = "facepipe"
BOX = (0, 200, 0)
DOT = (0, 0, 255)


def run(cfg: Config, max_frames: int | None = None) -> None:
    detector = ScrfdDetector(cfg.detector.model_path, cfg.detector.conf_threshold, cfg.detector.input_size)
    t0 = perf_counter()
    detector.load()
    print(f"detector: {cfg.detector.model_path} loaded in {(perf_counter() - t0) * 1000:.0f}ms")

    timer = StageTimer(("read", "detect", "display"))
    with WebcamSource(cfg.source.device, cfg.source.width, cfg.source.height) as source:
        last_report = perf_counter()
        while max_frames is None or timer.frames < max_frames:
            with timer.stage("read"):
                frame = source.read()
            if frame is None:
                break
            with timer.stage("detect"):
                detections = detector.infer(frame)
            with timer.stage("display"):
                draw_detections(frame, detections)
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


def draw_detections(frame: Frame, detections: list[Detection]) -> None:
    """Box, confidence, and the five landmarks, drawn in place."""
    for d in detections:
        x1, y1, x2, y2 = d.bbox.round().astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX, 2)
        cv2.putText(frame, f"{d.confidence:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX, 1, cv2.LINE_AA)
        for x, y in d.landmarks.round().astype(int).tolist():
            cv2.circle(frame, (x, y), 2, DOT, -1)
