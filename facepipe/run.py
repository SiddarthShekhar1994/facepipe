"""The live loop: capture, run the pipeline, draw labels, show, time. `facepipe run`."""

from time import perf_counter

import cv2

from facepipe.config import Config
from facepipe.draw import draw_results
from facepipe.pipeline import STAGES, Pipeline
from facepipe.sources import WebcamSource
from facepipe.timing import StageTimer

WINDOW = "facepipe"


def run(cfg: Config, max_frames: int | None = None) -> None:
    timer = StageTimer(("read", *STAGES, "display"))
    pipeline = Pipeline(cfg, timer)
    t0 = perf_counter()
    identities = pipeline.load()
    rows = sum(i.embeddings.shape[0] for i in identities)
    print(f"models loaded in {(perf_counter() - t0) * 1000:.0f}ms")
    if identities:
        print(f"gallery: {len(identities)} people, {rows} images: {', '.join(i.name for i in identities)}")
    else:
        print(f"gallery: empty ({cfg.store.path}); every face will be unknown. Enroll with `facepipe enroll`.")

    with WebcamSource(cfg.source.device, cfg.source.width, cfg.source.height) as source:
        last_report = perf_counter()
        while max_frames is None or timer.frames < max_frames:
            with timer.stage("read"):
                frame = source.read()
            if frame is None:
                break
            results = pipeline.process(frame)
            with timer.stage("display"):
                draw_results(frame, results)
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

