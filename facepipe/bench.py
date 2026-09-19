"""Benchmark: a fixed input through the whole pipeline, per-stage latency and FPS. `facepipe bench`.

The webcam is refused on purpose: a benchmark has to be repeatable, and a
camera paces the loop and changes the picture every run.
"""

from dataclasses import replace

import numpy as np

from facepipe.config import Config
from facepipe.pipeline import STAGES, Pipeline
from facepipe.sources import make_source
from facepipe.timing import StageTimer

PIPELINE_STAGES = STAGES  # detect, align, embed, match: what "pipeline fps" counts
PER_FACE_STAGES = ("align", "embed", "match")


def bench(cfg: Config, video: str | None, images: str | None, frames: int, warmup: int, repeat: int, input_size: int | None) -> int:
    if (video is None) == (images is None):
        print("give exactly one of --video FILE or --images DIR")
        return 2
    source_cfg = replace(cfg.source, kind="video" if video else "images", path=video or images)
    if input_size is not None:
        cfg = replace(cfg, detector=replace(cfg.detector, input_size=input_size))

    runs = []
    for r in range(repeat):
        timer = StageTimer(("read", *PIPELINE_STAGES))
        pipeline = Pipeline(cfg, timer)
        pipeline.load()
        faces = 0
        seen = 0  # frames processed including warm-up; the timer restarts at the reset
        with make_source(source_cfg) as source:
            while seen < warmup + frames:
                with timer.stage("read"):
                    frame = source.read()
                if frame is None:
                    break
                results = pipeline.process(frame)
                timer.end_frame()
                seen += 1
                if seen == warmup:
                    timer.reset()
                elif seen > warmup:
                    faces += len(results)
        if seen <= warmup:
            print(f"input ended during warm-up ({warmup} frames); nothing measured")
            return 1
        runs.append(_measure(timer, faces))
        print(f"run {r + 1}/{repeat}: {timer.frames} frames, {faces / timer.frames:.2f} faces/frame, "
              f"detect median {runs[-1]['detect']['median']:.1f} ms, pipeline {runs[-1]['pipeline_fps']:.1f} fps")

    print()
    print(f"input: {source_cfg.path}, {runs[0]['frames']} frames after {warmup} warm-up, "
          f"detector input_size {cfg.detector.input_size}, {repeat} run(s)")
    print(f"{'stage':<10} {'median ms':>16} {'p95 ms':>16} {'mean ms/face':>16}")
    for stage in ("read", *PIPELINE_STAGES):
        per_face = _span(runs, stage, "per_face") if stage in PER_FACE_STAGES else ""
        print(f"{stage:<10} {_span(runs, stage, 'median'):>16} {_span(runs, stage, 'p95'):>16} {per_face:>16}")
    print(f"{'pipeline fps':<27} {_span(runs, 'pipeline_fps'):>16}   (detect + align + embed + match)")
    print(f"{'end-to-end fps':<27} {_span(runs, 'e2e_fps'):>16}   (including read)")
    print(f"{'faces/frame':<27} {_span(runs, 'faces_per_frame'):>16}")
    return 0


def _measure(timer: StageTimer, faces: int) -> dict:
    out = {"frames": timer.frames, "faces_per_frame": faces / timer.frames}
    pipeline_ms = 0.0
    for stage in ("read", *PIPELINE_STAGES):
        s = timer.samples(stage)
        out[stage] = {"median": float(np.median(s)), "p95": float(np.percentile(s, 95)), "mean": float(s.mean())}
        if stage in PER_FACE_STAGES:
            out[stage]["per_face"] = float(s.sum() / faces) if faces else float("nan")
        if stage in PIPELINE_STAGES:
            pipeline_ms += s.sum()
    out["pipeline_fps"] = timer.frames / (pipeline_ms / 1000.0)
    out["e2e_fps"] = timer.frames / timer.elapsed
    return out


def _span(runs: list[dict], key: str, sub: str | None = None) -> str:
    """One number, or "lo - hi" across repeats, formatted for the table."""
    values = [r[key][sub] if sub else r[key] for r in runs]
    lo, hi = min(values), max(values)
    return f"{lo:.1f}" if len(runs) == 1 or abs(hi - lo) < 0.05 else f"{lo:.1f} - {hi:.1f}"
