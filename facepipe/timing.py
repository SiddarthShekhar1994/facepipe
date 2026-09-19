"""Per-stage wall-clock timing for the frame loop."""

from contextlib import contextmanager
from time import perf_counter

import numpy as np


class StageTimer:
    """Records how long each named stage takes per frame.

    `stage(name)` wraps one stage of one frame; `end_frame()` marks the frame
    done. `window()` reports the frames since the previous call, so the loop
    can print a rolling line once a second; `summary()` covers the whole run.
    """

    def __init__(self, stages: tuple[str, ...]):
        self._stages = stages
        self._samples: dict[str, list[float]] = {s: [] for s in stages}
        self._frame_ends: list[float] = [perf_counter()]
        self._window_start = 0  # index into _frame_ends of the last reported frame

    @contextmanager
    def stage(self, name: str):
        t0 = perf_counter()
        yield
        self._samples[name].append(perf_counter() - t0)

    def end_frame(self) -> None:
        self._frame_ends.append(perf_counter())

    @property
    def frames(self) -> int:
        return len(self._frame_ends) - 1

    def window(self) -> str:
        start, end = self._window_start, self.frames
        self._window_start = end
        return self._line(start, end)

    def summary(self) -> str:
        return self._line(0, self.frames, percentiles=True)

    def _line(self, start: int, end: int, percentiles: bool = False) -> str:
        n = end - start
        if n == 0:
            return "no frames"
        elapsed = self._frame_ends[end] - self._frame_ends[start]
        parts = []
        for name in self._stages:
            s = np.asarray(self._samples[name][start:end]) * 1000.0
            if s.size == 0:
                continue
            cell = f"{name} {s.mean():5.1f}ms"
            if percentiles:
                cell += f" (p95 {np.percentile(s, 95):5.1f})"
            parts.append(cell)
        parts.append(f"{n / elapsed:4.1f} fps over {n} frames")
        return "  ".join(parts)
