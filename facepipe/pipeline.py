"""One frame through every stage: detect, align, embed, match.

Built once from the config; used by the live loop and, without a window,
by the dashboard. This is the single-frame data flow the README walks
through, as one function.
"""

from pathlib import Path

import numpy as np

from facepipe.align import ArcFaceAligner
from facepipe.arcface import ArcFaceEmbedder
from facepipe.config import Config
from facepipe.matcher import CosineMatcher, build_gallery
from facepipe.scrfd import ScrfdDetector
from facepipe.store import DirectoryStore
from facepipe.timing import StageTimer
from facepipe.types import FaceResult, Frame, Identity

STAGES = ("detect", "align", "embed", "match")


class Pipeline:
    def __init__(self, cfg: Config, timer: StageTimer):
        self._timer = timer
        self.detector = ScrfdDetector(cfg.detector.model_path, cfg.detector.conf_threshold, cfg.detector.input_size)
        self.embedder = ArcFaceEmbedder(cfg.embedder.model_path)
        self.aligner: ArcFaceAligner | None = None  # sized from the embedder once it is loaded
        self.matcher = CosineMatcher(cfg.matcher.threshold, cfg.matcher.top_k)
        self.store = DirectoryStore(cfg.store.path, Path(cfg.embedder.model_path).name)
        self._gallery: tuple[list[str], np.ndarray] = ([], np.empty((0, 0), dtype=np.float32))

    def load(self) -> list[Identity]:
        """Load both models and the gallery. Returns the enrolled identities for reporting."""
        self.detector.load()
        self.embedder.load()
        self.aligner = ArcFaceAligner(self.embedder.input_size)
        return self.reload_gallery()

    def reload_gallery(self) -> list[Identity]:
        """Re-read the store. The gallery is one tuple swapped in by one assignment, so a
        frame being processed on another thread never sees new names with old rows."""
        identities = self.store.identities()
        self._gallery = build_gallery(identities)
        return identities

    def process(self, frame: Frame) -> list[FaceResult]:
        """Every face in the frame with its matches; an empty match list means unknown."""
        with self._timer.stage("detect"):
            detections = self.detector.infer(frame)
        with self._timer.stage("align"):
            crops = [self.aligner.align(frame, d.landmarks) for d in detections]
        with self._timer.stage("embed"):
            embeddings = [self.embedder.infer(c) for c in crops]
        names, gallery = self._gallery
        with self._timer.stage("match"):
            matches = [self.matcher.match(e, names, gallery) for e in embeddings]
        return [FaceResult(d, e, m) for d, e, m in zip(detections, embeddings, matches)]
