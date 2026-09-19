"""Abstract stage interfaces.

Each stage is a class so a backend can be swapped without touching its
neighbours. On the laptop, Detector and Embedder run on ONNX Runtime; on the
target system they run on an FPGA DPU, and Matcher and Store move to a
server. Every boundary is plain numpy in, plain numpy out.
"""

from abc import ABC, abstractmethod

import numpy as np

from facepipe.types import Detection, Embedding, Frame, Identity, Match


class FrameSource(ABC):
    """Produces frames from a webcam, a video file, or a directory of images."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the device or file. Called once, before read()."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Next frame, or None once the stream is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release the device or file."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class Detector(ABC):
    """Model: frame in, faces out."""

    @abstractmethod
    def load(self) -> None:
        """Load weights and open the inference session. Called once, before infer()."""

    @abstractmethod
    def infer(self, frame: Frame) -> list[Detection]:
        """Every face in the frame above the detector's confidence threshold."""


class Aligner(ABC):
    """Warps a face onto a canonical, fixed-size crop using its five landmarks."""

    @abstractmethod
    def align(self, frame: Frame, landmarks: np.ndarray) -> np.ndarray:
        """Return an HxWx3 uint8 BGR crop of the aligner's fixed size."""


class Embedder(ABC):
    """Model: aligned crop in, L2-normalized embedding out."""

    @abstractmethod
    def load(self) -> None:
        """Load weights and open the inference session. Called once, before infer()."""

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Side of the square crop infer() expects; the Aligner is built to produce it. Valid after load()."""

    @abstractmethod
    def infer(self, crop: np.ndarray) -> Embedding:
        """Embedding of one aligned crop, L2-normalized."""


class Matcher(ABC):
    """Compares one query embedding against the enrolled gallery."""

    @abstractmethod
    def match(self, query: Embedding, gallery_names: list[str], gallery: np.ndarray) -> list[Match]:
        """Identities above the threshold, best first, at most top_k. Empty means unknown.

        `gallery` is (N, D) with one row per enrolled embedding and
        `gallery_names[i]` is the identity row i belongs to, so a name can
        repeat when a person was enrolled from several images.
        """


class Store(ABC):
    """Persists enrolled identities: their embeddings, reference images, metadata."""

    @abstractmethod
    def add(self, identity: Identity) -> None:
        """Persist a newly enrolled identity."""

    @abstractmethod
    def identities(self) -> list[Identity]:
        """Every enrolled identity."""
