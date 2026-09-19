"""Data types that cross stage boundaries.

Everything here is a numpy array or a frozen dataclass of numpy arrays and
scalars, so the core pipeline stays free of framework imports and every
stage can be replaced by one that speaks plain arrays.
"""

from dataclasses import dataclass

import numpy as np

# HxWx3 uint8 BGR image, exactly as OpenCV delivers it. A model that wants
# RGB or float input converts inside its own wrapper; the source stays dumb.
Frame = np.ndarray

# (D,) float32, L2-normalized. The Embedder guarantees the normalization so
# the Matcher can treat a dot product as cosine similarity.
Embedding = np.ndarray


@dataclass(frozen=True)
class Detection:
    """One face found in a frame."""

    bbox: np.ndarray  # (4,) float32, [x1, y1, x2, y2] in pixels
    landmarks: np.ndarray  # (5, 2) float32 pixels: left eye, right eye, nose, left mouth, right mouth
    confidence: float


@dataclass(frozen=True)
class Match:
    """One enrolled identity that cleared the similarity threshold."""

    name: str
    similarity: float


@dataclass(frozen=True)
class Identity:
    """An enrolled person. `name` is the key; there is no separate id."""

    name: str
    embeddings: np.ndarray  # (K, D) float32, one row per enrolled image
    image_paths: list[str]  # the reference images those rows came from


@dataclass(frozen=True)
class FaceResult:
    """One face in a frame with its gallery matches, best first. No matches means unknown."""

    detection: Detection
    matches: list[Match]
