"""Matching: one query embedding against the enrolled gallery."""

import numpy as np

from facepipe.interfaces import Matcher
from facepipe.types import Embedding, Identity, Match


def build_gallery(identities: list[Identity]) -> tuple[list[str], np.ndarray]:
    """Flatten the store into one (N, D) matrix with a name per row."""
    names = [i.name for i in identities for _ in range(i.embeddings.shape[0])]
    if not identities:
        return names, np.empty((0, 0), dtype=np.float32)
    return names, np.concatenate([i.embeddings for i in identities]).astype(np.float32)


class CosineMatcher(Matcher):
    """Cosine similarity, which is a dot product because embeddings are unit length.

    A person with several enrolled rows is scored by their best row, so a
    query in a turned pose matches the turned photo rather than an average
    that fits nothing well. The price is that one mislabelled enrolled image
    produces false matches for that person, which is why enroll refuses to
    guess on multi-face images.
    """

    def __init__(self, threshold: float, top_k: int):
        if top_k < 1:
            raise ValueError(f"matcher.top_k must be at least 1, got {top_k}")
        self._threshold = threshold
        self._top_k = top_k

    def match(self, query: Embedding, gallery_names: list[str], gallery: np.ndarray) -> list[Match]:
        if len(gallery_names) == 0:
            return []
        similarities = gallery @ query
        best: dict[str, float] = {}
        for name, similarity in zip(gallery_names, similarities.tolist()):
            if similarity > best.get(name, -2.0):
                best[name] = similarity
        matches = [Match(name, s) for name, s in best.items() if s >= self._threshold]
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[: self._top_k]
