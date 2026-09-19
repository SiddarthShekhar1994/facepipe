"""Alignment: warp a face onto the ArcFace canonical template.

Every InsightFace recognition model was trained on crops in which the five
landmarks sit at fixed positions of a 112x112 image. Reproducing that at
inference is what makes embeddings comparable across pose and scale, so
this stage has to match the training-time alignment: a similarity
transform (rotation, uniform scale, translation) fitted by least squares
over all five points, then a warp. A full affine fit would shear the face.
"""

import cv2
import numpy as np

from facepipe.interfaces import Aligner
from facepipe.types import Frame

# ArcFace reference landmarks for a 112x112 crop, in the order the detector
# emits them: left eye, right eye, nose, left mouth, right mouth.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class ArcFaceAligner(Aligner):
    def __init__(self, crop_size: int):
        self._size = crop_size
        self._template = ARCFACE_TEMPLATE_112 * (crop_size / 112.0)

    def align(self, frame: Frame, landmarks: np.ndarray) -> np.ndarray:
        matrix = similarity_transform(landmarks, self._template)
        return cv2.warpAffine(frame, matrix, (self._size, self._size), borderValue=0)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform taking points `src` onto `dst` (Umeyama, 1991).

    Returns the 2x3 matrix M such that dst ~= M @ [x, y, 1]. Same solution
    as scikit-image's SimilarityTransform, which InsightFace uses.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1  # keep a rotation, not a reflection
    rotation = u @ np.diag(d) @ vt
    scale = (s @ d) / src_c.var(axis=0).sum()
    matrix = np.empty((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix
