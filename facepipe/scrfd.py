"""SCRFD face detector on ONNX Runtime.

SCRFD (Guo et al., 2021, InsightFace) is a single-shot detector with three
output strides. For every anchor at every stride the network emits a face
score, a box as four distances (left, top, right, bottom) from the anchor
centre, and five landmarks as offsets from that centre. The network is the
.onnx file; this module owns the letterboxing in front of it and the decode
and NMS behind it.
"""

import cv2
import numpy as np
import onnxruntime as ort

from facepipe.interfaces import Detector
from facepipe.types import Detection, Frame


class ScrfdDetector(Detector):
    STRIDES = (8, 16, 32)
    ANCHORS_PER_CELL = 2
    NMS_IOU = 0.4  # the model family's standard value; nobody tunes it, so it is not in config

    def __init__(self, model_path: str, conf_threshold: float, input_size: int):
        if input_size % max(self.STRIDES):
            raise ValueError(f"detector.input_size must be a multiple of {max(self.STRIDES)}, got {input_size}")
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._input_size = input_size
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._centers: list[np.ndarray] = []

    def load(self) -> None:
        opts = ort.SessionOptions()
        # The file's output-shape metadata was recorded at 640x640. At any
        # other input_size ORT logs a shape-mismatch warning for all nine
        # outputs on every run; the values are correct. Errors still print.
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(self._model_path, opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._centers = [self._anchor_centers(stride) for stride in self.STRIDES]

    def infer(self, frame: Frame) -> list[Detection]:
        blob, scale = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._decode(outputs, scale)

    def _anchor_centers(self, stride: int) -> np.ndarray:
        """(cells * anchors, 2) pixel centre of every anchor at this stride.

        Flattened row-major over (y, x) with the anchors of one cell adjacent,
        which is the order the network flattens its outputs in.
        """
        n = self._input_size // stride
        ys, xs = np.mgrid[:n, :n]
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
        return np.repeat(centers, self.ANCHORS_PER_CELL, axis=0)

    def _preprocess(self, frame: Frame) -> tuple[np.ndarray, float]:
        """Letterbox into an input_size square (top-left, zero padded), RGB, (x - 127.5) / 128, NCHW."""
        h, w = frame.shape[:2]
        scale = self._input_size / max(h, w)
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
        canvas = np.zeros((self._input_size, self._input_size, 3), np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        rgb = canvas[:, :, ::-1].astype(np.float32)
        blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob), scale

    def _decode(self, outputs: list[np.ndarray], scale: float) -> list[Detection]:
        n = len(self.STRIDES)
        boxes, landmarks, scores = [], [], []
        for i, stride in enumerate(self.STRIDES):
            s = outputs[i][:, 0]
            keep = s >= self._conf_threshold
            if not keep.any():
                continue
            c = self._centers[i][keep]
            d = outputs[i + n][keep] * stride  # distances: left, top, right, bottom
            k = outputs[i + 2 * n][keep] * stride  # 5 x (dx, dy)
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1))
            landmarks.append(k.reshape(-1, 5, 2) + c[:, None, :])
            scores.append(s[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale  # back to frame pixels
        landmarks = np.concatenate(landmarks) / scale
        scores = np.concatenate(scores)
        order = scores.argsort()[::-1]
        kept = order[_nms(boxes[order], self.NMS_IOU)]
        return [Detection(bbox=boxes[j], landmarks=landmarks[j], confidence=float(scores[j])) for j in kept]


def _nms(boxes: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. `boxes` is (N, 4) xyxy, already sorted best first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.arange(len(boxes))
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep
