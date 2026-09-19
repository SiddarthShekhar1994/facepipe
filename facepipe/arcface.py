"""ArcFace-family face embedder on ONNX Runtime.

The network takes one aligned crop and returns a 512-vector. The buffalo
files have no input normalization baked into the graph (they open with
Conv/PReLU, no Sub/Mul), so the wrapper applies (x - 127.5) / 127.5 on
RGB. The raw output is not unit length, so the wrapper L2-normalizes it:
that is the Embedding contract the matcher relies on.
"""

import numpy as np
import onnxruntime as ort

from facepipe.interfaces import Embedder
from facepipe.ort_session import open_session
from facepipe.types import Embedding


class ArcFaceEmbedder(Embedder):
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._session: ort.InferenceSession | None = None
        self._input_name = ""
        self._input_size = 0

    def load(self) -> None:
        self._session = open_session(self._model_path)
        inp = self._session.get_inputs()[0]
        n, c, h, w = inp.shape
        if not (isinstance(h, int) and h == w):
            raise ValueError(f"{self._model_path}: expected a fixed square input, got shape {inp.shape}")
        self._input_name = inp.name
        self._input_size = h

    @property
    def input_size(self) -> int:
        return self._input_size

    def infer(self, crop: np.ndarray) -> Embedding:
        rgb = crop[:, :, ::-1].astype(np.float32)
        blob = np.ascontiguousarray(((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None])
        (out,) = self._session.run(None, {self._input_name: blob})
        vec = out[0]
        return vec / np.linalg.norm(vec)
