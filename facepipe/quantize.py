"""Quantize the embedder to INT8 with ONNX Runtime's static quantizer. `facepipe quantize`.

Two schemes, because the gap between them is the point:

- dpu: symmetric INT8 weights and activations, one scale per tensor,
  then every scale rounded up to a power of two and the weights
  re-quantized to match. That is the arithmetic of a Vitis AI DPU
  (fixed-point, power-of-two scales, per-tensor), so this model's accuracy
  is roughly what the board would see. Rounding up never saturates but
  costs up to one bit; a real DPU quantizer picks the better neighbouring
  power of two, so this is a slightly pessimistic emulation.
- ort: per-channel INT8 weights, asymmetric UINT8 activations: the best
  case for INT8 on an x86 CPU.

Both use percentile calibration: MinMax latches onto activation outliers
and measured markedly worse (0.92 vs 0.96 mean agreement per-tensor,
0.97 vs 0.99 per-channel). The histogram calibrators keep every
activation of every calibration image in memory, which is why the default
is 100 images and not 500.

Static (calibrated) quantization rather than dynamic: a DPU has no float
datapath, so activations must be fixed-point too, which needs a
calibration set to choose their ranges. Output is QDQ format, the same
graph form AMD's vai_q_onnx emits for the Vitis AI execution provider.
The model is first upgraded from opset 11 to 13, which per-channel
DequantizeLinear requires; the upgraded float model agrees with the
original to six decimals.

Needs the `onnx` package: pip install -e ".[quant]"
"""

from pathlib import Path

import cv2
import numpy as np

from facepipe.align import ArcFaceAligner
from facepipe.arcface import ArcFaceEmbedder
from facepipe.config import Config
from facepipe.evaluate import nearest_centre, read_pairs
from facepipe.scrfd import ScrfdDetector

SCHEMES = ("dpu", "ort")


def quantize(cfg: Config, scheme: str, calib_dir: str, pairs: str | None, out: str | None, count: int, check: int) -> int:
    try:
        import onnx
        from onnx import version_converter
        from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        print('the onnx package is needed to quantize: pip install -e ".[quant]"')
        return 1

    float_path = Path(cfg.embedder.model_path)
    out_path = Path(out) if out else float_path.with_name(f"{float_path.stem}_int8_{scheme}.onnx")

    # ---- crops: calibration and a disjoint hold-out for the agreement check ----
    detector = ScrfdDetector(cfg.detector.model_path, cfg.detector.conf_threshold, cfg.detector.input_size)
    detector.load()
    embedder = ArcFaceEmbedder(str(float_path))
    embedder.load()
    aligner = ArcFaceAligner(embedder.input_size)
    excluded = set()
    if pairs:
        excluded = {p for fold in read_pairs(Path(pairs)) for kind in fold for pair in kind for p in pair}
    candidates = [p for p in sorted(Path(calib_dir).rglob("*.jpg")) if p.relative_to(calib_dir).as_posix() not in excluded]
    rng = np.random.default_rng(0)
    rng.shuffle(candidates)
    print(f"{len(candidates)} images available after excluding {len(excluded)} in the evaluation pairs")
    crops = []
    for path in candidates:
        if len(crops) == count + check:
            break
        image = cv2.imread(str(path))
        faces = detector.infer(image) if image is not None else []
        if faces:
            crops.append(aligner.align(image, nearest_centre(faces, image.shape).landmarks))
    calib, holdout = crops[:count], crops[count:]
    print(f"{len(calib)} calibration crops, {len(holdout)} held out for the agreement check")

    # ---- quantize ----
    pre_path = out_path.with_suffix(".pre.onnx")
    onnx.save(version_converter.convert_version(onnx.load(str(float_path)), 13), str(pre_path))
    quant_pre_process(str(pre_path), str(pre_path), skip_symbolic_shape=True)  # shape inference, BN folding
    input_name = embedder._input_name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(calib)

        def get_next(self):
            crop = next(self._it, None)
            return None if crop is None else {input_name: ArcFaceEmbedder.preprocess(crop)}

    if scheme == "dpu":
        quantize_static(str(pre_path), str(out_path), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=False,
                        calibrate_method=CalibrationMethod.Percentile,
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
        model = onnx.load(str(out_path))
        changed = round_scales_to_powers_of_two(model, onnx)
        onnx.save(model, str(out_path))
        print(f"rounded {changed} scales up to powers of two and re-quantized their tensors")
    else:
        quantize_static(str(pre_path), str(out_path), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                        calibrate_method=CalibrationMethod.Percentile)
    pre_path.unlink()
    print(f"{out_path}: {out_path.stat().st_size / 1e6:.1f} MB (float model {float_path.stat().st_size / 1e6:.1f} MB)")

    # ---- agreement: the same crop through both models ----
    quantized = ArcFaceEmbedder(str(out_path))
    quantized.load()
    agreement = np.array([float(embedder.infer(c) @ quantized.infer(c)) for c in holdout])
    print(f"float vs INT8 embedding cosine over {len(holdout)} held-out crops: "
          f"mean {agreement.mean():.4f}, p5 {np.percentile(agreement, 5):.4f}, min {agreement.min():.4f}")
    ops = sorted({n.op_type for n in onnx.load(str(out_path)).graph.node})
    print(f"ops in the quantized graph: {', '.join(ops)}")
    return 0


def round_scales_to_powers_of_two(model, onnx) -> int:
    """Emulate the DPU's fixed-point constraint on a QDQ graph.

    Every QuantizeLinear/DequantizeLinear scale becomes the next power of two
    at or above it (up, so nothing saturates), and any already-quantized
    initializer read through a DequantizeLinear (weights, biases) is
    re-quantized so it still represents the same values.
    """
    inits = {i.name: i for i in model.graph.initializer}
    done: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        scale_name = node.input[1]
        if scale_name in done or scale_name not in inits:
            continue
        done.add(scale_name)
        old = onnx.numpy_helper.to_array(inits[scale_name]).astype(np.float32)
        new = (2.0 ** np.ceil(np.log2(old))).astype(np.float32)
        if node.op_type == "DequantizeLinear" and node.input[0] in inits:
            q = onnx.numpy_helper.to_array(inits[node.input[0]])
            info = np.iinfo(q.dtype)
            requantized = np.clip(np.round(q.astype(np.float64) * old / new), info.min, info.max).astype(q.dtype)
            inits[node.input[0]].CopyFrom(onnx.numpy_helper.from_array(requantized, node.input[0]))
        inits[scale_name].CopyFrom(onnx.numpy_helper.from_array(new.reshape(old.shape), scale_name))
    return len(done)
