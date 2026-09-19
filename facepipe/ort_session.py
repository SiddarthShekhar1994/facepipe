"""One place to open ONNX Runtime sessions with the settings every model shares."""

import onnxruntime as ort


def open_session(model_path: str, silence_warnings: bool = False) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # Each session owns a thread pool whose workers spin-wait after a run.
    # With two models alternating every frame, the idle pool's spinners
    # compete with the active pool for cores: measured detect 37 -> 56 ms
    # and embed 10 -> 20 ms. Turning spinning off restores the solo numbers.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if silence_warnings:
        opts.log_severity_level = 3
    # Explicit provider list: on the board this line becomes VitisAIExecutionProvider.
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
