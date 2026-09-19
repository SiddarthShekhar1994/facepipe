# facepipe

A face detection and recognition pipeline that runs locally on a laptop:
webcam in, detected and recognized faces out, with a minimal dashboard for
the live feed and for enrolling people.

It is a stand-in for a system that will run detection and recognition on an
AMD FPGA board (on-device inference), with a server doing embedding
comparison and a web dashboard showing the live feed and managing enrolled
profiles. Every stage sits behind an interface so the inference backend can
be swapped for the FPGA one without touching its neighbours.

**Status:** P1 - webcam capture and face detection with per-stage timing. No
recognition yet.

## Architecture

```
  FrameSource   (webcam | video file | image directory)
       |  Frame: HxWx3 uint8 BGR
       v
  Detector      SCRFD-500M on ONNX Runtime              ---.
       |  list[Detection]: bbox, 5 landmarks, confidence   |
       v                                                   |  on the target system:
  Aligner       5-point similarity transform               |  FPGA board
       |  crop: fixed size, one per detection              |
       v                                                   |
  Embedder      load() / infer()                        ---'
       |  Embedding: (D,) float32, L2-normalized
       v
  Matcher       cosine vs gallery, threshold, top-k  <----  Store   ---.
       |  list[Match] per face; empty list = unknown                   |  server
       v                                                               |
  Dashboard     live feed with labels, enrolled list, enroll        ---'
```

Rules the code follows:

- The core pipeline is plain Python and numpy. No web framework imports
  below the dashboard layer.
- Every model sits behind a class with `load()` and `infer()`. Weight paths
  come from config, never from code.
- One config file, `config.toml`, holds thresholds, model paths, source and
  resolution.

## One frame, end to end

1. `FrameSource.read()` returns a `Frame`: one HxWx3 uint8 BGR array, exactly
   as OpenCV delivers it. `None` means the stream is exhausted.
2. `Detector.infer(frame)` returns a `Detection` per face: a pixel bounding
   box, five landmarks (left eye, right eye, nose, left mouth corner, right
   mouth corner) and a confidence.
3. For each detection, `Aligner.align(frame, landmarks)` computes the
   similarity transform that maps the five landmarks onto a canonical
   template and warps the face into a fixed-size crop. This is what makes
   embeddings comparable across pose and scale.
4. `Embedder.infer(crop)` returns an `Embedding`: a float32 vector,
   L2-normalized by contract, so cosine similarity is a dot product.
5. `Matcher.match(embedding, gallery_names, gallery)` compares it against
   every enrolled embedding from the `Store` and returns the identities that
   clear the threshold, best first. An empty list is the "unknown" path.
6. The dashboard draws each box with its label, or "unknown".

## Setup

Requires Python 3.11 or newer (developed on 3.14) and a webcam.

```
py -3.14 -m venv .venv            # python3 -m venv .venv outside Windows
.venv\Scripts\activate            # source .venv/bin/activate outside Windows
pip install -e .
```

Model weights are not in the repo (see [Model licenses](#model-licenses)).
Download the InsightFace `buffalo_sc` pack (15 MB) and unpack it into
`models/`, which is gitignored:

```
mkdir models
curl -L -o models/buffalo_sc.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
python -c "import zipfile; zipfile.ZipFile('models/buffalo_sc.zip').extractall('models/buffalo_sc')"
```

That yields `models/buffalo_sc/det_500m.onnx` (the detector, 2.5 MB) and
`w600k_mbf.onnx` (a recognition model, not used yet). Then:

```
facepipe show-config              # loads and validates config.toml, prints it
facepipe run                      # live webcam window; q or Esc quits
facepipe run --frames 300         # stop after 300 frames and print the timing summary
```

`run` prints one line per second with the rolling mean of each stage in
milliseconds and the frames per second, then a summary with p95s on exit.
The first second is camera warm-up (auto-exposure settling) and is slow;
ignore it.

`facepipe --help` lists the subcommands. Every subcommand takes `--config`
(default `config.toml`). Paths inside the config are relative to the
directory you run from, so run from the repo root.

## Configuration

`config.toml` is the single config file. It is loaded strictly: an unknown
key, a missing key, or a value of the wrong type stops the program at
startup with the key named, rather than silently using a default. Sections
are added by the phase that reads them, so every key that exists is
consumed by something.

## Layout

```
facepipe/           the package; one module per concern
  types.py          data that crosses stage boundaries: Frame, Detection, Embedding, Match, Identity
  interfaces.py     the six abstract stages
  config.py         TOML -> frozen dataclasses, strict
  cli.py            argparse entry point; the only module that reads argv
  sources.py        FrameSource implementations: WebcamSource
  scrfd.py          Detector implementation: SCRFD on ONNX Runtime (letterbox, anchor decode, NMS)
  timing.py         StageTimer: per-stage ms and FPS for the frame loop
  run.py            the live loop behind `facepipe run`
config.toml         the single config file
pyproject.toml      package metadata and exact dependency pins
```

## Performance

Measured on the development laptop (CPU only, 640x480 webcam), from
`facepipe run --frames 90`, after the camera warm-up second. The detector
runs on the CPU execution provider.

| Stage | `input_size=640` | `input_size=320` |
| --- | --- | --- |
| detect (SCRFD-500M) | 23-25 ms | 6.3 ms |
| display (draw + imshow + waitKey) | 7-9 ms | - |
| read (webcam) | 2-4 ms, camera-paced | - |
| end to end | ~27 fps (camera caps at 30) | - |

`read` is cheap because the camera produces frames at 30 fps in the
background; by the time a 25 ms detection finishes, the next frame is
already waiting. `display` is dominated by `waitKey`, which pumps the
window's message loop. A benchmark over a fixed input replaces these live
numbers once alignment, embedding and matching exist.

## Model licenses

| Model | Source | License |
| --- | --- | --- |
| SCRFD-500M, `det_500m.onnx` | InsightFace `buffalo_sc` pack, GitHub release v0.7 | InsightFace's code is MIT, but its README states that the training data and the models trained on it "are available for non-commercial research purposes only", and that this applies to manual downloads from GitHub as well. This project is research/educational use. A commercial deployment would need weights trained on licensed data; YuNet (MIT) is the permissively licensed detector option. |

## Tooling decisions

Each entry says what was chosen, what it replaced, and why. Ease-of-use
reasons are labelled as such.

| Choice | Rejected | Why |
| --- | --- | --- |
| Python 3.11+ | - | `tomllib` in the standard library. Developed on 3.14, the newest version both OpenCV and ONNX Runtime ship wheels for. |
| `pyproject.toml` with exact pins | `requirements.txt` | Either works for an application. One file instead of two, and it gives a `facepipe` console script. |
| numpy | - | The type of every stage boundary: frames, crops, embeddings. |
| TOML via `tomllib` | YAML (PyYAML), JSON | Zero dependencies and supports comments. YAML would add a dependency for no gain at this size; JSON cannot carry comments. |
| `dataclasses` for the config schema | pydantic | Pydantic is a dependency for the sake of ~a dozen keys. A 40-line strict mapper covers unknown keys, missing keys and wrong types. |
| ONNX Runtime | PyTorch checkpoints; OpenCV `cv2.dnn` | One runtime for every model, and the `.onnx` file is the same artifact the FPGA flow starts from: AMD's Vitis AI quantizes ONNX graphs and runs them through an ONNX Runtime execution provider, so the laptop path and the board path share a model file. PyTorch would add ~2 GB of dependency to run a 2.5 MB network. `cv2.dnn` would run it but has no quantization tooling and is not the deployment path. |
| SCRFD-500M with keypoints (InsightFace) | YuNet (OpenCV Zoo), RetinaFace-MobileNet, UltraFace, the `insightface` package | The detector must output 5 landmarks or there is nothing to align on, which rules out UltraFace. SCRFD's landmarks use the same convention as the ArcFace alignment template, so detection and recognition agree by construction. YuNet is MIT-licensed and smaller but has lower recall on small and hard faces, and its raw ONNX needs the same hand-written decode. RetinaFace is older with no advantage. The `insightface` pip package would do detect+align+embed in one call, which hides the module boundaries this project exists to show, and needs a C++ toolchain on Windows. SCRFD's weights are non-commercial research only; see Model licenses. Larger SCRFD variants (2.5G, 10G) are a `model_path` change. |
| OpenCV (`opencv-python`) | PyAV / imageio for capture + Pillow for drawing + Tk for a window | One library covers webcam capture, the display window, drawing, and later the alignment warp and image loading. The `-headless` wheel was rejected because it has no `imshow`. On Windows the default MSMF backend opened faster than DirectShow (0.4 s vs 0.7 s) and negotiated 640x480 on the first try, so no backend override. |
| `argparse` | click, typer | Standard library. A handful of subcommands with a few flags does not justify a dependency. Click is nicer to write; that is an ease argument and it lost. |
