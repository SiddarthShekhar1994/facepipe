# facepipe

A face detection and recognition pipeline that runs locally on a laptop:
webcam in, detected and recognized faces out, with a minimal dashboard for
the live feed and for enrolling people.

It is a stand-in for a system that will run detection and recognition on an
AMD FPGA board (on-device inference), with a server doing embedding
comparison and a web dashboard showing the live feed and managing enrolled
profiles. Every stage sits behind an interface so the inference backend can
be swapped for the FPGA one without touching its neighbours.

**Status:** P4 - the full pipeline runs live behind a web dashboard: the
labelled feed, the list of enrolled people, and enroll-from-webcam.
Threshold is provisional until P5.

## Architecture

```
  FrameSource   (webcam | video file | image directory)
       |  Frame: HxWx3 uint8 BGR
       v
  Detector      SCRFD-500M on ONNX Runtime                   ---.
       |  list[Detection]: bbox, 5 landmarks, confidence        |
       v                                                        |  on the target system:
  Aligner       Umeyama similarity fit onto the ArcFace         |  FPGA board
       |        template, warpAffine to 112x112                 |
       |  crop: 112x112 uint8 BGR, one per detection            |
       v                                                        |
  Embedder      MobileFaceNet on ONNX Runtime                ---'
       |  Embedding: (512,) float32, L2-normalized
       v
  Matcher       cosine vs gallery, threshold, top-k  <----  Store   ---.
       |  list[Match] per face; empty list = unknown          one dir  |  server
       v                                                    per person |
  Dashboard     stdlib HTTP: MJPEG feed, enrolled list, enroll form ---'
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
   every enrolled embedding from the `Store` (one matvec; a person with
   several enrolled images is scored by their best one) and returns the
   identities that clear the threshold, best first. An empty list is the
   "unknown" path.
6. The display draws each box with `name similarity`, or "unknown".

Steps 2-5 are `Pipeline.process(frame)` in `pipeline.py`; it returns one
`FaceResult` per face, and both the live loop and the dashboard call it.

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
`models/buffalo_sc/w600k_mbf.onnx` (the embedder, 13 MB). Then:

```
facepipe show-config              # loads and validates config.toml, prints it
facepipe run                      # live webcam window, a name or "unknown" on every face; q or Esc quits
facepipe run --frames 300         # stop after 300 frames and print the timing summary
facepipe enroll alice ./photos    # enroll one person from a folder of .jpg/.png, one face each
facepipe serve                    # dashboard at http://127.0.0.1:8000; Ctrl-C stops it
python -m unittest discover tests # the math tests
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

## Dashboard

`facepipe serve` binds `127.0.0.1:8000` (`--host`, `--port`; loopback by
default because the feed has no authentication) and serves one page: the
live feed with a name or "unknown" on every face, a form that enrolls the
face currently on screen under a typed name, and a table of enrolled
people with their image counts. Plain HTML, no JavaScript, no CSS
framework.

How it is put together, which is the only interesting part:

- One worker thread owns the camera and both models. It runs read ->
  `Pipeline.process` -> draw -> JPEG-encode and publishes the latest
  (raw frame, results, jpeg) as a snapshot under a condition variable.
  Request threads only read snapshots; no model ever runs on one.
- The feed is an MJPEG stream (`multipart/x-mixed-replace`): a single
  `<img src="/stream">`, and the handler writes one JPEG part per new
  snapshot, so the browser sees exactly the pipeline's frame rate.
- Enroll-from-webcam reuses the embedding the pipeline already computed
  for the face on screen (`FaceResult.embedding`), saves the raw frame
  through the same `Store.add` the CLI uses, and hot-swaps the gallery:
  `Pipeline.reload_gallery` replaces one tuple in one assignment, so a
  frame in flight never sees new names with old rows. Zero or several
  faces in view is refused, as in the CLI.
- The HTTP layer is the standard library's `ThreadingHTTPServer`; see the
  tooling table for why not Flask.

Not guarded against: enrolling the same person under two names. The
matcher will then report whichever name has the most similar image.

## Threshold

A face is named when its cosine similarity to an enrolled image reaches
`matcher.threshold` in `config.toml`. **The current value, 0.4, is
provisional.** It was set from a handful of observations, not from data:

| Pair | Cosine |
| --- | --- |
| Same person, same pose, frames 1.5 s apart | 0.92 |
| Five enrollment frames of one person, pairwise | 0.76 - 0.96 |
| Same person over 115 live frames (moving, turning) | 0.46 - 0.92, mean 0.78 |
| Frontal with a drawn overlay vs looking down | 0.53 |
| A face vs a random-noise crop | 0.13 |

Same-person similarities drop into the 0.4-0.5 range under motion blur and
head turns; a few live frames fall below 0.4 and flicker to "unknown".
Where different-person similarities sit is not known yet. The evaluation
(P5) measures both distributions on a labelled set and picks the threshold
from the TAR/FAR curve; this section is rewritten then.

## Enrollment and the store

`facepipe enroll <name> <folder>` runs every image through detect, align
and embed and writes the result under `data/enrolled/<name>/`:

```
data/enrolled/alice/
  meta.json          name, created_at, which embedder file produced the rows, images in row order
  embeddings.npy     (K, 512) float32; row i came from images[i]
  001.jpg 002.jpg    the reference images, copied in as given
```

Why this format:

- `ls data/enrolled` is the list of people. Adding a person, or more images
  of one, touches only that person's directory, so there is no global index
  that a crash mid-write can corrupt. Deleting a person is deleting a
  directory.
- `.npy` is exact float32 with zero dependencies and one line to load.
  JSON would be 512 floats of noise per row, pickle is opaque and unsafe to
  load, and SQLite (standard library, and the obvious next step if this
  grew) is more machinery than a handful of directories need.
- `meta.json` records the embedder file because embeddings from different
  models are silently incomparable. The store refuses to read or append a
  person enrolled with a different embedder than the one configured.
- The originals are stored, not the aligned crops, so the gallery can be
  rebuilt after a model change by re-running enroll.

Images with no face or with more than one face are skipped and named, not
guessed at: enrolling the wrong person from a group photo is silent and
poisons every later match. Crop such images to one face. Re-running
enroll on a name appends to it.

## Layout

```
facepipe/           the package; one module per concern
  types.py          data that crosses stage boundaries: Frame, Detection, Embedding, Match, Identity
  interfaces.py     the six abstract stages
  config.py         TOML -> frozen dataclasses, strict
  cli.py            argparse entry point; the only module that reads argv
  sources.py        FrameSource implementations: WebcamSource
  scrfd.py          Detector implementation: SCRFD on ONNX Runtime (letterbox, anchor decode, NMS)
  align.py          Aligner implementation: Umeyama similarity fit onto the ArcFace template, warpAffine
  arcface.py        Embedder implementation: MobileFaceNet on ONNX Runtime, L2-normalized output
  matcher.py        Matcher implementation: cosine, best row per person, threshold, top-k; build_gallery
  store.py          Store implementation: one directory per person
  ort_session.py    the one place ONNX Runtime sessions are opened (provider, threading, logging)
  pipeline.py       Pipeline.process(frame): detect -> align -> embed -> match, one FaceResult per face
  draw.py           boxes and labels onto a frame; used by the window and the dashboard
  enroll.py         the enroll command
  dashboard.py      the serve command: worker thread, MJPEG stream, enroll form; the only HTTP import
tests/              unittest; only the math that everything else depends on
  timing.py         StageTimer: per-stage ms and FPS for the frame loop
  run.py            the live loop behind `facepipe run`
config.toml         the single config file
pyproject.toml      package metadata and exact dependency pins
```

## Performance

Measured on the development laptop (16 logical cores, CPU execution
provider, 640x480 webcam, one face in frame), from `facepipe run --frames N`
after the camera warm-up second. Live numbers; the benchmark script over a
fixed input will replace them.

| Stage | P1 session, detect only, window | P3 session, full pipeline, window | P4 session, full pipeline, dashboard |
| --- | --- | --- | --- |
| read (webcam) | 2-4 ms | 8 ms | 5-10 ms |
| detect (SCRFD-500M, `input_size=640`) | 23-25 ms (6.3 ms at 320) | 37 ms | 37-39 ms |
| align (per face) | - | 0.7 ms | 0.7 ms |
| embed (MobileFaceNet, per face) | - | 13 ms | 13.5 ms |
| match (5-row gallery) | - | 0.1 ms | 0.1 ms |
| display (draw + imshow + waitKey) | 7-9 ms | 12 ms | - |
| encode (draw + JPEG) | - | - | 1.7 ms |
| end to end | ~27 fps (camera caps at 30) | 13.3 fps | 16 fps |

Two things to read out of this:

- The same detector on the same input measured 23 ms in one session and
  37 ms in another. That is the laptop (CPU frequency and thermal state),
  not the code, and it is why the benchmark will report a spread rather
  than one number.
- With two ONNX Runtime sessions alternating every frame, each pool's
  workers spin-wait after their run and compete with the other pool:
  detect went 37 -> 56 ms and embed 10 -> 20 ms. Disabling spinning
  (`session.intra_op.allow_spinning = 0`, in `ort_session.py`) restores the
  solo numbers. Fewer threads made it worse; both models use the cores.

`display` is dominated by `waitKey`, which pumps the window's message
loop; the dashboard pays 1.7 ms of JPEG encoding instead, which is why it
is faster than the window.

## Model licenses

| Model | Source | License |
| --- | --- | --- |
| MobileFaceNet, `w600k_mbf.onnx` | InsightFace `buffalo_sc` pack, GitHub release v0.7 | Same terms as the detector: non-commercial research only. Trained on WebFace600K, itself a research-use dataset. Every high-accuracy face recognition weight set available is in this position because the training sets are; SFace from OpenCV Zoo (Apache-2.0) is the permissive option at lower accuracy. |
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
| Umeyama similarity fit, in numpy | `cv2.estimateAffinePartial2D`; scikit-image `SimilarityTransform` | The alignment must be a similarity (rotation, uniform scale, translation) fitted over all five landmarks, exactly as at training time. OpenCV's estimator is RANSAC/LMedS-based and with five points may discard one as an outlier. scikit-image is what InsightFace calls, and it computes this same closed form; not worth a dependency for 15 lines. A full affine fit was rejected because it would shear the face. |
| MobileFaceNet (`w600k_mbf`, 512-d) | ResNet-50 (`w600k_r50`, 166 MB); SFace (OpenCV Zoo) | Already in the downloaded pack, ~10x cheaper than the ResNet-50, and the class of network that would actually be deployed to a DPU, which makes the quantization experiment representative. The accuracy cost is measured in the evaluation; switching is a `model_path` change plus a download. SFace is Apache-2.0 but weaker. |
| Store: a directory per person, `.npy` + `meta.json` | one JSON index; SQLite; pickle | See "Enrollment and the store". |
| Matching: cosine, best enrolled row per person | mean embedding per person | Cosine is one dot product because embeddings are unit length. Scoring a person by their best row is what makes enrolling several photos useful: a half-turned query matches the half-turned photo, where a mean of frontal and turned fits neither. The cost is that one mislabelled enrolled image gives that person false matches, which is why enroll refuses to guess on multi-face images. |
| `http.server` (standard library) for the dashboard | Flask; FastAPI + uvicorn | Three endpoints and one page. Flask would be about 40 lines shorter and is the ease choice; it costs seven packages. FastAPI is async and ten-plus packages for a page with no concurrency problem. The dashboard is not what the project is about, and a target box is happier with fewer packages, so zero-dep won. |
| MJPEG stream for the live feed | polling a JPEG URL from JavaScript; WebSocket | One `<img>` tag and no JavaScript; every browser supports it; the server pushes at the pipeline's rate. Polling jitters and needs JS; WebSocket needs a library or a hand-written handshake. |
| `unittest` | pytest | Standard library. pytest is nicer to write and read; that is an ease argument, and it lost against adding a dependency for a handful of tests. |
| OpenCV (`opencv-python`) | PyAV / imageio for capture + Pillow for drawing + Tk for a window | One library covers webcam capture, the display window, drawing, and later the alignment warp and image loading. The `-headless` wheel was rejected because it has no `imshow`. On Windows the default MSMF backend opened faster than DirectShow (0.4 s vs 0.7 s) and negotiated 640x480 on the first try, so no backend override. |
| `argparse` | click, typer | Standard library. A handful of subcommands with a few flags does not justify a dependency. Click is nicer to write; that is an ease argument and it lost. |
