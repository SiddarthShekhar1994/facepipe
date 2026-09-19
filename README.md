# facepipe

A face detection and recognition pipeline that runs locally on a laptop:
webcam in, detected and recognized faces out, with a minimal dashboard for
the live feed and for enrolling people.

It is a stand-in for a system that will run detection and recognition on an
AMD FPGA board (on-device inference), with a server doing embedding
comparison and a web dashboard showing the live feed and managing enrolled
profiles. Every stage sits behind an interface so the inference backend can
be swapped for the FPGA one without touching its neighbours.

**Status:** P0 - skeleton, stage interfaces, config loading, CLI. No models yet.

## Architecture

```
  FrameSource   (webcam | video file | image directory)
       |  Frame: HxWx3 uint8 BGR
       v
  Detector      load() / infer()                        ---.
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

Requires Python 3.11 or newer (developed on 3.14).

```
py -3.14 -m venv .venv            # python3 -m venv .venv outside Windows
.venv\Scripts\activate            # source .venv/bin/activate outside Windows
pip install -e .
```

## Layout

```
facepipe/           the package; one module per concern
config.toml         the single config file
pyproject.toml      package metadata and exact dependency pins
```

## Tooling decisions

Each entry says what was chosen, what it replaced, and why. Ease-of-use
reasons are labelled as such.

| Choice | Rejected | Why |
| --- | --- | --- |
| Python 3.11+ | - | `tomllib` in the standard library. Developed on 3.14, the newest version both OpenCV and ONNX Runtime ship wheels for. |
| `pyproject.toml` with exact pins | `requirements.txt` | Either works for an application. One file instead of two, and it gives a `facepipe` console script. |
| numpy | - | The type of every stage boundary: frames, crops, embeddings. |
