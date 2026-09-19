# facepipe

A face detection and recognition pipeline that runs locally on a laptop:
webcam in, detected and recognized faces out, with a minimal dashboard for
the live feed and for enrolling people.

It is a stand-in for a system that will run detection and recognition on an
AMD FPGA board (on-device inference), with a server doing embedding
comparison and a web dashboard showing the live feed and managing enrolled
profiles. Every stage sits behind an interface so the inference backend can
be swapped for the FPGA one without touching its neighbours.

**Status:** P5 - the full pipeline, the dashboard, a benchmark, and an
evaluation on LFW plus webcam probes. The threshold is set from that data.

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
`models/buffalo_sc/w600k_mbf.onnx` (the embedder, 13 MB).

Only for `facepipe eval`: LFW and its pairs file, 172 MB, 13k files.
Extract them anywhere (a synced folder is a poor choice for 13k files)
and point `--lfw` / `--pairs` at them:

```
curl -L -o lfw.tgz https://ndownloader.figshare.com/files/5976018
curl -L -o pairs.txt https://ndownloader.figshare.com/files/5976006
tar -xzf lfw.tgz          # -> lfw/<Person_Name>/<Person_Name>_0001.jpg ...
```

Then:

```
facepipe show-config              # loads and validates config.toml, prints it
facepipe run                      # live webcam window, a name or "unknown" on every face; q or Esc quits
facepipe run --frames 300         # stop after 300 frames and print the timing summary
facepipe enroll alice ./photos    # enroll one person from a folder of .jpg/.png, one face each
facepipe serve                    # dashboard at http://127.0.0.1:8000; Ctrl-C stops it
facepipe bench --video clip.mp4   # per-stage latency and FPS over a fixed input (or --images DIR)
facepipe eval --lfw DIR --pairs F  # similarity distributions, TAR/FAR, threshold; see Threshold below
python -m unittest discover tests # the math tests
```

`run` prints one line per second with the rolling mean of each stage in
milliseconds and the frames per second, then a summary with p95s on exit.
The first second is camera warm-up (auto-exposure settling) and is slow;
ignore it.

`facepipe --help` lists the subcommands. Every subcommand takes `--config`
(default `config.toml`). Paths inside the config are relative to the
directory you run from, so run from the repo root.

`[source]` in the config selects the input for `run` and `serve`: the
webcam (`kind = "webcam"`), a video file (`kind = "video"`, `path`) or a
directory of images (`kind = "images"`, `path`), so both work without a
camera.

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

`matcher.threshold = 0.26`. A face is named when its cosine similarity to
the best of a person's enrolled images reaches this; below it the face is
"unknown". The value is the point where the false accept rate on LFW's
different-person pairs is 0.1%, rounded up (rounding up can only lower
the false accept rate). Everything below is the output of `facepipe eval`.

### The test set

- **Strangers: LFW** (Labeled Faces in the Wild), 13,233 images of 5,749
  people, with its official `pairs.txt`: 10 folds of 300 same-person and
  300 different-person pairs, all 6,000 used. LFW is not committed; see
  Setup for the download and Model licenses for its terms. LFW frames the
  labelled person in the middle, so when several faces are detected the
  one nearest the centre is taken (that is the protocol's definition; it
  is not what enrollment does). 10 of the 7,701 images had no detection at
  `conf_threshold = 0.5`: those pairs are excluded from the distributions
  and counted as errors in the second accuracy figure.
- **The enrolled person: 30 probe frames** of the one enrolled person,
  captured in a single sitting a day after enrollment - head turned and
  tilted, leaning in and back, talking, gesturing; same room, same
  lighting, glasses on throughout - scored against the 5 enrolled images.
  This is the deployed situation and an easy one, and it is reported
  separately for that reason.
- **Strangers at the camera:** all 7,691 detected LFW faces scored against
  the enrolled gallery, best of the 5 rows, exactly as the matcher scores
  a live face. This is the false-accept case that matters for a system
  with an "unknown" path.

### LFW: similarity distributions

| pairs | n | min | p1 | p5 | p25 | median | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same person | 2989 | -0.004 | 0.289 | 0.407 | 0.539 | 0.616 | 0.692 | 0.791 | 0.862 | 0.963 |
| different person | 2993 | -0.216 | -0.146 | -0.103 | -0.041 | 0.004 | 0.048 | 0.112 | 0.166 | 0.332 |

### LFW: operating points

| Criterion | Threshold | TAR | FAR |
| --- | --- | --- | --- |
| FAR = 1.0% | 0.169 | 0.9963 | 0.0097 |
| **FAR = 0.1%** | **0.256** | **0.9920** | **0.0007** |
| equal error rate | 0.188 | 0.9957 | 0.0043 |
| max accuracy (0.9963) | 0.209 | 0.9943 | 0.0017 |

10-fold accuracy with the threshold fitted on the other nine folds:
**99.52% +/- 0.28%** excluding the detection failures, 99.22% +/- 0.60%
counting them as errors. InsightFace does not publish an LFW figure for
this particular WebFace600K MobileFaceNet; the MobileFaceNet paper (Chen
et al., 2018) reports 99.55% for its MS1M-trained model. Being in that
range is the check that the alignment and preprocessing here reproduce
what the model was trained with; a wrong landmark order or a wrong
normalization would cost whole points.

### The enrolled person and strangers at the camera

| pairs | n | min | p1 | p5 | p25 | median | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probe vs enrolled, all 30 x 5 pairs | 150 | 0.471 | 0.482 | 0.503 | 0.566 | 0.613 | 0.649 | 0.688 | 0.695 | 0.709 |
| probe vs enrolled, best per probe (what the matcher uses) | 30 | 0.585 | 0.589 | 0.604 | 0.632 | 0.650 | 0.686 | 0.695 | 0.705 | 0.709 |
| LFW face vs enrolled, best per face | 7691 | -0.217 | -0.113 | -0.078 | -0.023 | 0.014 | 0.056 | 0.114 | 0.155 | 0.239 |

| Threshold | from | probes accepted (TAR) | LFW faces accepted as the enrolled person |
| --- | --- | --- | --- |
| 0.169 | LFW FAR 1% | 30/30 | 38 of 7691 (0.49%) |
| 0.188 | LFW EER | 30/30 | 13 of 7691 (0.17%) |
| 0.209 | LFW max accuracy | 30/30 | 3 of 7691 (0.04%) |
| **0.256** | **LFW FAR 0.1%** | **30/30** | **0 of 7691** |
| 0.400 | the provisional value | 30/30 | 0 of 7691 |

### Reading it

- The two LFW distributions barely overlap: same-person p5 is 0.41,
  different-person p95 is 0.11. That is the model working.
- The webcam same-person set is easy (best-per-probe minimum 0.585) because
  it was captured in the conditions it was enrolled in. LFW's same-person
  tail (p1 = 0.29, minimum below zero) is what years, lighting and pose do,
  and it is why the threshold is set on LFW rather than on the webcam set.
- 0.26 keeps 99.2% of LFW same-person pairs and rejects every one of 7,691
  strangers against this gallery, with 0.02 to spare above the closest
  stranger (0.239). The provisional 0.40 would have rejected ~5% of
  LFW-style same-person pairs and sat within 0.06 of the motion-blurred
  live frames seen in P3.
- The false accept rate of the deployed system grows with the gallery:
  each enrolled person, and each extra image per person under best-row
  scoring, is another chance for a stranger to clear the threshold. This
  evaluation is one person with five images; re-run `facepipe eval` after
  the gallery grows and look at the last table again.

Curves (threshold vs LFW TAR/FAR, probe TAR, strangers accepted) are
written to `--out` as CSV. Re-run:

```
facepipe eval --lfw <lfw dir> --pairs <pairs.txt> --probes data/eval/probes --out data/eval/results
```

Probes are `data/eval/probes/<name>/*.jpg` for any enrolled `<name>`;
capture them with any camera app. Per-image embeddings are cached per
embedder file, so a re-run costs seconds and the INT8 comparison reuses
the float side.

## Known limitations

Specific to this build, with the evidence where there is any.

- **Pose.** Alignment is a 2D similarity transform; it cannot undo a head
  turn. A profile crop puts the template's far-eye point on the side of the
  head (seen in P2), and profile embeddings score far below frontal ones:
  the dashboard showed "unknown" for the enrolled person in profile with a
  hand at the face. Enroll frontal images; expect misses beyond ~45
  degrees.
- **Lighting and blur.** Live similarity of the enrolled person ranged
  0.46-0.92 over 150 frames under motion; the probe set, captured still,
  never dropped below 0.585. Strong backlight and fast motion are the
  cases that reach the threshold.
- **Occlusion.** Glasses were on in every enrolled and probe image, so
  their effect is unmeasured here; masks and hands over the face reduce
  the detector's confidence first (it uses `conf_threshold = 0.5`) and
  the embedding second.
- **Single-image enrollment is weak.** Best-row scoring means one enrolled
  image covers one pose and one lighting; the five-image enrollment here
  scored its own probes 0.47-0.71 pairwise but 0.59-0.71 best-per-probe.
  Enroll several images, in the conditions the camera will see.
- **Demographic bias in the pretrained weights.** Both models were trained
  on web-scraped datasets (WIDER FACE for the detector, WebFace600K for the
  embedder) whose demographic balance is not controlled, and face
  recognition models are documented to have higher error rates for some
  groups than others (NIST FRVT reports). LFW itself is skewed towards
  public figures, mostly white and male, so the 99.5% here is not a
  guarantee for other faces. This project measures one enrolled person.
- **Same person, two names** is not prevented; the matcher reports
  whichever name has the closest image.
- **No liveness detection.** A photo of an enrolled person held up to the
  camera is that person. Out of scope by design.
- **Small-face recall.** SCRFD-500M at `input_size = 640` handles faces
  down to roughly 20 px; at 320 it loses small and distant faces, which is
  the price of its 3x speed.
- **False accepts scale with the gallery**, as described under Threshold.

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
  sources.py        FrameSource implementations: webcam, video file, image directory; make_source()
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
  bench.py          the bench command: fixed input, warm-up, median/p95 per stage, FPS, ranges over repeats
  evaluate.py       the eval command: LFW pairs protocol, webcam probes vs the gallery, TAR/FAR, threshold
tests/              unittest; only the math that everything else depends on
  timing.py         StageTimer: per-stage ms and FPS for the frame loop
  run.py            the live loop behind `facepipe run`
config.toml         the single config file
pyproject.toml      package metadata and exact dependency pins
```

## Performance

`facepipe bench` runs a fixed input through the whole pipeline: warm-up
frames are discarded, then N frames are measured, and it reports median
and p95 per stage, mean per face for the per-face stages, and FPS for the
pipeline alone (detect + align + embed + match) and end to end. It refuses
the webcam on purpose: a camera paces the loop and changes the picture
every run. `--repeat` reports ranges across runs, because this laptop's CPU
changes state between sessions (the same detector on the same input has
measured 16 ms and 37 ms hours apart). Every row below is `--repeat 3`,
300 frames after 30 warm-up.

Development laptop: 16 logical cores, ONNX Runtime 1.30 CPU execution
provider, one thread pool per model, spinning off (see `ort_session.py`).

| Input | detector `input_size` | read | detect (median, p95) | align per face | embed per face | match | pipeline fps | end-to-end fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| webcam clip, 640x480, 30 s, one face | 640 | 0.3 ms | 16.0-16.9 ms, p95 20 | 0.3 ms | 5.8-6.1 ms | 0.0 ms | 42.6-43.9 | 41.7-43.0 |
| webcam clip, 640x480, 30 s, one face | 320 | 0.3 ms | 5.2-6.3 ms, p95 6-8 | 0.3 ms | 5.6-7.0 ms | 0.0 ms | 72-88 | 70-85 |
| LFW `George_W_Bush/`, 530 press photos 250x250, 1.3 faces/frame | 640 | 0.7-1.5 ms | 14.4-16.5 ms, p95 19 | 0.3 ms | 5.8-6.2 ms | 0.0 ms | 40.0-42.5 | 37.6-41.0 |

Reproduce with `facepipe bench --video data/bench/clip.mp4 --repeat 3`
(any 640x480 clip with one face; ours is a 30-second webcam recording and is
not committed), `--input-size 320` for the second row, and
`facepipe bench --images <lfw>/George_W_Bush --repeat 3` for the third,
which anyone with LFW can run.

What the numbers say: the detector is the cost, and its cost is set by
`input_size`, not by the frame - halving the letterbox side quarters the
work and buys 3x on detection. Embedding is ~6 ms per face and scales with
faces in frame. Alignment and matching are noise. The camera, when there is
one, caps everything at 30 fps.

### Live sessions

The same stages measured inside `facepipe run` and `facepipe serve` in
earlier phases, with the camera pacing the loop; kept because they show the
display cost and the day-to-day variance.

| Stage | P1, detect only, window | P3, full pipeline, window | P4, full pipeline, dashboard |
| --- | --- | --- | --- |
| read (webcam) | 2-4 ms | 8 ms | 5-10 ms |
| detect (`input_size=640`) | 23-25 ms | 37 ms | 37-39 ms |
| align + embed + match (one face) | - | 14 ms | 14 ms |
| display (imshow + waitKey) / encode (JPEG) | 7-9 ms | 12 ms | 1.7 ms |
| end to end | ~27 fps | 13.3 fps | 16 fps |

Two things happened between sessions: the two ONNX Runtime sessions were
found to fight over cores when their thread pools spin-wait (detect went
37 -> 56 ms until spinning was disabled), and the laptop itself ran the
identical detector at 23, 37 and 16 ms on different days. The benchmark
table above is the one to quote; the live numbers are what a user sees.

## Model licenses

| Model | Source | License |
| --- | --- | --- |
| MobileFaceNet, `w600k_mbf.onnx` | InsightFace `buffalo_sc` pack, GitHub release v0.7 | Same terms as the detector: non-commercial research only. Trained on WebFace600K, itself a research-use dataset. Every high-accuracy face recognition weight set available is in this position because the training sets are; SFace from OpenCV Zoo (Apache-2.0) is the permissive option at lower accuracy. |
| LFW (evaluation data, not a model) | University of Massachusetts; mirrored on figshare, files 5976018 (`lfw.tgz`, 172 MB) and 5976006 (`pairs.txt`) | Distributed for research; the images are web-scraped and their subjects did not consent to face-recognition use, which is why it is used here for measurement only and never committed. |
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
| LFW pairs protocol for the evaluation | photographing people I know; a synthetic set | Public, research-licensed, 6,000 labelled pairs with a published reference point for this class of model, so a subtly wrong alignment or normalization would show up as lost accuracy. A hand-made set could not have said that. It is combined with webcam probes of the enrolled person because LFW says nothing about this camera. |
| Threshold at LFW FAR = 0.1% | EER; max accuracy; the midpoint between the webcam distributions | A system with an "unknown" path is judged on strangers it lets in, so a false-accept target is the right criterion. EER and max-accuracy thresholds (0.19-0.21) let 13 and 3 of 7,691 strangers through this one-person gallery; a webcam-only midpoint (~0.41) would be tuned to one sitting's conditions. |
| `facepipe bench` over a file, medians and ranges | timing the live loop | A camera paces the loop at its own frame rate and changes the picture every run; a file does neither. Median and p95 instead of mean because the first frames after a model loads are slow and a mean hides the shape. Ranges over repeats because the laptop's CPU state moves the numbers by 2x between sessions and one number would be a lie. |
| `unittest` | pytest | Standard library. pytest is nicer to write and read; that is an ease argument, and it lost against adding a dependency for a handful of tests. |
| OpenCV (`opencv-python`) | PyAV / imageio for capture + Pillow for drawing + Tk for a window | One library covers webcam capture, the display window, drawing, and later the alignment warp and image loading. The `-headless` wheel was rejected because it has no `imshow`. On Windows the default MSMF backend opened faster than DirectShow (0.4 s vs 0.7 s) and negotiated 640x480 on the first try, so no backend override. |
| `argparse` | click, typer | Standard library. A handful of subcommands with a few flags does not justify a dependency. Click is nicer to write; that is an ease argument and it lost. |
