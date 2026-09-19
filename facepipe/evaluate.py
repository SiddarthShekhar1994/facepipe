"""Evaluation: similarity distributions, TAR/FAR, and the threshold. `facepipe eval`.

Two labelled sets:

- LFW with its official pairs.txt: 10 folds of 300 same-person and 300
  different-person pairs. Strangers to the gallery, and a published
  reference for this model, so a wrong alignment or preprocessing shows.
- The webcam set: every probe under <probes>/<name>/ scored against that
  name's enrolled images (same person) and every LFW face scored against
  the enrolled gallery (a stranger at the camera). This is the deployed
  situation: one gallery, one threshold, "unknown" below it.

Per-image embeddings are cached per embedder file so a re-run, or the
INT8 comparison, does not repeat ten minutes of inference.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from facepipe.config import Config
from facepipe.pipeline import STAGES, Pipeline
from facepipe.timing import StageTimer
from facepipe.types import Detection

GRID = np.round(np.arange(-0.2, 1.0001, 0.001), 3)  # candidate thresholds
FAR_TARGETS = (0.01, 0.001)


def evaluate(cfg: Config, lfw: str, pairs: str, probes: str, folds: int | None, out: str) -> int:
    lfw_dir, out_dir = Path(lfw), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(cfg, StageTimer(STAGES))
    identities = pipeline.load()
    cache = EmbeddingCache(out_dir / f"embeddings_{Path(cfg.embedder.model_path).name}.npz")

    # ---- LFW pairs ----
    pair_folds = read_pairs(Path(pairs))
    if folds:
        pair_folds = pair_folds[:folds]
    unique = sorted({p for fold in pair_folds for kind in fold for pair in kind for p in pair})
    print(f"LFW: {len(pair_folds)} folds, {sum(len(f[0]) + len(f[1]) for f in pair_folds)} pairs, {len(unique)} images")
    lfw_failed = 0
    for n, rel in enumerate(unique, 1):
        if not cache.has(rel):
            cache.put(rel, embed_file(pipeline, lfw_dir / rel, nearest_centre))
        if cache.get(rel) is None:
            lfw_failed += 1
        if n % 500 == 0 or n == len(unique):
            print(f"  embedded {n}/{len(unique)}", flush=True)
    cache.save()
    print(f"LFW: {lfw_failed} of {len(unique)} images with no face detected")

    def scores(pairs_):
        return np.array([cosine(cache.get(a), cache.get(b)) for a, b in pairs_])  # NaN where an image failed

    fold_scores = [(scores(same), scores(diff)) for same, diff in pair_folds]
    same = np.concatenate([s for s, _ in fold_scores])
    diff = np.concatenate([d for _, d in fold_scores])
    same_ok, diff_ok = same[~np.isnan(same)], diff[~np.isnan(diff)]

    print()
    print("### LFW similarity distributions (pairs with a detection failure excluded)")
    print()
    print_percentiles({"same person": same_ok, "different person": diff_ok})

    print()
    print("### LFW operating points")
    print()
    print("| Criterion | Threshold | TAR | FAR |")
    print("| --- | --- | --- | --- |")
    chosen = {}
    for target in FAR_TARGETS:
        t = threshold_at_far(diff_ok, target)
        chosen[f"FAR {target:.1%}"] = t
        print(f"| FAR = {target:.1%} | {t:.3f} | {tar(same_ok, t):.4f} | {far(diff_ok, t):.4f} |")
    t_eer = eer_threshold(same_ok, diff_ok)
    chosen["EER"] = t_eer
    print(f"| equal error rate | {t_eer:.3f} | {tar(same_ok, t_eer):.4f} | {far(diff_ok, t_eer):.4f} |")
    t_acc = best_accuracy_threshold(same_ok, diff_ok)
    chosen["max accuracy"] = t_acc
    print(f"| max accuracy ({accuracy(same_ok, diff_ok, t_acc):.4f}) | {t_acc:.3f} | {tar(same_ok, t_acc):.4f} | {far(diff_ok, t_acc):.4f} |")

    accs_excl, accs_err = [], []
    for k, (s_k, d_k) in enumerate(fold_scores if len(fold_scores) > 1 else []):
        s_train = np.concatenate([s for j, (s, _) in enumerate(fold_scores) if j != k])
        d_train = np.concatenate([d for j, (_, d) in enumerate(fold_scores) if j != k])
        s_train, d_train = s_train[~np.isnan(s_train)], d_train[~np.isnan(d_train)]
        t_k = best_accuracy_threshold(s_train, d_train)
        correct = np.sum(s_k >= t_k) + np.sum(d_k < t_k)  # NaN compares False: a failure counts as an error
        accs_err.append(correct / (len(s_k) + len(d_k)))
        accs_excl.append(correct / (np.sum(~np.isnan(s_k)) + np.sum(~np.isnan(d_k))))
    if accs_excl:
        print()
        print(f"{len(fold_scores)}-fold LFW accuracy (threshold fitted on the other folds): "
              f"{np.mean(accs_excl):.4f} +/- {np.std(accs_excl):.4f} excluding detection failures, "
              f"{np.mean(accs_err):.4f} +/- {np.std(accs_err):.4f} counting them as errors")

    # ---- webcam set: probes vs the enrolled gallery, LFW faces vs the enrolled gallery ----
    probe_sets = collect_probes(pipeline, Path(probes), identities)
    lfw_embeddings = np.stack([e for rel in unique if (e := cache.get(rel)) is not None])
    print()
    print("### Webcam set")
    print()
    for name, (probe_emb, enrolled, n_failed, n_multi) in probe_sets.items():
        pairwise = probe_emb @ enrolled.T  # (probes, enrolled rows)
        best = pairwise.max(axis=1)  # what the matcher scores each probe by
        stranger_best = (lfw_embeddings @ enrolled.T).max(axis=1)
        print(f"{name}: {len(probe_emb)} probes vs {len(enrolled)} enrolled images "
              f"({n_failed} probes with no face, {n_multi} with several: largest taken); "
              f"{len(lfw_embeddings)} LFW faces as strangers at the camera")
        print()
        print_percentiles({
            "probe vs enrolled, all pairs": pairwise.ravel(),
            "probe vs enrolled, best per probe": best,
            "LFW face vs enrolled, best per face": stranger_best,
        })
        print()
        print("| Threshold | from | probes accepted (TAR) | LFW faces accepted as this person (FAR) |")
        print("| --- | --- | --- | --- |")
        rows = [(t, label) for label, t in chosen.items()] + [(0.4, "provisional")]
        for t, label in sorted(rows):
            print(f"| {t:.3f} | {label} | {tar(best, t):.4f} | {far(stranger_best, t):.5f} |")

        with (out_dir / f"tar_far_{name}_{Path(cfg.embedder.model_path).stem}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "lfw_tar", "lfw_far", f"{name}_probe_tar", f"lfw_faces_accepted_as_{name}"])
            for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
                w.writerow([f"{t:.2f}", f"{tar(same_ok, t):.4f}", f"{far(diff_ok, t):.4f}",
                            f"{tar(best, t):.4f}", f"{far(stranger_best, t):.5f}"])
    print()
    print(f"curves written to {out_dir}")
    return 0


# ---- embedding ----

class EmbeddingCache:
    """Path -> embedding, or None for an image with no detected face. One .npz per embedder."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, np.ndarray | None] = {}
        if path.exists():
            z = np.load(path)
            for key, vec in zip(z["keys"].tolist(), z["vectors"]):
                self._data[key] = None if np.isnan(vec[0]) else vec
            print(f"cache: {len(self._data)} embeddings from {path}")

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> np.ndarray | None:
        return self._data[key]

    def put(self, key: str, vec: np.ndarray | None) -> None:
        self._data[key] = vec

    def save(self) -> None:
        if not self._data:
            return
        dim = next(len(v) for v in self._data.values() if v is not None)
        keys = list(self._data)
        vectors = np.stack([self._data[k] if self._data[k] is not None else np.full(dim, np.nan, np.float32) for k in keys])
        np.savez(self._path, keys=np.array(keys), vectors=vectors.astype(np.float32))


def embed_file(pipeline: Pipeline, path: Path, pick) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    faces = pipeline.detector.infer(image)
    if not faces:
        return None
    return pipeline.embedder.infer(pipeline.aligner.align(image, pick(faces, image.shape).landmarks))


def nearest_centre(faces: list[Detection], shape) -> Detection:
    """LFW frames its labelled person in the middle; background faces are not the subject."""
    cy, cx = shape[0] / 2, shape[1] / 2
    return min(faces, key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2)


def largest(faces: list[Detection], shape) -> Detection:
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    return float(a @ b) if a is not None and b is not None else float("nan")


def read_pairs(path: Path) -> list[tuple[list, list]]:
    """pairs.txt: a header `folds pairs`, then per fold `pairs` same lines and `pairs` different lines."""
    lines = path.read_text().splitlines()
    n_folds, n_per = (int(x) for x in lines[0].split())
    folds, i = [], 1
    for _ in range(n_folds):
        same, diff = [], []
        for _ in range(n_per):
            name, a, b = lines[i].split()
            i += 1
            same.append((f"{name}/{name}_{int(a):04d}.jpg", f"{name}/{name}_{int(b):04d}.jpg"))
        for _ in range(n_per):
            n1, a, n2, b = lines[i].split()
            i += 1
            diff.append((f"{n1}/{n1}_{int(a):04d}.jpg", f"{n2}/{n2}_{int(b):04d}.jpg"))
        folds.append((same, diff))
    return folds


def collect_probes(pipeline: Pipeline, probes_dir: Path, identities) -> dict:
    """{name: (probe embeddings (P, D), enrolled rows (K, D), probes with no face, probes with several)}."""
    out = {}
    enrolled = {i.name: i.embeddings for i in identities}
    if not probes_dir.exists():
        print(f"no probes directory {probes_dir}; skipping the webcam set")
        return out
    for folder in sorted(p for p in probes_dir.iterdir() if p.is_dir()):
        if folder.name not in enrolled:
            print(f"probes/{folder.name}: nobody enrolled under that name, skipped")
            continue
        vectors, n_failed, n_multi = [], 0, 0
        for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            image = cv2.imread(str(path))
            faces = pipeline.detector.infer(image) if image is not None else []
            if not faces:
                n_failed += 1
                continue
            n_multi += len(faces) > 1
            vectors.append(pipeline.embedder.infer(pipeline.aligner.align(image, largest(faces, image.shape).landmarks)))
        if vectors:
            out[folder.name] = (np.stack(vectors), enrolled[folder.name], n_failed, n_multi)
    return out


# ---- metrics ----

def tar(same: np.ndarray, t: float) -> float:
    return float(np.mean(same >= t))


def far(diff: np.ndarray, t: float) -> float:
    return float(np.mean(diff >= t))


def accuracy(same: np.ndarray, diff: np.ndarray, t: float) -> float:
    return float((np.sum(same >= t) + np.sum(diff < t)) / (len(same) + len(diff)))


def threshold_at_far(diff: np.ndarray, target: float) -> float:
    """The lowest threshold that lets through at most `target` of the different-person pairs."""
    s = np.sort(diff)
    allowed = int(np.floor(target * len(s)))
    if allowed == 0:
        return float(s[-1]) + 1e-4
    return float((s[-allowed - 1] + s[-allowed]) / 2)


def eer_threshold(same: np.ndarray, diff: np.ndarray) -> float:
    fars = np.array([far(diff, t) for t in GRID])
    frrs = np.array([1.0 - tar(same, t) for t in GRID])
    return float(GRID[np.argmin(np.abs(fars - frrs))])


def best_accuracy_threshold(same: np.ndarray, diff: np.ndarray) -> float:
    accs = np.array([accuracy(same, diff, t) for t in GRID])
    return float(GRID[np.argmax(accs)])


def print_percentiles(named: dict[str, np.ndarray]) -> None:
    qs = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    print("| pairs | n | " + " | ".join(f"p{q}" for q in qs) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in qs) + " |")
    for label, values in named.items():
        p = np.percentile(values, qs)
        print(f"| {label} | {len(values)} | " + " | ".join(f"{v:.3f}" for v in p) + " |")
