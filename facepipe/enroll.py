"""Enrollment: a folder of images of one person in, rows in the store out. `facepipe enroll`."""

from pathlib import Path

import cv2
import numpy as np

from facepipe.align import ArcFaceAligner
from facepipe.arcface import ArcFaceEmbedder
from facepipe.config import Config
from facepipe.scrfd import ScrfdDetector
from facepipe.store import DirectoryStore
from facepipe.types import Identity

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def enroll(cfg: Config, name: str, folder: str) -> int:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {folder}")
        return 1

    detector = ScrfdDetector(cfg.detector.model_path, cfg.detector.conf_threshold, cfg.detector.input_size)
    detector.load()
    embedder = ArcFaceEmbedder(cfg.embedder.model_path)
    embedder.load()
    aligner = ArcFaceAligner(embedder.input_size)
    store = DirectoryStore(cfg.store.path, Path(cfg.embedder.model_path).name)

    embeddings, used = [], []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  {path.name}: unreadable, skipped")
            continue
        faces = detector.infer(image)
        if len(faces) == 0:
            print(f"  {path.name}: no face found, skipped")
            continue
        if len(faces) > 1:
            # Enrolling the wrong person from a group photo is silent and
            # poisons every later match, so this is not guessed at.
            print(f"  {path.name}: {len(faces)} faces found, skipped (crop it to one face)")
            continue
        crop = aligner.align(image, faces[0].landmarks)
        embeddings.append(embedder.infer(crop))
        used.append(str(path))
        print(f"  {path.name}: ok (detection confidence {faces[0].confidence:.2f})")

    if not embeddings:
        print("nothing enrolled")
        return 1
    store.add(Identity(name=name, embeddings=np.stack(embeddings), image_paths=used))
    total = next(i for i in store.identities() if i.name == name).embeddings.shape[0]
    print(f"{name}: {len(used)} of {len(paths)} images enrolled; {total} embeddings in {Path(cfg.store.path) / name}")
    return 0
