"""Store implementation: one directory per enrolled person.

    <root>/<name>/
        meta.json        name, created_at, embedder file, images in row order
        embeddings.npy   (K, D) float32; row i came from images[i]
        001.jpg ...      the reference images, copied in as given

`ls <root>` is the list of people. Adding to a name touches only that
name's directory, so there is no global index to corrupt. Deleting a
person is deleting their directory. The embedder's file name is recorded
because embeddings from different models are silently incomparable.
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facepipe.interfaces import Store
from facepipe.types import Identity

META = "meta.json"
EMBEDDINGS = "embeddings.npy"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")  # a safe directory name and a drawable label


class DirectoryStore(Store):
    def __init__(self, root: str, embedder_name: str):
        self._root = Path(root)
        self._embedder_name = embedder_name

    def add(self, identity: Identity) -> None:
        """Create the person, or append these rows and images to an existing one."""
        if not NAME_PATTERN.match(identity.name):
            raise ValueError(f"name {identity.name!r}: use letters, digits, '_' or '-' only")
        if identity.embeddings.shape[0] != len(identity.image_paths):
            raise ValueError("one image path per embedding row is required")
        folder = self._root / identity.name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / META).exists():
            meta = self._read_meta(folder)
            rows = np.load(folder / EMBEDDINGS)
        else:
            meta = {
                "name": identity.name,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "embedder": self._embedder_name,
                "images": [],
            }
            rows = np.empty((0, identity.embeddings.shape[1]), dtype=np.float32)
        for src in identity.image_paths:
            dst = folder / f"{len(meta['images']) + 1:03d}{Path(src).suffix.lower()}"
            shutil.copyfile(src, dst)
            meta["images"].append(dst.name)
        rows = np.concatenate([rows, identity.embeddings.astype(np.float32)])
        np.save(folder / EMBEDDINGS, rows)
        (folder / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def identities(self) -> list[Identity]:
        out = []
        if not self._root.exists():
            return out
        for folder in sorted(p for p in self._root.iterdir() if (p / META).exists()):
            meta = self._read_meta(folder)
            rows = np.load(folder / EMBEDDINGS)
            if rows.shape[0] != len(meta["images"]):
                raise ValueError(f"{folder}: {rows.shape[0]} embeddings but {len(meta['images'])} images listed")
            out.append(
                Identity(
                    name=meta["name"],
                    embeddings=rows,
                    image_paths=[str(folder / img) for img in meta["images"]],
                )
            )
        return out

    def _read_meta(self, folder: Path) -> dict:
        meta = json.loads((folder / META).read_text(encoding="utf-8"))
        if meta["embedder"] != self._embedder_name:
            raise ValueError(
                f"{folder} was enrolled with {meta['embedder']} but the configured embedder is "
                f"{self._embedder_name}; delete the folder and enroll again"
            )
        return meta
