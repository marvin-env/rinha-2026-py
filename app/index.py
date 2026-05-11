from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import faiss
import numpy as np

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
REFS_GZ = RESOURCES / "references.json.gz"
INDEX_PATH = RESOURCES / "references.faiss"
LABELS_PATH = RESOURCES / "references.labels.npy"

DIM = 14
K = 5
NLIST = 1024
NPROBE = 8
TRAIN_SAMPLE = 200_000


def _read_references() -> tuple[np.ndarray, np.ndarray]:
    vectors: list[list[float]] = []
    labels: list[int] = []
    with gzip.open(REFS_GZ, "rt") as f:
        data = json.load(f)
    for row in data:
        vectors.append(row["vector"])
        labels.append(1 if row["label"] == "fraud" else 0)
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.uint8),
    )


def _build_faiss() -> tuple[faiss.Index, np.ndarray]:
    vecs, lbls = _read_references()
    n = vecs.shape[0]

    rng = np.random.default_rng(42)
    train_idx = rng.choice(n, size=min(TRAIN_SAMPLE, n), replace=False)
    train = np.ascontiguousarray(vecs[train_idx])

    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, DIM, NLIST, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
    )
    index.train(train)
    index.add(vecs)
    index.nprobe = NPROBE

    faiss.write_index(index, str(INDEX_PATH))
    np.save(LABELS_PATH, lbls)
    return index, lbls


def load_index() -> tuple[faiss.Index, np.ndarray]:
    if INDEX_PATH.exists() and LABELS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        index.nprobe = NPROBE
        labels = np.load(LABELS_PATH)
        return index, labels
    return _build_faiss()


class FaissIndex:
    def __init__(self) -> None:
        self.index, self.labels = load_index()

    def search_batch(self, queries: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        _d, ids = self.index.search(q, K)
        return ids


if __name__ == "__main__":
    _build_faiss()
    print(
        f"faiss built at {INDEX_PATH} "
        f"({os.path.getsize(INDEX_PATH)} bytes), "
        f"labels at {LABELS_PATH} ({os.path.getsize(LABELS_PATH)} bytes)"
    )
