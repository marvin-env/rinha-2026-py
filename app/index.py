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
FAST_CELLS_PATH = RESOURCES / "fast_cells.npy"
FAST_SCORES_PATH = RESOURCES / "fast_scores.npy"

DIM = 14
K = 5
NLIST = 1024
NPROBE = 8
TRAIN_SAMPLE = 200_000
DEDUP_GRID = 16  # 0 disables near-duplicate collapse; ~50% reduction at 16
FAST_LOOKUP_GRID = 8  # coarse grid for fast-path decision lookup
FAST_MIN_SAMPLES = 10  # cell must have at least this many refs to be cached
FAST_MIN_CONFIDENCE = 0.95  # >= this fraud-rate caches as fraud; <= 1-this caches as legit


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


def _dedup_near_duplicates(
    vecs: np.ndarray, lbls: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Snap each coord to a 1/DEDUP_GRID grid, then keep one original vector per
    # unique cell. Label is the majority vote across the collapsed vectors
    # (ties resolved as fraud, since false negatives cost more in the scoring).
    quantized = np.round(vecs * DEDUP_GRID).astype(np.int16)
    _, first_idx, inverse, counts = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    label_sums = np.zeros(first_idx.shape[0], dtype=np.int64)
    np.add.at(label_sums, inverse, lbls.astype(np.int64))
    majority = (label_sums * 2 >= counts).astype(np.uint8)
    return vecs[first_idx], majority


def _build_fast_lookup(vecs: np.ndarray, lbls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Quantize ALL original reference vectors (before dedup) to a coarse grid,
    # aggregate fraud rate per cell, and emit only cells with enough samples
    # and a near-unanimous label. The runtime hash-lookup then returns the
    # cached score directly, bypassing FAISS for those queries.
    quantized = np.round(vecs * FAST_LOOKUP_GRID).astype(np.int8)
    unique_cells, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    fraud_counts = np.zeros(unique_cells.shape[0], dtype=np.int64)
    np.add.at(fraud_counts, inverse, lbls.astype(np.int64))
    fraud_rate = fraud_counts / counts

    confident = (counts >= FAST_MIN_SAMPLES) & (
        (fraud_rate >= FAST_MIN_CONFIDENCE) | (fraud_rate <= 1.0 - FAST_MIN_CONFIDENCE)
    )
    cells = unique_cells[confident]
    scores = fraud_rate[confident].astype(np.float32)
    return cells, scores


def _build_faiss() -> tuple[faiss.Index, np.ndarray, np.ndarray, np.ndarray]:
    raw_vecs, raw_lbls = _read_references()
    fast_cells, fast_scores = _build_fast_lookup(raw_vecs, raw_lbls)
    print(
        f"fast lookup: {fast_cells.shape[0]} confident cells "
        f"(grid={FAST_LOOKUP_GRID}, min_samples={FAST_MIN_SAMPLES}, "
        f"conf={FAST_MIN_CONFIDENCE})"
    )
    np.save(FAST_CELLS_PATH, fast_cells)
    np.save(FAST_SCORES_PATH, fast_scores)

    vecs, lbls = raw_vecs, raw_lbls
    if DEDUP_GRID > 0:
        n_before = vecs.shape[0]
        vecs, lbls = _dedup_near_duplicates(vecs, lbls)
        print(f"dedup: {n_before} -> {vecs.shape[0]} ({100 * (1 - vecs.shape[0] / n_before):.1f}% reduction)")
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
    return index, lbls, fast_cells, fast_scores


def load_index() -> tuple[faiss.Index, np.ndarray, np.ndarray, np.ndarray]:
    if (
        INDEX_PATH.exists()
        and LABELS_PATH.exists()
        and FAST_CELLS_PATH.exists()
        and FAST_SCORES_PATH.exists()
    ):
        index = faiss.read_index(str(INDEX_PATH))
        index.nprobe = NPROBE
        labels = np.load(LABELS_PATH)
        fast_cells = np.load(FAST_CELLS_PATH)
        fast_scores = np.load(FAST_SCORES_PATH)
        return index, labels, fast_cells, fast_scores
    return _build_faiss()


class FaissIndex:
    def __init__(self) -> None:
        self.index, self.labels, fast_cells, fast_scores = load_index()
        # Build dict[bytes -> float] for O(1) hash lookup at query time.
        self.fast_lookup: dict[bytes, float] = {
            fast_cells[i].tobytes(): float(fast_scores[i])
            for i in range(fast_cells.shape[0])
        }

    def search_batch(self, queries: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        _d, ids = self.index.search(q, K)
        return ids

    def fast_score(self, vec: np.ndarray) -> float | None:
        # Quantize the query to the same coarse grid used at build time and
        # check the precomputed map. Returns None if the cell isn't confident.
        key = np.round(vec * FAST_LOOKUP_GRID).astype(np.int8).tobytes()
        return self.fast_lookup.get(key)


if __name__ == "__main__":
    _build_faiss()
    print(
        f"faiss built at {INDEX_PATH} "
        f"({os.path.getsize(INDEX_PATH)} bytes), "
        f"labels at {LABELS_PATH} ({os.path.getsize(LABELS_PATH)} bytes)"
    )
