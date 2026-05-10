from __future__ import annotations

import asyncio

import numpy as np
import orjson

from app.index import FaissIndex
from app.vectorize import vectorize

THRESHOLD = 0.6
K = 5

faiss = FaissIndex()
labels = faiss.labels

_HEADERS_JSON = [(b"content-type", b"application/json")]
_EMPTY = b""

_WARMUP_BATCHES = 8
_WARMUP_BATCH_SIZE = 64


def _warmup_index() -> None:
    # Touch IVF posting lists ahead of traffic so the first real batches
    # don't pay cold-page-cache cost (index is ~66MB on disk).
    rng = np.random.default_rng(0)
    for _ in range(_WARMUP_BATCHES):
        q = rng.random((_WARMUP_BATCH_SIZE, 14), dtype=np.float32)
        faiss.search_batch(q)


class BatchMaker:
    def __init__(self, max_batch_size: int = 32, batch_window: float = 0.005) -> None:
        self._pending: list[tuple[asyncio.Future, np.ndarray]] = []
        self._running: bool = False
        self._max_batch_size = max_batch_size
        self._batch_window = batch_window
        self._batch_ready: asyncio.Event | None = None

    def _fraud_score(self, row: int):
        how_many_similar_vectors_are_frauds = int(labels[row].sum())
        return how_many_similar_vectors_are_frauds / K

    async def submit(self, requested_vectorized_payload: np.ndarray) -> float:
        loop = asyncio.get_running_loop()
        pending_fraud_score_future = loop.create_future()
        self._pending.append((pending_fraud_score_future, requested_vectorized_payload))

        if self._batch_ready is not None and len(self._pending) >= self._max_batch_size:
            self._batch_ready.set()

        if not self._running:
            self._running = True
            asyncio.create_task(self._batch_processing())

        return await pending_fraud_score_future

    async def _batch_processing(self) -> None:
        if self._batch_ready is None:
            self._batch_ready = asyncio.Event()
        try:
            while self._pending:
                # Coalesce: wait up to batch_window OR until cap hit (event set in submit).
                self._batch_ready.clear()
                if len(self._pending) < self._max_batch_size:
                    try:
                        await asyncio.wait_for(self._batch_ready.wait(), timeout=self._batch_window)
                    except asyncio.TimeoutError:
                        pass

                current_batch = self._pending[:self._max_batch_size]
                self._pending = self._pending[self._max_batch_size:]

                try:
                    batch_pending_vectors = np.stack([v for _, v in current_batch])
                    similar_vectors = await asyncio.to_thread(faiss.search_batch, batch_pending_vectors)

                    for (peding_fraud_score_future, _), row in zip(current_batch, similar_vectors):
                        if not peding_fraud_score_future.done():
                            fraud_score = self._fraud_score(row)
                            peding_fraud_score_future.set_result(fraud_score)

                except Exception as e:
                    for peding_fraud_score_future, _ in current_batch:
                        if not peding_fraud_score_future.done():
                            peding_fraud_score_future.set_exception(e)
        finally:
            self._running = False
            if self._pending:
                self._running = True
                asyncio.create_task(self._batch_processing())


batch_maker = BatchMaker()


async def app(scope, receive, send):
    scope_type = scope["type"]

    if scope_type == "http":
        path = scope["path"]
        method = scope["method"]

        if method == "POST" and path == "/fraud-score":
            byte_body = bytearray()
            has_body_to_receive = True
            while has_body_to_receive:
                received_msg = await receive()
                byte_body.extend(received_msg.get("body", b""))
                has_body_to_receive = received_msg.get("more_body", False)

            received_payload = orjson.loads(bytes(byte_body))
            vectorized_payload = vectorize(received_payload)
            fraud_score = await batch_maker.submit(vectorized_payload)
            response = orjson.dumps({"approved": fraud_score < THRESHOLD, "fraud_score": fraud_score})

            await send({"type": "http.response.start", "status": 200, "headers": _HEADERS_JSON})
            await send({"type": "http.response.body", "body": response})
            return

        if method == "GET" and path == "/ready":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": _EMPTY})
            return

        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": _EMPTY})
        return

    if scope_type == "lifespan":
        while True:
            received_msg = await receive()
            if received_msg["type"] == "lifespan.startup":
                await asyncio.to_thread(_warmup_index)
                await send({"type": "lifespan.startup.complete"})
            elif received_msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
