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


class BatchMaker:
    def __init__(self, max_batch_size: int = 128) -> None:
        self._pending: list[tuple[asyncio.Future, np.ndarray]] = []
        self._running: bool = False
        self._max_batch_size = max_batch_size

    def _fraud_score(self, row: int):
        how_many_similar_vectors_are_frauds = int(labels[row].sum())
        return how_many_similar_vectors_are_frauds / K

    async def submit(self, requested_vectorized_payload: np.ndarray) -> float:
        loop = asyncio.get_running_loop()
        pending_fraud_score_future = loop.create_future()
        self._pending.append((pending_fraud_score_future, requested_vectorized_payload))
        
        if not self._running:
            self._running = True
            asyncio.create_task(self._batch_processing())

        return await pending_fraud_score_future

    async def _batch_processing(self) -> None:
        try:
            while self._pending:
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
                await send({"type": "lifespan.startup.complete"})
            elif received_msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
