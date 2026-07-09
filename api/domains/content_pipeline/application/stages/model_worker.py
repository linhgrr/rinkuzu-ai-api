"""Persistent subprocess for sentence-transformer pipeline workloads."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import multiprocessing
import time
from typing import Any

from loguru import logger

from api.domains.content_pipeline.domain.errors import PipelineStageTimeoutError


def _normalize_matrix(encoded: Any) -> list[list[float]]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [[float(value) for value in row] for row in encoded]


def _model_worker_entrypoint(conn: Any) -> None:
    from pyvi import ViTokenizer
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cache: dict[tuple[str, str, int | None], Any] = {}

    while True:
        try:
            message = conn.recv()
        except EOFError:
            break

        op = message.get("op")
        if op == "shutdown":
            conn.send({"status": "ok", "payload": None})
            break

        try:
            if op != "encode":
                raise ValueError(f"Unsupported model-worker op: {op}")

            model_name = str(message["model_name"])
            max_seq_length = message.get("max_seq_length")
            cache_key = (model_name, device, max_seq_length)
            model = model_cache.get(cache_key)
            if model is None:
                load_started_at = time.perf_counter()
                model = SentenceTransformer(model_name, device=device)
                if max_seq_length:
                    model.max_seq_length = int(max_seq_length)
                model_cache[cache_key] = model
                logger.info(
                    "[ModelWorkerChild] Loaded model={} device={} max_seq_length={} load_ms={:.1f}",
                    model_name,
                    device,
                    max_seq_length,
                    (time.perf_counter() - load_started_at) * 1000,
                )

            texts = [str(text or "") for text in message.get("texts", [])]
            if message.get("use_vi_tokenizer", False):
                texts = [ViTokenizer.tokenize(text) if text else "" for text in texts]

            encoded = model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=bool(message.get("normalize_embeddings", False)),
                batch_size=int(message.get("batch_size", 32)),
                show_progress_bar=bool(message.get("show_progress_bar", False)),
            )
            conn.send({"status": "ok", "payload": _normalize_matrix(encoded)})
        except BaseException as exc:  # pragma: no cover - subprocess boundary
            conn.send(
                {
                    "status": "err",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    conn.close()


@dataclass
class _WorkerState:
    process: Any
    conn: Any


@dataclass
class _WorkerMetrics:
    starts: int = 0
    restarts: int = 0
    requests: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    cancellations: int = 0


class SentenceTransformerWorkerClient:
    """Single persistent subprocess for pipeline embedding/SAINT encoding."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state: _WorkerState | None = None
        self._metrics = _WorkerMetrics()
        self._request_seq = 0

    def _start_locked(self, *, restart: bool) -> _WorkerState:
        started_at = time.perf_counter()
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_model_worker_entrypoint,
            args=(child_conn,),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._metrics.starts += 1
        if restart:
            self._metrics.restarts += 1
        startup_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "[ModelWorker] Worker {} pid={} startup_ms={:.1f} starts={} restarts={}",
            "restarted" if restart else "started",
            process.pid,
            startup_ms,
            self._metrics.starts,
            self._metrics.restarts,
        )
        return _WorkerState(process=process, conn=parent_conn)

    async def _ensure_started_locked(self) -> _WorkerState:
        if self._state is None:
            self._state = self._start_locked(restart=False)
        elif not self._state.process.is_alive():
            logger.warning("[ModelWorker] Worker pid={} died; restarting", self._state.process.pid)
            await self._close_locked(reason="worker_died")
            self._state = self._start_locked(restart=True)
        return self._state

    async def _close_locked(self, *, reason: str) -> None:
        if self._state is None:
            return
        started_at = time.perf_counter()
        state = self._state
        self._state = None
        with suppress(Exception):
            state.conn.close()
        if state.process.is_alive():
            state.process.kill() if hasattr(state.process, "kill") else state.process.terminate()
        await asyncio.to_thread(state.process.join, 1.0)
        shutdown_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "[ModelWorker] Closed worker pid={} reason={} shutdown_ms={:.1f}",
            state.process.pid,
            reason,
            shutdown_ms,
        )

    async def encode(
        self,
        *,
        texts: list[str],
        model_name: str,
        batch_size: int,
        normalize_embeddings: bool,
        use_vi_tokenizer: bool,
        max_seq_length: int | None,
        show_progress_bar: bool,
        stage_name: str,
        timeout_sec: float | None,
    ) -> list[list[float]]:
        async with self._lock:
            state = await self._ensure_started_locked()
            self._request_seq += 1
            request_id = self._request_seq
            self._metrics.requests += 1
            started_at = time.perf_counter()
            message = {
                "op": "encode",
                "texts": texts,
                "model_name": model_name,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "use_vi_tokenizer": use_vi_tokenizer,
                "max_seq_length": max_seq_length,
                "show_progress_bar": show_progress_bar,
            }
            logger.info(
                "[ModelWorker] Encode request_id={} stage={} pid={} texts={} model={} timeout_sec={}",
                request_id,
                stage_name,
                state.process.pid,
                len(texts),
                model_name,
                timeout_sec,
            )
            state.conn.send(message)

            try:
                if timeout_sec is None:
                    while not await asyncio.to_thread(state.conn.poll, 0.1):
                        await asyncio.sleep(0)
                else:
                    deadline = time.monotonic() + timeout_sec
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        if await asyncio.to_thread(state.conn.poll, min(0.1, remaining)):
                            break
                        await asyncio.sleep(0)
                response = state.conn.recv()
            except asyncio.CancelledError:
                self._metrics.cancellations += 1
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.warning(
                    "[ModelWorker] Encode cancelled request_id={} stage={} pid={} elapsed_ms={:.1f}",
                    request_id,
                    stage_name,
                    state.process.pid,
                    elapsed_ms,
                )
                await self._close_locked(reason=f"cancelled:{stage_name}")
                raise
            except TimeoutError as exc:
                self._metrics.timeouts += 1
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.warning(
                    "[ModelWorker] Encode timeout request_id={} stage={} pid={} elapsed_ms={:.1f} timeout_sec={}",
                    request_id,
                    stage_name,
                    state.process.pid,
                    elapsed_ms,
                    timeout_sec,
                )
                await self._close_locked(reason=f"timeout:{stage_name}")
                raise PipelineStageTimeoutError(stage_name, float(timeout_sec or 0.0)) from exc
            except EOFError as exc:
                self._metrics.failures += 1
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.warning(
                    "[ModelWorker] Encode EOF request_id={} stage={} pid={} elapsed_ms={:.1f}",
                    request_id,
                    stage_name,
                    state.process.pid,
                    elapsed_ms,
                )
                await self._close_locked(reason=f"eof:{stage_name}")
                raise RuntimeError(f"{stage_name} worker exited before returning a result") from exc

            if response.get("status") == "ok":
                payload = response.get("payload")
                self._metrics.successes += 1
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "[ModelWorker] Encode complete request_id={} stage={} pid={} texts={} rows={} elapsed_ms={:.1f}",
                    request_id,
                    stage_name,
                    state.process.pid,
                    len(texts),
                    len(payload) if isinstance(payload, list) else 0,
                    elapsed_ms,
                )
                return payload if isinstance(payload, list) else []

            self._metrics.failures += 1
            error_type = response.get("type", "WorkerError")
            error_message = response.get("message", "unknown worker failure")
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.error(
                "[ModelWorker] Encode failed request_id={} stage={} pid={} elapsed_ms={:.1f} error_type={} error={}",
                request_id,
                stage_name,
                state.process.pid,
                elapsed_ms,
                error_type,
                error_message,
            )
            await self._close_locked(reason=f"error:{stage_name}")
            raise RuntimeError(f"{stage_name} worker failed [{error_type}]: {error_message}")

    async def shutdown(self) -> None:
        async with self._lock:
            if self._state is None:
                return
            started_at = time.perf_counter()
            state = self._state
            self._state = None
            try:
                state.conn.send({"op": "shutdown"})
                if await asyncio.to_thread(state.conn.poll, 1.0):
                    _ = state.conn.recv()
            except Exception:
                logger.warning("[ModelWorker] Graceful shutdown failed, forcing process stop")
            finally:
                with suppress(Exception):
                    state.conn.close()
                if state.process.is_alive():
                    state.process.kill() if hasattr(
                        state.process, "kill"
                    ) else state.process.terminate()
                await asyncio.to_thread(state.process.join, 1.0)
                shutdown_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "[ModelWorker] Shutdown pid={} shutdown_ms={:.1f} requests={} successes={} failures={} timeouts={} cancellations={} starts={} restarts={}",
                    state.process.pid,
                    shutdown_ms,
                    self._metrics.requests,
                    self._metrics.successes,
                    self._metrics.failures,
                    self._metrics.timeouts,
                    self._metrics.cancellations,
                    self._metrics.starts,
                    self._metrics.restarts,
                )


_worker_client: SentenceTransformerWorkerClient | None = None


def get_sentence_transformer_worker() -> SentenceTransformerWorkerClient:
    global _worker_client
    if _worker_client is None:
        _worker_client = SentenceTransformerWorkerClient()
    return _worker_client


async def encode_texts_with_sentence_transformer_worker(
    *,
    texts: list[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    use_vi_tokenizer: bool,
    max_seq_length: int | None,
    show_progress_bar: bool,
    stage_name: str,
    timeout_sec: float | None,
) -> list[list[float]]:
    return await get_sentence_transformer_worker().encode(
        texts=texts,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        use_vi_tokenizer=use_vi_tokenizer,
        max_seq_length=max_seq_length,
        show_progress_bar=show_progress_bar,
        stage_name=stage_name,
        timeout_sec=timeout_sec,
    )


async def shutdown_sentence_transformer_worker() -> None:
    global _worker_client
    if _worker_client is None:
        return
    client = _worker_client
    _worker_client = None
    await client.shutdown()
