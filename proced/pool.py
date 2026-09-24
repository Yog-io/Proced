"""Spawn-safe process pool helpers (archi §2 memory/concurrency guardrails).

* explicit **spawn** context — safe with GDAL (no fork-after-use)
* per-worker GDAL cache/thread tuning via ``init_worker``
* executor recycled every ``cfg.pool_chunk`` jobs so worker RSS stays bounded
* stable per-key RNG seeds so parallel scheduling cannot change outcomes
"""

from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("proced.pool")


def init_worker(gdal_cache_mb: int, gdal_num_threads: str) -> None:
    """Per-process GDAL tuning (runs once per worker; archi §2)."""
    os.environ["GDAL_CACHEMAX"] = str(int(gdal_cache_mb))
    os.environ["GDAL_NUM_THREADS"] = str(gdal_num_threads)
    try:
        import rasterio
        rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb), GDAL_NUM_THREADS=str(gdal_num_threads))
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
    )


def stable_rng(seed: int, *parts: str) -> random.Random:
    h = seed
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(h)


def numpy_rng(seed: int, *parts: str) -> np.random.Generator:
    r = stable_rng(seed, *parts)
    return np.random.default_rng(r.randrange(2 ** 31))


def map_chunked(
    cfg: PipelineConfig,
    jobs: Sequence[Any],
    fn: Callable[[Any], Any],
    *,
    init: Optional[Callable[..., None]] = None,
    initargs: Optional[tuple] = None,
    on_result: Optional[Callable[[Any], None]] = None,
    label: str = "jobs",
) -> List[Any]:
    """Run ``fn`` over ``jobs`` in a spawn ProcessPoolExecutor.

    The executor is torn down and recreated every ``cfg.pool_chunk`` jobs so
    worker RSS cannot grow without bound on long runs (archi memory guardrail).
    """
    if not jobs:
        return []
    if init is None:
        init = init_worker
        initargs = (cfg.gdal_cache_mb_per_worker, str(cfg.gdal_num_threads))
    elif initargs is None:
        initargs = ()

    workers = max(1, int(cfg.workers))
    chunk_size = max(1, int(cfg.pool_chunk))
    results: List[Any] = []
    n_done = 0
    total = len(jobs)

    for i in range(0, total, chunk_size):
        chunk = list(jobs[i:i + chunk_size])
        import multiprocessing as mp
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=init,
            initargs=initargs,
        ) as ex:
            futures = [ex.submit(fn, job) for job in chunk]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                n_done += 1
                if on_result is not None:
                    on_result(res)
                if n_done % 5 == 0 or n_done == total:
                    log.info("Progress: %d/%d %s", n_done, total, label)
    return results


def run_serial_or_pool(
    cfg: PipelineConfig,
    jobs: Sequence[Any],
    fn: Callable[[Any], Any],
    *,
    init: Optional[Callable[..., None]] = None,
    initargs: Optional[tuple] = None,
    label: str = "jobs",
) -> List[Any]:
    """Pool when workers > 1 and there is work; serial path for workers == 1.

    The serial path avoids spawn overhead in tests / tiny runs and keeps
    tracebacks readable.
    """
    if not jobs:
        return []
    if max(1, int(cfg.workers)) <= 1:
        if init is not None and initargs is not None:
            init(*initargs)
        return [fn(job) for job in jobs]
    return map_chunked(cfg, jobs, fn, init=init, initargs=initargs, label=label)
