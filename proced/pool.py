"""Spawn-safe process pool helpers (archi §2 memory/concurrency guardrails).

* explicit **spawn** context — safe with GDAL (no fork-after-use)
* per-worker GDAL cache/thread tuning via ``init_worker``
* BLAS/OpenMP pinned to 1 thread per worker (the parallelism is *processes*;
  letting every worker run a full BLAS team oversubscribes the CPUs)
* executor recycled every ``cfg.pool_chunk`` jobs so worker RSS stays bounded
* ``gc.collect()`` between chunks so freed arrays never accumulate
* live progress bar for every job (see :mod:`proced.progress`)
* stable per-key RNG seeds so parallel scheduling cannot change outcomes
"""

from __future__ import annotations

import gc
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from .config import PipelineConfig
from .progress import ProgressBar, bar as progress_bar

log = logging.getLogger("proced.pool")

# One BLAS/OpenMP thread per worker process: the pipeline parallelises across
# processes, so nested threading only steals cores from sibling workers.
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_blas_threads(n: int = 1) -> None:
    """Default every BLAS/OpenMP runtime to ``n`` threads (user env wins)."""
    for var in BLAS_THREAD_VARS:
        os.environ.setdefault(var, str(n))


def init_worker(gdal_cache_mb: int, gdal_num_threads: str) -> None:
    """Per-process GDAL/BLAS tuning (runs once per worker; archi §2)."""
    os.environ["GDAL_CACHEMAX"] = str(int(gdal_cache_mb))
    os.environ["GDAL_NUM_THREADS"] = str(gdal_num_threads)
    pin_blas_threads(1)
    try:
        import rasterio
        rasterio.Env(GDAL_CACHEMAX=int(gdal_cache_mb), GDAL_NUM_THREADS=str(gdal_num_threads))
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
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
    A progress bar tracks all jobs across chunks; ``gc.collect()`` runs after
    each chunk so released arrays are returned to the OS promptly.
    """
    if not jobs:
        return []
    if init is None:
        init = init_worker
        initargs = (cfg.gdal_cache_mb_per_worker, str(cfg.gdal_num_threads))
    elif initargs is None:
        initargs = ()

    # Children inherit this environment at spawn — set it before forking the
    # pool so their first BLAS load already sees the single-thread default.
    pin_blas_threads(1)

    workers = max(1, int(cfg.workers))
    chunk_size = max(1, int(cfg.pool_chunk))
    results: List[Any] = []
    n_done = 0
    total = len(jobs)
    pb: ProgressBar = progress_bar(total, label, unit=label)

    try:
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
                    pb.update()
                    if on_result is not None:
                        on_result(res)
                    if not pb.enabled and (n_done % 5 == 0 or n_done == total):
                        log.info("Progress: %d/%d %s", n_done, total, label)
            # Recycled workers are gone; drop chunk leftovers before the next.
            gc.collect()
    finally:
        pb.close()
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
    tracebacks readable. Both paths show the same progress bar.
    """
    if not jobs:
        return []
    if max(1, int(cfg.workers)) <= 1:
        if init is not None and initargs is not None:
            init(*initargs)
        out: List[Any] = []
        pb = progress_bar(len(jobs), label, unit=label)
        try:
            for job in jobs:
                out.append(fn(job))
                pb.update()
        finally:
            pb.close()
        return out
    return map_chunked(cfg, jobs, fn, init=init, initargs=initargs, label=label)
