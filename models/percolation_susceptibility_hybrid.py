"""Small clusters on CPU threads, huge ones on the GPU -- registered as
MODELS["percolation_susceptibility_hybrid"] in tools/models.py.

The estimand, lattice and seed draw are percolation_susceptibility_layers'
(and so the explore model's): Y = |C(0)| 1{0 open} / p at p = p_c - eps0/x,
site states a pure function of (sample seed, site). Same rng -> same Y, sample
by sample, as both CPU models and as percolation_susceptibility_layers_gpu.

Why (user, 2026-11-12): experiment 59 measured one cluster on the GPU at
~0.12 ms + 5.8 us a layer + 0.7 ns a site, against 87 ns a site on one CPU
thread: the GPU loses below |C| ~ 1.6e4 and wins by 25-55x above 1e7. At
d = 5 almost every sample is small and almost all the work is in a few huge
ones, so each kind goes where it is cheap.

How a sample moves
------------------
Every sample starts on a CPU thread (numba prange), grown breadth first with
a per-thread table that is cleared, not reallocated, between samples. If the
cluster would exceed `cut` sites, the thread gives up and the sample's SEED
goes to the GPU, which regrows the cluster from the origin
(percolation_susceptibility_layers_gpu.grow_cluster). Nothing of the partial
cluster is transferred: the lattice is a function of the seed, so the regrown
cluster is the same one. The RAM -> VRAM traffic of a hand-off is one integer;
the price is the up to `cut` sites the CPU had already grown (at most
cut * ~90 ns), paid only by handed-off samples.

The CPU side needs no layer recycling: it never holds more than `cut` sites
(a table of 2 cut slots, 8 bytes each, per thread). The GPU side recycles
layers as percolation_susceptibility_layers_gpu does.

Concurrency
-----------
The samples are processed in chunks. A chunk runs on all numba threads; its
hand-offs go into a queue that one Python thread feeds to the GPU, while the
next chunk runs on the CPU (the numba batch is nogil, and cupy releases the
GIL while it waits on the device). The GPU thread shares a core with the numba
threads. A call ends when both sides have finished; experiment 60 measures how
the wall time splits between them as a function of `cut`.
"""

from __future__ import annotations

import queue
import threading
import time

import numba
import numpy as np
from numba import njit, prange

from models.percolation_susceptibility import p_at
from models.percolation_susceptibility_explore import (
    DIMS, PROBE_SAMPLES_PER_THREAD, _EMPTY, _is_open, _slot)
from models import percolation_susceptibility_layers_gpu as G

#: sites a CPU thread grows before handing the sample to the GPU. Placeholder
#: until experiment 60 measures the balance; 59's single-thread crossover was
#: ~1.6e4, and a loaded CPU is slower per thread, so the cut is likely lower.
DEFAULT_CUT = 1 << 16

#: samples per chunk, per numba thread
DEFAULT_CHUNK_PER_THREAD = 4096

_OK, _EDGE, _HAND = 0, 1, 2


@njit
def _small(seed, thresh, key0, bits, dim, cut, keys, slots, fifo):
    """(|C| or -1, status) for one seed, holding at most `cut` sites.

    keys must be all _EMPTY on entry; it is again on return (only the slots
    used, recorded in `slots`, are cleared)."""
    if not _is_open(seed, key0, thresh):
        return 0, _OK
    fmask = np.int64((1 << bits) - 1)
    h = _slot(keys, key0)
    keys[h] = key0
    slots[0] = h
    fifo[0] = key0
    n = 1
    head = 0
    status = _OK
    while head < n and status == _OK:
        k = fifo[head]
        head += 1
        for j in range(dim):
            shift = bits * j
            f = (k >> shift) & fmask
            if f == 0 or f == fmask:
                status = _EDGE
                break
            step = np.int64(1) << shift
            for nk in (k + step, k - step):
                if not _is_open(seed, nk, thresh):
                    continue
                h = _slot(keys, nk)
                if keys[h] == nk:
                    continue
                if n == cut:
                    status = _HAND
                    break
                keys[h] = nk
                slots[n] = h
                fifo[n] = nk
                n += 1
            if status != _OK:
                break
    for i in range(n):
        keys[slots[i]] = _EMPTY
    return (n if status == _OK else -1), status


@njit(parallel=True, nogil=True)
def _cpu_batch(seeds, thresh, key0, bits, dim, cut, n_chunks):
    n = seeds.size
    size = np.empty(n, dtype=np.int64)
    status = np.empty(n, dtype=np.int64)
    tsize = 1 << 4
    while tsize < 2 * cut:
        tsize <<= 1
    for c in prange(n_chunks):
        keys = np.full(tsize, _EMPTY, dtype=np.int64)
        slots = np.empty(cut, dtype=np.int64)
        fifo = np.empty(cut, dtype=np.int64)
        for r in range(c * n // n_chunks, (c + 1) * n // n_chunks):
            size[r], status[r] = _small(seeds[r], thresh, key0, bits, dim, cut,
                                        keys, slots, fifo)
    return size, status


def device_fill() -> int:
    """ModelSpec.device_fill, as the CPU layers model's."""
    return PROBE_SAMPLES_PER_THREAD * numba.get_num_threads()


def explore_hybrid(x: int, n: int = 1, dim: int = 3, eps0: float = 0.15,
                   p: float | None = None, p_c: float | None = None,
                   cut: int = DEFAULT_CUT, max_live: int = G.DEFAULT_MAX_LIVE,
                   chunk: int | None = None,
                   rng: np.random.Generator | None = None,
                   ) -> tuple[np.ndarray, dict]:
    """n samples of Y = |C(0)| / p (same rng -> same Y as explore_layers).

    info: size, reads (1 + 2 dim |C|), on_gpu (bool per sample), p, and the
    call's timings: wall, cpu_s (time inside the CPU batches), gpu_s (time
    the GPU thread spent growing), gpu_wait_s (wall after the last CPU chunk,
    i.e. time only the GPU was working)."""
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}; got {dim}")
    if int(cut) < 1:
        raise ValueError(f"cut must be >= 1; got {cut}")
    x = int(x)
    p = p_at(x, dim, eps0, p_c) if p is None else float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    bits, key0 = G._geometry(dim)
    rng = rng if rng is not None else np.random.default_rng()
    # the layers model's draw, so all these models see the same lattices
    seeds = rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64)
    thresh = min(int(p * 2.0 ** 64), 2 ** 64 - 1)
    threads = numba.get_num_threads()
    chunk = int(chunk) if chunk else DEFAULT_CHUNK_PER_THREAD * threads

    size = np.zeros(n, dtype=np.int64)
    on_gpu = np.zeros(n, dtype=bool)
    todo: queue.Queue = queue.Queue()
    gpu = {"s": 0.0, "err": None, "stop": False}

    def gpu_worker() -> None:
        while True:
            item = todo.get()
            if item is None:
                return
            if gpu["err"] is not None or gpu["stop"]:
                continue            # drain: the call is failing anyway
            r, s = item
            t = time.perf_counter()
            try:
                size[r] = G.grow_cluster(s, thresh, key0, bits, dim, max_live)["size"]
            except BaseException as err:  # re-raised in the caller's thread
                gpu["err"] = err
            gpu["s"] += time.perf_counter() - t

    worker = threading.Thread(target=gpu_worker, daemon=True)
    t_wall = time.perf_counter()
    worker.start()
    cpu_s = 0.0
    try:
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            t = time.perf_counter()
            sz, st = _cpu_batch(seeds[lo:hi], np.uint64(thresh), np.int64(key0),
                                bits, dim, int(cut), max(1, min(hi - lo, threads)))
            cpu_s += time.perf_counter() - t
            if np.any(st == _EDGE):
                raise RuntimeError(
                    f"a cluster reached the +-2**{bits - 1} key range at x = {x}, "
                    f"dim = {dim} (p = {p:.8g}); it is not wrapped or truncated, so "
                    "the sample cannot be reported")
            size[lo:hi] = np.where(st == _OK, sz, 0)
            for i in np.flatnonzero(st == _HAND):
                on_gpu[lo + i] = True
                todo.put((lo + i, int(seeds[lo + i])))
            if gpu["err"] is not None:
                break
    except BaseException:
        gpu["stop"] = True          # don't grow queued clusters for a failed call
        raise
    finally:
        t_cpu_done = time.perf_counter()
        todo.put(None)
        worker.join()
    if gpu["err"] is not None:
        raise gpu["err"]
    wall = time.perf_counter() - t_wall
    reads = np.where(size > 0, 1 + 2 * dim * size, 1)
    return size / p, {"size": size, "reads": reads, "on_gpu": on_gpu, "p": p,
                      "wall": wall, "cpu_s": cpu_s, "gpu_s": gpu["s"],
                      "gpu_wait_s": time.perf_counter() - t_cpu_done}


def simulate(x: int, n: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """MODELS["percolation_susceptibility_hybrid"].simulate.

    params: {"dim", "eps0", "p", "p_c": as percolation_susceptibility_layers,
             "cut": sites a CPU thread grows before handing off (default 2**16),
             "max_live": the GPU side's cap (default 2**26)}."""
    y, _ = explore_hybrid(
        x, n=n, dim=int(params.get("dim", 3)),
        eps0=float(params.get("eps0", 0.15)),
        p=params.get("p"), p_c=params.get("p_c"),
        cut=int(params.get("cut", DEFAULT_CUT)),
        max_live=int(params.get("max_live", G.DEFAULT_MAX_LIVE)), rng=rng)
    return y
