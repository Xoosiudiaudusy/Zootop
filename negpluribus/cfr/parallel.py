"""Multiprocess MCCFR: K independent workers, periodic additive merge.

    trainer = MCCFRTrainer(spec, bucketer, seed=0)
    trainer.train_parallel(30_000, workers=8, sync_every=1000)
    trainer.strategy() / trainer.save_checkpoint(...)   # as after train()

How it works (the Pluribus recipe, in processes instead of threads):

* every round the master broadcasts its tables (regrets, strategy sums, visits) to K worker
  processes; worker ``w`` runs the iterations ``t = base + 1 + w, base + 1 + w + K, ...`` of the
  round with its own RNG stream, so Linear CFR weights ``t`` cover the same range a serial run
  would cover and the global iteration count simply advances by ``K * sync_every``;
* each worker sends back the *difference* between its tables and the broadcast (plus every
  new key); regrets, strategy sums and visit counts are additive, so the master sums the
  differences of all workers into its tables and the next round starts from there.

What this changes versus a serial run: inside a round the workers act on a strategy that is
``sync_every`` iterations stale for the updates the others make - the same "stale sigma"
semantics as lock-free multithreaded MCCFR.  It is statistically equivalent, not seed-identical;
``scripts/bench_backends.py`` shows the exploitability curve against the serial one.

The workers are plain ``MCCFRTrainer`` objects (any backend), kept alive between rounds so their
E[HS] caches persist.  ``spawn`` start method: works on Windows, needs no fork.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

from ..abstraction import EquityBucketer, PotentialAwareBucketer
from .game import GameSpec

Tables = Dict[str, Tuple[List[str], List[float], List[float], int]]

_worker = None  # per-process trainer


def _bucketer_state(bk) -> dict:
    """Picklable description of a fitted bucketer of either kind (rebuilt in each worker)."""
    if isinstance(bk, PotentialAwareBucketer):
        return bk.to_dict()
    return {"kind": "ehs", "n_buckets": bk.n_buckets, "samples": bk.samples, "boundaries": bk.boundaries}


def _bucketer_from_state(d: dict):
    if d.get("kind", "ehs") == "potential":
        return PotentialAwareBucketer.from_dict(d)
    bk = EquityBucketer(d["n_buckets"], d["samples"])
    bk.boundaries = {int(k): list(v) for k, v in d["boundaries"].items()}
    return bk


def _init_worker(spec: GameSpec, bucketer_state: dict, linear: bool, backend: str, cache_caps=None) -> None:
    global _worker
    from .mccfr import MCCFRTrainer

    bk = _bucketer_from_state(bucketer_state)
    kwargs = {"backend": backend}
    if backend == "cpp":
        kwargs["threads"] = 1
        kwargs["cache_caps"] = cache_caps
    _worker = MCCFRTrainer(spec, bk, seed=0, linear=linear, **kwargs)


def _load_tables(trainer, tables: Tables) -> None:
    from .mccfr import Node

    nodes = {}
    for k, (actions, regret, ssum, visits) in tables.items():
        n = Node(list(actions))
        n.regret, n.strategy_sum, n.visits = list(regret), list(ssum), visits
        nodes[k] = n
    trainer.nodes = nodes


def _worker_round(args) -> Tuple[Tables, int]:
    worker_id, seed, ts, tables = args
    import random

    tr = _worker
    _load_tables(tr, tables)
    tr.rng = random.Random(seed)
    touched0 = tr.nodes_touched
    for t in ts:
        tr.iteration = t - 1
        tr.iterate()
    delta: Tables = {}
    nodes = tr.nodes
    for k, n in nodes.items():
        base = tables.get(k)
        if base is None:
            delta[k] = (list(n.actions), list(n.regret), list(n.strategy_sum), n.visits)
            continue
        _, br, bs, bv = base
        if n.visits == bv and n.regret == br and n.strategy_sum == bs:
            continue
        delta[k] = (
            list(n.actions),
            [a - b for a, b in zip(n.regret, br)],
            [a - b for a, b in zip(n.strategy_sum, bs)],
            n.visits - bv,
        )
    return delta, tr.nodes_touched - touched0


def merge_into(nodes: dict, delta: Tables) -> None:
    """Add a worker's differences into a ``{key: Node}`` table (creating missing keys)."""
    from .mccfr import Node

    for k, (actions, dr, ds, dv) in delta.items():
        n = nodes.get(k)
        if n is None:
            n = Node(list(actions))
            nodes[k] = n
        if n.actions != list(actions):
            continue  # cannot happen in self-play; never mix tables of different grids
        for i in range(len(n.regret)):
            n.regret[i] += dr[i]
            n.strategy_sum[i] += ds[i]
        n.visits += dv


def train_parallel(master, iterations: int, workers: Optional[int] = None, sync_every: int = 1000, log_every: int = 0):
    """Run ``iterations`` more iterations of ``master`` (an MCCFRTrainer) on ``workers`` processes."""
    workers = max(1, int(workers or (os.cpu_count() or 1)))
    sync_every = max(1, int(sync_every))
    spec = master.spec
    bk = master.bucketer
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    done = 0
    rounds = 0
    cpp_master = hasattr(master, "add_tables")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(spec, _bucketer_state(bk), master.linear, master.backend, getattr(master, "cache_caps", None))) as pool:
        while done < iterations:
            base = master.iteration
            n_round = min(sync_every * workers, iterations - done)
            if cpp_master:
                tables: Tables = master.export_tables()
            else:
                tables = {k: (n.actions, n.regret, n.strategy_sum, n.visits) for k, n in master.nodes.items()}
            jobs = []
            for w in range(workers):
                ts = list(range(base + 1 + w, base + n_round + 1, workers))
                if not ts:
                    continue
                seed = (getattr(master, "seed", 0) or 0) * 1_000_003 + rounds * 7919 + w
                jobs.append((w, seed, ts, tables))
            results = pool.map(_worker_round, jobs, chunksize=1)
            touched = 0
            if cpp_master:  # additive merge inside the core, no Python copy of the table
                for delta, n_touched in results:
                    master.add_tables(delta)
                    touched += n_touched
            else:
                nodes = master.nodes
                for delta, n_touched in results:
                    merge_into(nodes, delta)
                    touched += n_touched
            del tables, results
            master.nodes_touched += touched
            master.iteration = base + n_round
            done += n_round
            rounds += 1
            if log_every and (rounds % max(1, log_every) == 0 or done >= iterations):
                dt = time.perf_counter() - t0
                print(
                    f"  iter {master.iteration:>7,}  infosets {len(master.nodes):>8,}  "
                    f"nodes/s {master.nodes_touched / max(dt, 1e-9):>8,.0f}  elapsed {dt:6.0f}s  [{workers} workers]",
                    flush=True,
                )
    return master
