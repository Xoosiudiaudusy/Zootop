"""GPU trainer: check that the device gives the tables of the CPU trainers bit for bit, then measure it.

    python scripts/build_fast.py --clean          # with the CUDA Toolkit 12.8+ installed (sm_120 = RTX 50xx)
    python scripts/gpu_bench.py                   # checks + speed on HU 100bb (narrow grid, E[HS]-8)
    python scripts/gpu_bench.py --iters 2000000 --batches 4096,16384,32768 --cpu-threads 16

1. cuda_available(), device name.
2. Identity: push/fold 10bb (3000 iterations, batch 256) and HU 100bb (--check-iters, batch 4096): the GPU
   tables must equal the batched CPU reference (Trainer batch mode) exactly; any difference is a bug.
3. Speed on HU 100bb: GPU iterations/s per batch size (device time of traversal and sort+apply of the last
   batch, and the whole wall time incl. the CPU's deals/buckets), against the ordinary CPU trainer on
   --cpu-threads threads.
Bucket tables (NEGPLURIBUS_BUCKET_TABLES) make the CPU part (buckets of every deal) much faster; without
them E[HS] buckets are Monte-Carlo and the CPU part dominates.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import EquityBucketer  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.cfr.mccfr import MCCFRTrainer  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402


def flat(core, spec, bk, seed, batch, threads, gpu):
    ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), seed, True, threads)
    ft.batch_size = batch
    if gpu:
        ft.use_gpu(0)
    return ft


def table_flat(ft):
    return {k: (list(r), list(s), v) for k, (r, s, v) in ft.export_nodes().items()}


def table_ref(tr):
    return {k: (list(r), list(s), v) for k, (_, r, s, v) in tr._core.export_nodes().items()}


def check(core, name, spec, bk, seed, batch, iters, threads):
    t0 = time.time()
    ft = flat(core, spec, bk, seed, batch, threads, True)
    ft.train(iters)
    g = table_flat(ft)
    tr = MCCFRTrainer(spec, bucketer=bk, seed=seed, backend="cpp", threads=threads).set_batch(batch)
    tr.train(iters)
    r = table_ref(tr)
    diff = [k for k in r if g.get(k) != r[k]]
    extra = set(g) - set(r)
    ok = not diff and not extra
    print(f"  {name}: {iters} it, batch {batch}: infosets {len(r)}, differing {len(diff)}, extra {len(extra)} -> "
          f"{'IDENTICAL' if ok else 'MISMATCH'}  [{time.time() - t0:.1f}s]", flush=True)
    if diff:
        k = diff[0]
        print(f"    first difference {k}:\n      gpu {g.get(k)}\n      ref {r[k]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1_000_000, help="iterations per speed run")
    ap.add_argument("--check-iters", type=int, default=100_000)
    ap.add_argument("--batches", default="4096,16384,32768")
    ap.add_argument("--cpu-threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    core = fast.core()
    if core is None or not hasattr(core, "cuda_available"):
        print("C++ core missing or too old: python scripts/build_fast.py --clean")
        return 2
    ok, why = core.cuda_available()
    print(f"cuda_available: {ok} {why}")
    if not ok:
        return 2
    T = args.cpu_threads
    bk100 = EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    hu100 = GameSpec(n_players=2, stack_bb=100, max_street=Street.RIVER, preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0),
                     max_raises_per_street=2, n_buckets=8)
    ft = flat(core, hu100, bk100, 0, 4096, T, True)
    print(f"device: {ft.gpu_device}; HU100 flat game {ft.game_stats()}", flush=True)
    if not args.skip_check:
        print("identity (GPU vs the batched CPU reference):")
        pf = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
        good = check(core, "push/fold 10bb", pf, EquityBucketer(n_buckets=8), 4, 256, 3000, T)
        good &= check(core, "HU 100bb", hu100, bk100, 0, 4096, args.check_iters, T)
        if not good:
            print("GPU != CPU: stop here and send this output")
            return 1
    print(f"speed on HU 100bb, {args.iters} iterations:")
    t0 = time.time()
    MCCFRTrainer(hu100, bucketer=bk100, seed=0, backend="cpp", threads=T).train(args.iters)
    cpu = time.time() - t0
    print(f"  CPU trainer, {T} threads: {cpu:.1f}s = {args.iters / cpu:,.0f} it/s", flush=True)
    for B in [int(x) for x in args.batches.split(",")]:
        ft = flat(core, hu100, bk100, 0, B, T, True)
        ft.train(B)  # warm-up batch (allocations)
        t0 = time.time()
        ft.train(args.iters)
        wall = time.time() - t0
        s = ft.gpu_stats()
        print(f"  GPU batch {B}: {wall:.1f}s = {args.iters / wall:,.0f} it/s (x{cpu / wall:.2f} vs CPU); last batch: "
              f"{s['items']:,} items, {s['records']:,} records, device {s['ms_traverse']:.1f} ms traverse + {s['ms_apply']:.1f} ms apply "
              f"= {B / max(1e-9, (s['ms_traverse'] + s['ms_apply']) / 1000):,.0f} it/s device-only", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
