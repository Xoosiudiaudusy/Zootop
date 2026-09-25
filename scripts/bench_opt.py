"""Throughput of the C++ MCCFR trainer on one fixed window (docs: comms from-optimizer).

    python scripts/bench_opt.py --buckets data/buckets_<tag>.json [--grid wide] [--threads 1,2,4,8,16]

HU 100bb, four streets.  For every thread count: warm-up iterations, then a measured window;
prints iterations/s, nodes/s and nodes per iteration (equal nodes/iteration = the same work, so
compare runs on nodes/s).  Bucket tables are used when NEGPLURIBUS_BUCKET_TABLES points at a
directory holding the table of --buckets (scripts/build_bucket_table.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.cfr.mccfr import MCCFRTrainer  # noqa: E402
from negpluribus.engine import Street  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--grid", choices=["narrow", "wide"], default="wide")
    ap.add_argument("--threads", default="1,2,4,8,16")
    ap.add_argument("--warmup", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=200_000)
    args = ap.parse_args()
    bk = load_bucketer(args.buckets)
    kw = (dict(preflop_fracs=(1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3) if args.grid == "wide"
          else dict(preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0), max_raises_per_street=2))
    spec = GameSpec(n_players=2, stack_bb=100, max_street=Street.RIVER, n_buckets=bk.n_buckets,
                    bucket_kind="potential" if type(bk).__name__ == "PotentialAwareBucketer" else "ehs", **kw)
    for t in [int(x) for x in args.threads.split(",")]:
        tr = MCCFRTrainer(spec, bk, seed=0, backend="cpp", threads=t)
        tr.train(args.warmup)
        n0 = tr.nodes_touched
        t0 = time.time()
        tr.train(args.iters)
        dt = time.time() - t0
        nodes = tr.nodes_touched - n0
        print(f"{args.grid} threads={t:2d} table={type(tr._core_bucketer).__name__ == 'TabulatedBucketer'}: "
              f"{args.iters / dt:9,.0f} it/s  {nodes / dt / 1e6:6.2f} M nodes/s  {nodes / args.iters:.0f} nodes/it", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
