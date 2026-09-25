"""Benchmark the MCCFR backends and compare exploitability curves (docs/backends.md).

    python scripts/bench_backends.py                 # everything (python serial takes a few minutes)
    python scripts/bench_backends.py --skip-python   # only multiprocess + cpp
    python scripts/bench_backends.py --workers 8 --threads 16 --iters 30000

Two measurements:

1. nodes/s and wall time on GameSpec(n_players=3, stack_bb=15, max_street=FLOP) for --iters
   iterations: python (1 core), multiprocess (K workers, python traversal), cpp (T threads).
2. push/fold 10bb: exact exploitability (bb/100) at several iteration counts for the serial
   python trainer, the multiprocess driver and the cpp trainer (1 thread = bit-identical to
   python, T threads = statistically equivalent).  Needs data/class_equity.json.
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

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def flop_bucketer(spec: GameSpec) -> EquityBucketer:
    p = os.path.join(DATA, f"buckets_{spec.n_players}p_{spec.stack_bb}bb_flop.json")
    if os.path.exists(p):
        return EquityBucketer.load(p)
    print("fitting buckets (once)...", flush=True)
    bk = spec.make_bucketer().fit(n_situations=1200, seed=0)
    os.makedirs(DATA, exist_ok=True)
    bk.save(p)
    return bk


def bench_flop(args) -> None:
    spec = GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    bk = flop_bucketer(spec)
    print(f"\n== throughput: {spec.describe()}, {args.iters:,} iterations ==")
    rows = []

    def run(label, make, train):
        tr = make()
        t0 = time.perf_counter()
        train(tr)
        dt = time.perf_counter() - t0
        rows.append((label, tr.nodes_touched, dt, len(tr.nodes)))
        print(f"  {label:<28} {dt:8.2f}s  {tr.nodes_touched:>12,} nodes  {tr.nodes_touched / dt:>12,.0f} nodes/s  {len(tr.nodes):>8,} infosets", flush=True)

    if not args.skip_python:
        run("python (1 core)", lambda: MCCFRTrainer(spec, bk, seed=0, backend="python"), lambda t: t.train(args.iters))
    for k in args.workers:
        run(f"multiprocess ({k} workers)", lambda: MCCFRTrainer(spec, bk, seed=0, backend="python"),
            lambda t, k=k: t.train_parallel(args.iters, workers=k, sync_every=args.sync_every))
    if fast.core_available():
        for th in args.threads:
            run(f"cpp ({th} threads)", lambda th=th: MCCFRTrainer(spec, bk, seed=0, backend="cpp", threads=th), lambda t: t.train(args.iters))
    else:
        print("  cpp: core not built (python scripts/build_fast.py)")
    print("\n| backend | wall time | nodes/s | infosets |\n|---|---:|---:|---:|")
    for label, nodes, dt, n in rows:
        print(f"| {label} | {dt:.2f} s | {nodes / dt:,.0f} | {n:,} |")


def bench_pushfold(args) -> None:
    from negpluribus.cfr.exploit import ClassEquity, exact_exploitability

    path = os.path.join(DATA, "class_equity.json")
    if not os.path.exists(path):
        print("\n(pushfold curves skipped: data/class_equity.json missing; scripts/exploitability.py pushfold builds it)")
        return
    equity = ClassEquity.load(path)
    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(), forbid_open_limp=True)
    checkpoints = [c for c in (1000, 5000, 10000, 30000, 100000, 300000) if c <= args.pf_iters]
    print(f"\n== push/fold 10bb: exploitability (bb/100) vs iterations ==")
    variants = []
    if not args.skip_python:
        variants.append(("python", lambda: MCCFRTrainer(spec, seed=0, backend="python"), lambda t, n: t.train(n)))
    k = args.workers[0] if args.workers else 4
    variants.append((f"multiprocess x{k}", lambda: MCCFRTrainer(spec, seed=0, backend="python"),
                     lambda t, n: t.train_parallel(n, workers=k, sync_every=args.sync_every)))
    if fast.core_available():
        variants.append(("cpp x1 (== python)", lambda: MCCFRTrainer(spec, seed=0, backend="cpp", threads=1), lambda t, n: t.train(n)))
        th = args.threads[-1] if args.threads else fast.default_threads()
        variants.append((f"cpp x{th}", lambda th=th: MCCFRTrainer(spec, seed=0, backend="cpp", threads=th), lambda t, n: t.train(n)))
    header = "| iterations | " + " | ".join(v[0] for v in variants) + " |"
    print(header)
    print("|---:|" + "---:|" * len(variants))
    trainers = [(v[0], v[1](), v[2]) for v in variants]
    times = {name: 0.0 for name, _, _ in trainers}
    for c in checkpoints:
        cells = []
        for name, tr, train in trainers:
            t0 = time.perf_counter()
            train(tr, c - tr.iteration)
            times[name] += time.perf_counter() - t0
            cells.append(f"{exact_exploitability(spec, tr.strategy(), equity).bb100:.2f}")
        print(f"| {c:,} | " + " | ".join(cells) + " |", flush=True)
    print("| train time | " + " | ".join(f"{times[n]:.1f} s" for n, _, _ in trainers) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--pf-iters", type=int, default=100000)
    ap.add_argument("--workers", type=int, nargs="*", default=[8])
    ap.add_argument("--threads", type=int, nargs="*", default=[1, 8, 16])
    ap.add_argument("--sync-every", type=int, default=1000)
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--only", choices=["flop", "pushfold"], default=None)
    args = ap.parse_args()
    print("backends:", fast.describe())
    if args.only in (None, "flop"):
        bench_flop(args)
    if args.only in (None, "pushfold"):
        bench_pushfold(args)


if __name__ == "__main__":
    main()
