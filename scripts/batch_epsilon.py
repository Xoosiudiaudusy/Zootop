"""Batched (synchronous, delayed-feedback) MCCFR against the ordinary trainer, by exact exploitability.

    python scripts/batch_epsilon.py --game pushfold --batches 0,4096,16384,65536 --seeds 0,1,2
    python scripts/batch_epsilon.py --game preflop20 --checkpoints 262144,1048576,4194304,16777216,67108864

Preflop-only 2-player games, where the exact best response exists (cfr/exploit.py).  Batch 0 = the ordinary
trainer; B > 0 = Trainer batch mode (the CPU reference that the GPU trainer equals bit for bit).  Per (batch,
seed): epsilon (bb/100) at every checkpoint.  At the end: mean and spread over seeds per (batch, checkpoint),
and the number of batches (checkpoint / B).  The question: from how many batches per run is batched training
no worse than the ordinary trainer beyond the seed spread.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.cfr import ClassEquity, GameSpec, MCCFRTrainer, exact_exploitability  # noqa: E402
from negpluribus.engine import Street  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
GAMES = {
    "pushfold": lambda: GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(), forbid_open_limp=True),
    "preflop10": lambda: GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,)),
    "preflop20": lambda: GameSpec(n_players=2, stack_bb=20, max_street=Street.PREFLOP, preflop_fracs=(0.5, 1.0, 3.0)),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=list(GAMES), default="pushfold")
    ap.add_argument("--batches", default="0,4096,16384,65536")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--checkpoints", default="262144,1048576,4194304,16777216,67108864")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    spec = GAMES[args.game]()
    equity = ClassEquity.load_or_build(os.path.join(DATA, "class_equity.json"))
    cps = [int(c) for c in args.checkpoints.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"game {args.game}: {spec.describe()}; checkpoints {cps}; batches {batches}; seeds {seeds}", flush=True)
    eps = {}  # (batch, checkpoint) -> [eps per seed]
    for b in batches:
        for seed in seeds:
            tr = MCCFRTrainer(spec, seed=seed, backend="cpp", threads=args.threads)
            if b > 0:
                tr.set_batch(b)
            row, t0 = [], time.perf_counter()
            for c in cps:
                tr.train(c - tr.iteration)
                e = exact_exploitability(spec, tr.strategy(), equity).bb100
                eps.setdefault((b, c), []).append(e)
                row.append(e)
            label = "ordinary" if b == 0 else f"batch {b}"
            print(f"{label:>12} seed {seed}: eps " + "  ".join(f"{e:8.3f}" for e in row) + f"   {time.perf_counter() - t0:6.0f}s", flush=True)
    print("\nmean eps (bb/100) +/- sample std over seeds; [number of batches]")
    print(f"{'':>12} " + "  ".join(f"{c:>20,}" for c in cps))
    for b in batches:
        cells = []
        for c in cps:
            v = eps[(b, c)]
            m = statistics.mean(v)
            s = statistics.stdev(v) if len(v) > 1 else 0.0
            nb = "" if b == 0 else f" [{c // b}]"
            cells.append(f"{m:7.3f}+/-{s:5.3f}{nb:>7}")
        print(f"{('ordinary' if b == 0 else f'batch {b}'):>12} " + "  ".join(f"{x:>20}" for x in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
