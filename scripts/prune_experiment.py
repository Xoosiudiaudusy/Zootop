"""Regret-based pruning (Pluribus) against no pruning, by the result: exact exploitability.

    python scripts/prune_experiment.py --game preflop-r1 --seeds 0,1,2 --prune-below 1e6,1e7,1e8

One-street preflop games (the exact best response exists there), so pruning is allowed on the
last street (Pluribus excludes it; here the only street).  Per seed and setting: epsilon (bb/100)
at the checkpoints, nodes touched, actions pruned, seconds.  The seed spread of the unpruned runs
is the noise floor.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.cfr import ClassEquity, GameSpec, MCCFRTrainer, exact_exploitability  # noqa: E402
from negpluribus.engine import Street  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["pushfold", "preflop-r1"], default="preflop-r1")
    ap.add_argument("--stack", type=int, default=10)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prune-below", default="1e6,1e7,1e8", help="thresholds (stored regret units) to compare with no pruning")
    ap.add_argument("--prob", type=float, default=0.95)
    ap.add_argument("--after", type=int, default=1000)
    ap.add_argument("--checkpoints", default="10000,30000,100000,300000")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    equity = ClassEquity.load_or_build(os.path.join(DATA, "class_equity.json"))
    if args.game == "pushfold":
        spec = GameSpec(n_players=2, stack_bb=args.stack, max_street=Street.PREFLOP, preflop_fracs=(), forbid_open_limp=True)
    else:
        spec = GameSpec(n_players=2, stack_bb=args.stack, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    cps = [int(c) for c in args.checkpoints.split(",")]
    settings = [0.0] + [float(x) for x in args.prune_below.split(",")]
    print(f"game {spec.describe()}; checkpoints {cps}; pruning prob {args.prob}, after {args.after:,}")
    for x in settings:
        for seed in [int(s) for s in args.seeds.split(",")]:
            tr = MCCFRTrainer(spec, seed=seed, backend="cpp", threads=args.threads)
            if x > 0:
                tr.set_pruning(x, args.prob, args.after, last_street=True)
            row, t0 = [], time.perf_counter()
            for c in cps:
                tr.train(c - tr.iteration)
                row.append(exact_exploitability(spec, tr.strategy(), equity).bb100)
            dt = time.perf_counter() - t0
            label = "off" if x == 0 else f"-{x:g}"
            eps = "  ".join(f"{e:7.3f}" for e in row)
            print(f"prune {label:>7} seed {seed}: eps {eps}   nodes {tr.nodes_touched:>12,}  pruned {tr.pruned_actions:>11,}  {dt:5.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
