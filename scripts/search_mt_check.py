"""Multi-threaded search: two builds give equivalent results (within the noise between seeds).

    python scripts/search_mt_check.py --blueprint B.bin --buckets K.json --threads 4 --out mt_new.json
    python scripts/search_mt_check.py ... --compare mt_ref.json      # on the other build: the table of both

With several threads the order of the regret updates depends on the scheduler, so two runs differ even
with the same seed; the check is statistical.  Turn roots of the HU 200bb game (the spots of
scripts/search_quality.py), the agent's iterations (1M, x --scale), depth "pluribus", --seeds seeds per
hand.  Per search:
* the exact exploitability of the subgame, bb per deal (SubgameSearch.subgame_exploitability, the best
  responder deviates on the turn and the river), average and final-iteration profiles;
* the clear-deviation mass: the probability the search's average strategy for our actual hole puts on
  actions the blueprint plays with less than --clear probability (scripts/search_deviation_report.py
  counts the same thing per played decision);
* iterations per second.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.abstraction.infoset import infoset_key  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

core = fast.core()
FLOP_LINE = ["r1", "c", "c", "r0.5", "c"]
SPOTS = {"turn, BB first to act": FLOP_LINE, "turn, BB faces a pot bet": FLOP_LINE + ["c", "r1"]}
METRICS = ("expl_avg", "expl_final", "dev_mass", "it_s")


def summary(xs):
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="x the agent's 1M turn iterations")
    ap.add_argument("--hands", type=int, default=3, help="hands per spot")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7, help="the deals")
    ap.add_argument("--clear", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                    postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets,
                    bucket_kind=getattr(bk, "kind", "ehs"))
    bp = load_blueprint(args.blueprint)
    game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000)), bp.lookup)
    n_it = max(1, int(1_000_000 * args.scale))
    print(f"{args.threads} threads, {n_it:,} iterations, module {core.__file__}", flush=True)
    rng = random.Random(args.seed)
    rows = []
    for spot, line in SPOTS.items():
        for h in range(args.hands):
            order = list(range(52))
            rng.shuffle(order)
            st = spec.new_hand(order, button=0)
            acts = []
            for name in line:
                obs = st.observe(st.current_player)
                a = spec.grid.to_concrete(obs, name)
                acts.append((int(a.type), int(a.amount)))
                st.apply(a)
            obs = st.observe(st.current_player)
            legal = spec.grid.abstract_actions(obs)
            bp_probs = dict(zip(legal, bp.policy(infoset_key(obs, bk, spec.grid), legal)))
            for seed in range(1, args.seeds + 1):
                s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                       iterations=n_it, time_budget=0.0, threads=args.threads, seed=seed, depth="pluribus")
                r = s.solve()
                dev = sum(p for name, p in zip(r["actions"], r["average"]) if bp_probs.get(name, 0.0) < args.clear)
                row = {"hand": f"{spot} #{h}", "seed": seed, "expl_avg": s.subgame_exploitability(0, 2)[0],
                       "expl_final": s.subgame_exploitability(1, 2)[0], "dev_mass": dev,
                       "it_s": r["iterations"] / max(r["seconds"], 1e-9)}
                rows.append(row)
                print(f"{row['hand']:30s} seed {seed}: exploitability avg {row['expl_avg']:7.4f} final {row['expl_final']:7.4f}  "
                      f"deviation mass {dev:5.3f}  {row['it_s']:>10,.0f} it/s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f)
    print("\nmean +/- std over hands x seeds:")
    for m in METRICS:
        mu, sd = summary([r[m] for r in rows])
        print(f"  {m:10s} {mu:12.4f} +/- {sd:.4f}")
    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        print(f"\nthis build against {args.compare} (same hands and seeds; differences are thread-order noise):")
        print(f"  {'metric':10s} {'this':>10s} {'other':>10s} {'mean diff':>10s} {'+/- 95%':>9s}   seed noise inside a build (std)")
        for m in METRICS:
            a = [r[m] for r in rows]
            b = [r[m] for r in ref]
            d = [x - y for x, y in zip(a, b)]
            mu, sd = summary(d)
            # noise between seeds of the same hand, within this build
            per_hand = {}
            for r in rows:
                per_hand.setdefault(r["hand"], []).append(r[m])
            within = statistics.fmean(statistics.stdev(v) for v in per_hand.values() if len(v) > 1) if args.seeds > 1 else 0.0
            print(f"  {m:10s} {statistics.fmean(a):10.4f} {statistics.fmean(b):10.4f} {mu:10.4f} {1.96 * sd / len(d) ** 0.5:9.4f}   {within:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
