"""Time the C++ subgame search on a real blueprint at turn and river roots; stability across seeds.

    python scripts/search_timing.py --blueprint data/blueprint_hunl200w3_pot16_s0.json \
        --buckets data/buckets_hunl200w3_pot16_s0.json --threads 14 --seconds 2 --seeds 5

The game is the HU 200bb one of the Slumbot blueprint (grid preflop 0.5/1/3, postflop 0.5/1/2/4,
3 raises; the flags change it).  Spots: after "SB opens pot, BB calls; flop BB checks, SB bets half,
BB calls", the turn with BB first to act (a fresh round: nothing fixed), the turn with BB facing
SB's pot bet after checking (BB's check fixed for its hole), and the same two on the river after
the turn goes check-check.  For each spot and hand: the time the range update took, iterations
and iterations per second within the budget, the table size, then the final-iteration strategy
of our actual hole over several seeds (mean and max total variation between seeds).
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

core = fast.core()
FLOP_LINE = ["r1", "c", "c", "r0.5", "c"]
SPOTS = {
    "turn, BB first to act": FLOP_LINE,
    "turn, BB faces a pot bet": FLOP_LINE + ["c", "r1"],
    "river, BB first to act": FLOP_LINE + ["c", "c"],
    "river, BB faces a pot bet": FLOP_LINE + ["c", "c", "c", "r1"],
}


def fracs(s: str):
    return tuple(float(x) for x in s.split(",") if x.strip())


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--stack", type=int, default=200)
    ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hands", type=int, default=3, help="hands per spot")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    disable_power_throttling()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=args.stack, max_street=Street.RIVER, preflop_fracs=fracs(args.preflop_fracs),
                    postflop_fracs=fracs(args.postflop_fracs), max_raises_per_street=args.max_raises,
                    n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
    t = time.perf_counter()
    bp = load_blueprint(args.blueprint)
    print(f"game: {spec.describe()}\nblueprint: {len(bp):,} infosets, C++ lookup loaded in {time.perf_counter() - t:.1f}s; "
          f"{args.threads} threads, {args.seconds}s per search, {args.seeds} seeds")
    game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk), bp.lookup)
    rng = random.Random(args.seed)
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
            finals, avgs, rows = [], [], []
            for seed in range(args.seeds):
                t = time.perf_counter()
                s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                       iterations=0, time_budget=args.seconds, threads=args.threads, seed=seed)
                t_build = time.perf_counter() - t
                r = s.solve()
                finals.append(r["final"])
                avgs.append(r["average"])
                rows.append((t_build, s.root_info()["range_seconds"], r))
            r0 = rows[0][2]
            its = [row[2]["iterations"] for row in rows]
            secs = [row[2]["seconds"] for row in rows]
            pairs = list(itertools.combinations(range(args.seeds), 2))
            tv_f = [tv(finals[i], finals[j]) for i, j in pairs]
            tv_a = [tv(avgs[i], avgs[j]) for i, j in pairs]
            mean_f = [sum(f[k] for f in finals) / len(finals) for k in range(len(finals[0]))]
            print(f"\n[{spot}] hole {obs.hole} board {obs.board}, path {len(s.path())} actions, {r0['actions']}")
            print(f"   build {rows[0][0] * 1000:.0f} ms (ranges {rows[0][1] * 1000:.0f} ms) first seed, then "
                  f"{sum(row[0] for row in rows[1:]) / max(1, len(rows) - 1) * 1000:.0f} ms; "
                  f"iterations {min(its):,}-{max(its):,} in {min(secs):.2f}-{max(secs):.2f}s = "
                  f"{sum(its) / sum(secs):,.0f} it/s; table {r0['table_size']:,} nodes; nodes touched/it "
                  f"{r0['nodes_touched'] / max(1, r0['iterations']):.0f}")
            print("   final by seed: " + " | ".join(" ".join(f"{x:.2f}" for x in f) for f in finals))
            print(f"   final mean {' '.join(f'{x:.2f}' for x in mean_f)}; TV between seeds: final mean {sum(tv_f) / len(tv_f):.3f} "
                  f"max {max(tv_f):.3f}; average mean {sum(tv_a) / len(tv_a):.3f} max {max(tv_a):.3f}", flush=True)


if __name__ == "__main__":
    main()
