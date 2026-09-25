"""A saved blueprint against the archetype bots (and random), duplicate deals, bb/100 +/- 95% CI.

    python scripts/eval_archetypes.py --spec 2p_100bb_river --max-raises 2 \
        --blueprint data/blueprint_X.json --buckets data/buckets_X.json --deals 100000

Opponents: the archetypes, "random" (arbitrary chip amounts: tests action translation too) and
"gridrandom" (uniform over the blueprint's own grid: no translation, tests the abstraction alone).
The villain line-up for N players is N-1 copies.
The Python evaluation is the reference; the blueprint answers through its own abstraction,
exactly as at the table.  --blueprint takes either format (binary .bin or JSON); with the C++
core built the strategy is looked up in C++ (the same probabilities, a tenth of the memory;
NEGPLURIBUS_BLUEPRINT=python for the dict).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.agents.gridrandom import GridRandomAgent  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval import duplicate_match  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402

STREETS = {"preflop": Street.PREFLOP, "flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help='"<players>p_<stack>bb_<street>", e.g. 2p_100bb_river')
    ap.add_argument("--preflop-fracs", default="1.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--opponents", default="random,gridrandom,tag,nit,maniac,station,lag")
    ap.add_argument("--deals", type=int, default=100000, help="duplicate deals per opponent (100k = 200k hands, +/- 4..10 bb/100 in HUNL 100bb)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    disable_power_throttling()  # scheduling only, results unchanged

    parts = args.spec.split("_")
    n, stack, street = int(parts[0].rstrip("p")), int(parts[1].rstrip("bb")), parts[2]
    bk = load_bucketer(args.buckets)
    spec = GameSpec(
        n_players=n, stack_bb=stack, max_street=STREETS[street],
        preflop_fracs=tuple(float(x) for x in args.preflop_fracs.split(",") if x.strip()),
        postflop_fracs=tuple(float(x) for x in args.postflop_fracs.split(",") if x.strip()),
        max_raises_per_street=args.max_raises, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"),
    )
    bp = load_blueprint(args.blueprint)
    print(f"game: {spec.describe()}")
    print(f"blueprint: {len(bp):,} infosets ({args.blueprint})")
    for opp in args.opponents.split(","):
        t = time.perf_counter()
        hero = BlueprintAgent(bp, bk, spec.grid, seed=1, name="hero")
        if opp == "gridrandom":
            vils = [GridRandomAgent(spec.grid, name=f"{opp}{i}", seed=10 + i) for i in range(n - 1)]
        else:
            vils = [make_agent(opp, seed=10 + i, label=f"{opp}{i}") for i in range(n - 1)]
        res = duplicate_match(hero, vils, n_deals=args.deals, seed=args.seed, sb=spec.sb, bb=spec.bb,
                              stack_bb=spec.stack_bb, max_street=spec.max_street)
        print(f"  vs {opp:>8}: {res.bb100:+8.1f} bb/100  (95% CI +/-{res.ci95:.1f}, {res.n_hands} hands, "
              f"off-map {hero.fallback_rate:.1%}, {time.perf_counter() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
