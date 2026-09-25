"""Paired duplicate match between two saved blueprints (checkpoints, seeds, abstractions).

    python scripts/compare_checkpoints.py --spec 2p_100bb_river --a data/blueprint_X.it2000000.json --b data/blueprint_X.it1000000.json --deals 5000

Plays A (hero) against B in every seat on identical deals and the reverse, reports bb/100 with
95% CI.  Uses: (1) convergence in multi-street games, where no exact epsilon exists: a later
checkpoint should stop beating an earlier one as training plateaus; (2) abstraction or seed
comparisons — ALWAYS pair with a control (same abstraction, other seed) before concluding.

--spec is "<players>p_<stack>bb_<street>[_<extra>]" or explicit --players/--stack/--street flags;
the bucketer is loaded by kind from --buckets (default data/buckets_<spec>.json).  --a / --b take
either format (binary .bin or JSON), looked up in C++ when the core is built (same probabilities).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval import duplicate_match  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
STREETS = {"preflop": Street.PREFLOP, "flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}


def parse_spec(text: str):
    parts = text.split("_")
    return int(parts[0].rstrip("p")), int(parts[1].rstrip("bb")), parts[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help='e.g. "2p_100bb_river"')
    ap.add_argument("--players", type=int, default=None)
    ap.add_argument("--stack", type=int, default=None)
    ap.add_argument("--street", default=None)
    ap.add_argument("--preflop-fracs", default="1.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--buckets", default=None, help="bucketer JSON for both sides (default data/buckets_<spec>.json)")
    ap.add_argument("--buckets-a", default=None, help="bucketer JSON of A (when A and B use different abstractions)")
    ap.add_argument("--buckets-b", default=None, help="bucketer JSON of B")
    ap.add_argument("--vs-random", type=int, default=0, help="also play A against the random agent for this many deals (Nash sanity: must not lose)")
    ap.add_argument("--a", required=True, help="blueprint of A (.bin or .json)")
    ap.add_argument("--b", required=True, help="blueprint of B (.bin or .json)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    disable_power_throttling()  # scheduling only, results unchanged

    if args.spec:
        n, stack, street = parse_spec(args.spec)
    else:
        n, stack, street = args.players, args.stack, args.street
    bk_path = args.buckets or os.path.join(DATA, f"buckets_{args.spec}.json")
    bk_a = load_bucketer(args.buckets_a or bk_path)
    bk_b = load_bucketer(args.buckets_b or bk_path)
    spec = GameSpec(
        n_players=n, stack_bb=stack, max_street=STREETS[street],
        preflop_fracs=tuple(float(x) for x in args.preflop_fracs.split(",") if x.strip()),
        postflop_fracs=tuple(float(x) for x in args.postflop_fracs.split(",") if x.strip()),
        max_raises_per_street=args.max_raises, n_buckets=bk_a.n_buckets, bucket_kind=getattr(bk_a, "kind", "ehs"),
    )
    A = load_blueprint(args.a)
    B = load_blueprint(args.b)
    print("game:", spec.describe())
    print(f"{args.label_a}: {len(A):,} infosets ({args.a})\n{args.label_b}: {len(B):,} infosets ({args.b})")

    def play(hero_s, vil_s, hero_label, vil_label, bk_hero, bk_vil):
        t = time.perf_counter()
        hero = BlueprintAgent(hero_s, bk_hero, spec.grid, seed=1, name="hero")
        vils = [BlueprintAgent(vil_s, bk_vil, spec.grid, seed=7 + i, name=f"villain{i}") for i in range(n - 1)]
        res = duplicate_match(hero, vils, n_deals=args.deals, seed=args.seed, sb=spec.sb, bb=spec.bb,
                              stack_bb=spec.stack_bb, max_street=spec.max_street)
        print(f"  {hero_label} vs {vil_label}: {res.bb100:+7.2f} bb/100  (95% CI +/-{res.ci95:.2f}, {res.n_hands} hands, "
              f"off-map {hero.fallback_rate:.1%}, {time.perf_counter() - t:.0f}s)")
        return res

    play(A, B, args.label_a, args.label_b, bk_a, bk_b)
    play(B, A, args.label_b, args.label_a, bk_b, bk_a)
    if args.vs_random:
        from negpluribus.agents import make_agent

        t = time.perf_counter()
        hero = BlueprintAgent(A, bk_a, spec.grid, seed=1, name="hero")
        vils = [make_agent("random", seed=5 + i, label=f"random{i}") for i in range(n - 1)]
        res = duplicate_match(hero, vils, n_deals=args.vs_random, seed=args.seed + 11, sb=spec.sb, bb=spec.bb,
                              stack_bb=spec.stack_bb, max_street=spec.max_street)
        print(f"  {args.label_a} vs random: {res.bb100:+7.2f} bb/100  (95% CI +/-{res.ci95:.2f}, {res.n_hands} hands, "
              f"off-map {hero.fallback_rate:.1%}, {time.perf_counter() - t:.0f}s)")


if __name__ == "__main__":
    main()
