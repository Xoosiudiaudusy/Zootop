"""See the real-time search change a blueprint decision, and measure it.

    python scripts/search_demo.py                 # spots + continuation strategies + ranges
    python scripts/search_demo.py --eval-deals 150 --villain tag   # paired gain of search over blueprint

Needs data/blueprint_2p_20bb_flop.json (run scripts/train_blueprint.py first).

Sections:
  1. one flop spot, several hero hands: blueprint probabilities vs search probabilities
     (search averaged over --seeds runs so you can see the seed-to-seed noise)
  2. the four continuation strategies at one leaf information set
  3. the villain's range at the root as the search sees it (top particles)
  4. optional: SearchAgent vs BlueprintAgent on identical deals (gain in bb/100 +/- CI)
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.agents.search import SearchAgent  # noqa: E402
from negpluribus.cards import cards_from_str, cards_to_str, hole_class  # noqa: E402
from negpluribus.cfr import ContinuationPolicy, GameSpec, SearchConfig, SubgameSolver  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint, tagged_path  # noqa: E402
from negpluribus.cfr.search import CONTINUATIONS, RangeSampler  # noqa: E402
from negpluribus.engine import CALL, Street, raise_to  # noqa: E402
from negpluribus.eval import compare_heroes  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def spot(spec, villain, hero, board, line):
    order = cards_from_str(villain) + cards_from_str(hero) + cards_from_str(board)
    order += [c for c in range(52) if c not in order]
    h = spec.new_hand(order, button=0)  # seat0 = BTN/SB (villain), seat1 = BB (hero)
    for a in line:
        h.apply(a)
    return h


def fmt(legal, probs):
    return "  ".join(f"{a}:{p:.2f}" for a, p in zip(legal, probs)) if probs else "unknown to blueprint"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="2p_20bb_flop")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=2, help="-1 = solve to the end of the hand (no leaves)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-deals", type=int, default=0)
    ap.add_argument("--eval-iters", type=int, default=600)
    ap.add_argument("--villain", default="tag")
    args = ap.parse_args()

    bk = load_bucketer(os.path.join(DATA, f"buckets_{args.tag}.json"))  # E[HS] or potential-aware, by the file's kind
    spec = GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
    bp = load_blueprint(tagged_path(DATA, "blueprint", args.tag), backend="python")  # .bin or .json, newest
    cfg = SearchConfig(iterations=args.iters, depth=None if args.depth < 0 else args.depth, n_particles=200)
    print("game:", spec.describe())
    print("search:", cfg)

    if args.eval_deals:
        section(f"4. SearchAgent vs BlueprintAgent on identical deals, villain = {args.villain}")
        ecfg = SearchConfig(iterations=args.eval_iters, depth=cfg.depth, n_particles=200)
        a = SearchAgent(spec, bp, bk, ecfg, seed=1, name="search")
        b = BlueprintAgent(bp, bk, spec.grid, seed=1, name="blueprint")
        vil = [make_agent(args.villain, seed=10, label=args.villain)]
        t = time.perf_counter()
        ra, rb, gain, ci = compare_heroes(
            a, b, vil, n_deals=args.eval_deals, seed=21,
            sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street,
        )
        print(ra)
        print(rb)
        print(f"GAIN of search over blueprint: {gain:+.2f} bb/100 (95% CI +/-{ci:.2f})  "
              f"[{'significant' if abs(gain) > ci else 'not significant'}]")
        print(f"searches: {a.n_searches}, search fallbacks: {a.n_search_fallback}, {time.perf_counter() - t:.0f}s")
        return

    section("1. Flop spot: SB opened (r1 = 3.5bb), BB called; flop Kh 9h 2c; BB (hero) acts first, pot 7bb")
    hands = [("Kc Ks", "top set"), ("Ah Qh", "nut flush draw + overcards"), ("9s 8s", "middle pair"),
             ("7c 7d", "underpair"), ("6d 5d", "air")]
    for hero, label in hands:
        villain = "Ad 3d" if not set(cards_from_str(hero)) & set(cards_from_str("Ad 3d")) else "Qd 3d"
        h = spot(spec, villain, hero, "Kh 9h 2c", [raise_to(350), CALL])
        obs = h.observe()
        legal = spec.grid.abstract_actions(obs)
        runs = []
        t = time.perf_counter()
        for seed in range(args.seeds):
            solver = SubgameSolver(spec, bp, bk, cfg, seed=seed)
            runs.append(solver.solve(obs))
        dt = (time.perf_counter() - t) / args.seeds
        key = solver.last_root_key
        print(f"\n  {hero} ({label})   key = {key}")
        print(f"    blueprint : {fmt(legal, bp.policy(key, legal))}")
        mean = [statistics.mean(r[j] for r in runs) for j in range(len(legal))]
        sd = [statistics.pstdev(r[j] for r in runs) for j in range(len(legal))]
        print(f"    search    : " + "  ".join(f"{a}:{m:.2f}+-{s:.2f}" for a, m, s in zip(legal, mean, sd))
              + f"   ({dt:.1f}s per search, {args.seeds} seeds)")

    section("2. The four continuation strategies at one leaf (what players may do beyond the depth limit)")
    cont = ContinuationPolicy(bp, bias_factor=cfg.bias_factor)
    h = spot(spec, "Ad 3d", "9s 8s", "Kh 9h 2c", [raise_to(350), CALL, raise_to(350)])  # BB bet 0.5 pot, SB to act
    obs = h.observe()
    legal = spec.grid.abstract_actions(obs)
    solver = SubgameSolver(spec, bp, bk, cfg, seed=0)
    key = solver._key(h, obs, None)
    print(f"  SB facing a half-pot bet, key = {key}")
    for kind in CONTINUATIONS:
        print(f"    {kind:>5}: {fmt(legal, cont.probs(kind, obs.seat, key, legal))}")
    print("  In the search each player *chooses* one of these at the leaf, by regret matching, so the")
    print("  opponent ends up with the continuation that hurts us most.  ContinuationPolicy.bias_factors(seat)")
    print("  is where opponent stats will plug in: e.g. a station gets {'call': 20} instead of 5.")

    class StationAware(ContinuationPolicy):
        def bias_factors(self, seat):
            return {"fold": 1.0, "call": 20.0, "raise": 1.0} if seat == 0 else super().bias_factors(seat)

    sa = StationAware(bp, bias_factor=cfg.bias_factor)
    print(f"    example, station-aware 'call' continuation for seat 0: {fmt(legal, sa.probs('call', 0, key, legal))}")

    section("3. The villain's range at the root, as the search sees it")
    h = spot(spec, "Ad 3d", "Ah Qh", "Kh 9h 2c", [raise_to(350), CALL])
    obs = h.observe()
    rs = RangeSampler(bp, bk, spec.grid, SearchConfig(n_particles=2000))
    pts = rs.particles(obs, seat=0, rng=random.Random(0))
    freq = {}
    for hole, w in pts:
        freq[hole_class(*hole)] = freq.get(hole_class(*hole), 0) + w
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:15]
    total = sum(freq.values())
    print("  SB opened preflop; hands the blueprint opens with are weighted by P(open | hand):")
    print("  " + "  ".join(f"{c}:{w / total * 100:.1f}%" for c, w in top))
    print("  RangeSampler is the second hook for opponent stats: replace blueprint reach with a range")
    print("  built from the opponent's observed VPIP/PFR (the 'frequency -> range' bridge).")


if __name__ == "__main__":
    main()
