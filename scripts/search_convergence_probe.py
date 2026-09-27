"""Replay the search at the decisions where it left the blueprint's line, with more iterations:
does the deviating action's probability fall as the search converges (under-convergence), or
stay (a different solution)?  From a search agent's duel hand log, the flop decisions where the
search bet/raised while the blueprint gives check/call at least --bp-min are replayed as the agent
searched them (same subgame, no overrides at a flop root), for each iteration count, and the
average and final strategies are printed next to the blueprint's row.

    python scripts/search_convergence_probe.py --hands SP/p3/hands_duel2_bp.jsonl --blueprint ... --buckets ... \
        --cache ... --iterations 350000,1400000,5600000 --max-decisions 20 --threads 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.abstraction.infoset import infoset_key  # noqa: E402
from negpluribus.agents.core_search import SearchResources  # noqa: E402
from negpluribus.cards import card_to_str as card_str  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.engine import Action, ActionType, Street  # noqa: E402


def kind(name):
    return "fold" if name.startswith("f") else ("check/call" if name == "c" else "bet/raise")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--iterations", default="350000,1400000,5600000")
    ap.add_argument("--seeds", default="1,2", help="seeds at the first iteration count (search noise)")
    ap.add_argument("--max-decisions", type=int, default=20)
    ap.add_argument("--bp-min", type=float, default=0.95, help="blueprint check/call probability at least this")
    ap.add_argument("--street", type=int, default=1, help="1 flop (no range overrides needed)")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--stack", type=int, default=200)
    ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    args = ap.parse_args()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=args.stack, max_street=Street.RIVER,
                    preflop_fracs=tuple(float(x) for x in args.preflop_fracs.split(",")),
                    postflop_fracs=tuple(float(x) for x in args.postflop_fracs.split(",")),
                    max_raises_per_street=args.max_raises, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
    res = SearchResources.load(spec, args.blueprint, bucketer=bk, cache_path=args.cache)
    bp, grid, core = res.blueprint, spec.grid, fast.core()
    iters = [int(x) for x in args.iterations.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    found = 0
    agg = {}  # (K, seed) -> [sum of the deviating action's average prob, sum of its final prob, n]
    for line in open(args.hands, encoding="utf-8"):
        if found >= args.max_decisions:
            break
        h = json.loads(line)
        seat = h["hero_seat"]
        holes, board = h["holes"], h["board"]
        used = set(holes[0]) | set(holes[1]) | set(board)
        order = list(holes[0]) + list(holes[1]) + list(board) + [c for c in range(52) if c not in used]
        st = spec.new_hand(order, button=h["button"])
        hit = None
        for street, actor, t, amount in h["events"]:
            if actor == seat and street == args.street:
                obs = st.observe(seat)
                legal = grid.abstract_actions(obs)
                probs = bp.policy(infoset_key(obs, bk, grid), legal)
                taken = Action(ActionType(t), amount)
                name = next((nm for nm in legal if grid.to_concrete(obs, nm) == taken), None)
                if probs is not None and name is not None and kind(name) == "bet/raise" and "c" in legal \
                        and probs[legal.index("c")] >= args.bp_min:
                    hit = (obs, legal, probs, name, taken)
                    break
            if actor == seat and street > args.street:
                break
            st.apply(Action(ActionType(t), amount))
        if hit is None:
            continue
        found += 1
        obs, legal, probs, name, taken = hit
        actions = [(int(e.action.type), int(e.action.amount) if e.action.type == ActionType.RAISE else 0) for e in obs.events]
        stacks = list(obs.starting_stacks) if obs.starting_stacks else [spec.stack_bb * spec.bb] * obs.n_players
        pos = "SB" if seat == h["button"] else "BB"
        path = " ".join(f"{'H' if e.seat == seat else 'V'}:{e.action}" for e in obs.events if int(e.street) == args.street)
        print(f"\n#{found} deal {h['deal']} {pos} {card_str(obs.hole[0])}{card_str(obs.hole[1])} board {' '.join(card_str(c) for c in obs.board)} "
              f"| flop so far: {path or '-'} | search took {taken} = {name}")
        print("   blueprint: " + "  ".join(f"{nm} {p:.2f}" for nm, p in zip(legal, probs)))
        for K in iters:
            for sd in (seeds if K == iters[0] else seeds[:1]):
                t0 = time.perf_counter()
                s = core.SubgameSearch(res.game, stacks, obs.button, actions, list(obs.board), seat, list(obs.hole),
                                       iterations=K, time_budget=0.0, threads=args.threads, seed=sd, focus=0.5, min_prob=1e-3,
                                       linear=True, overrides=None, depth="pluribus", rollouts=3, bias=5.0)
                r = s.solve()
                names = [("f" if tt == int(ActionType.FOLD) else "c" if tt == int(ActionType.CALL) else f"r{am}")
                         for tt, am in zip(r["types"], r["amounts"])]
                idx = next((i for i, (tt, am) in enumerate(zip(r["types"], r["amounts"]))
                            if tt == int(taken.type) and (tt != int(ActionType.RAISE) or am == int(taken.amount))), None)
                avg, fin = r["average"], r["final"]
                pa = avg[idx] if idx is not None else float("nan")
                pf = fin[idx] if idx is not None else float("nan")
                agg.setdefault((K, sd), [0.0, 0.0, 0])
                agg[(K, sd)][0] += pa
                agg[(K, sd)][1] += pf
                agg[(K, sd)][2] += 1
                bet_avg = sum(p for nm, p in zip(names, avg) if nm.startswith("r"))
                print(f"   {K:>9,} it seed {sd}: taken action avg {pa:.2f} final {pf:.2f}; all bets avg {bet_avg:.2f}; "
                      + "  ".join(f"{nm} {p:.2f}" for nm, p in zip(names, avg)) + f"  ({time.perf_counter() - t0:.0f}s)")
    print("\nmean probability of the taken (deviating) action over the replayed decisions:")
    for (K, sd), (sa, sf, n) in sorted(agg.items()):
        print(f"   {K:>9,} it seed {sd}: average {sa / n:.3f}  final {sf / n:.3f}  (n={n})")


if __name__ == "__main__":
    main()
