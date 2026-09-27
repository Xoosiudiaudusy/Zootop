"""A saved blueprint against the archetype bots (and random), duplicate deals, bb/100 +/- 95% CI.

    python scripts/eval_archetypes.py --spec 2p_100bb_river --max-raises 2 \
        --blueprint data/blueprint_X.json --buckets data/buckets_X.json --deals 100000

    # the real-time search agent (C++ core) instead of the blueprint agent, against the blueprint itself
    python scripts/eval_archetypes.py --spec 2p_200bb_river --preflop-fracs 0.5,1.0,3.0 \
        --postflop-fracs 0.5,1.0,2.0,4.0 --blueprint data/blueprint_hunl200w3_pot16_s0.bin \
        --buckets data/buckets_hunl200w3_pot16_s0.json --cache data/bucketcache_hunl200w3_pot16_s0.bin \
        --agent search --search-budget 0.5 --opponents blueprint --deals 1000 --log data/duel.jsonl

Opponents: the archetypes, "random" (arbitrary chip amounts: tests action translation too),
"gridrandom" (uniform over the blueprint's own grid: no translation, tests the abstraction alone),
"blueprint" (the blueprint agent on --blueprint: a duel against the hero's own blueprint) and
"overbettor" (the blueprint, but its strong raises become off-grid overbets of --overbet-mult x the
largest grid size: the translation test of docs/scale_4street.md).  The villain line-up for N
players is N-1 copies.

--agent: "blueprint" (default) or "search" (negpluribus/agents/core_search.py: the blueprint preflop,
a C++ subgame search at every decision from the flop; --search-* flags).  Heads-up, the search hero
(or any hero with --luck) is also scored with the card-luck correction (chance nodes, exact equities;
negpluribus/eval/duel.py: the same decks and seeds as duplicate_match), and the hero's seconds per
hand are timed.  --log writes one JSON line per deal (raw and corrected results, seconds), so a long
match can be read while it runs; --progress prints every N deals.
The Python evaluation is the reference; the blueprint answers through its own abstraction,
exactly as at the table.  --blueprint takes either format (binary .bin or JSON); with the C++
core built the strategy is looked up in C++ (the same probabilities, a tenth of the memory;
NEGPLURIBUS_BLUEPRINT=python for the dict).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.agents.core_search import CoreSearchAgent, SearchResources, add_search_args, search_config_from_args  # noqa: E402
from negpluribus.agents.gridrandom import GridRandomAgent  # noqa: E402
from negpluribus.agents.overbettor import ValueOverbettor  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval import duplicate_match  # noqa: E402
from negpluribus.eval.duel import duplicate_duel  # noqa: E402
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
    ap.add_argument("--opponent-blueprint", default=None,
                    help="the blueprint opponent from another file (default: the hero's --blueprint)")
    ap.add_argument("--opponent-buckets", default=None, help="its bucketer (default: --buckets)")
    ap.add_argument("--tables", default=None,
                    help="bucket-table directory (scripts/build_bucket_table.py): the Python agents look their buckets up "
                         "there instead of computing them (exact features cost ~80 ms per flop bucket otherwise)")
    ap.add_argument("--deals", type=int, default=100000, help="duplicate deals per opponent (100k = 200k hands, +/- 4..10 bb/100 in HUNL 100bb)")
    ap.add_argument("--first-deal", type=int, default=0, help="start at this deal (continue a match on the same decks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", choices=("blueprint", "search"), default="blueprint", help="the hero")
    ap.add_argument("--overbet-mult", type=float, default=2.5, help="the overbettor's size: this x the largest grid size")
    ap.add_argument("--log", default=None, help="append one JSON line per deal (per opponent) to this file")
    ap.add_argument("--log-hands", default=None, help="append one JSON line per hand: cards, actions, the hero's net and "
                    "card luck, and each hero decision (for the search agent: searched or not, why, iterations)")
    ap.add_argument("--progress", type=int, default=0, help="print a line every N deals")
    ap.add_argument("--luck", action="store_true", help="heads-up card-luck correction for the blueprint hero too "
                    "(on by default for the search hero; costs about 0.1 s per deal for the exact preflop equity)")
    ap.add_argument("--no-luck", action="store_true", help="no card-luck correction")
    add_search_args(ap)
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
    res = None
    if args.agent == "search":
        res = SearchResources.load(spec, args.blueprint, bucketer=bk, cache_path=args.cache,
                                   presample_seed=args.seed if args.presample else None)
        bp = res.blueprint
        cfg = search_config_from_args(args)
        print(f"search agent: {cfg}")
        print(f"resources loaded in {res.load_seconds:.1f}s (bucket cache: {res.cache_loaded})")
    else:
        bp = load_blueprint(args.blueprint)
    print(f"game: {spec.describe()}")
    print(f"blueprint: {len(bp):,} infosets ({args.blueprint})")

    def for_play(bucketer):  # the bucketer a Python agent uses at the table
        if not args.tables:
            return bucketer
        from negpluribus.fast.tables import tabulated
        from negpluribus.fast.trainer import core_bucketer
        return tabulated(core_bucketer(bucketer), args.tables)

    bk_play = for_play(bk)
    obp, obk = bp, bk_play
    if args.opponent_blueprint:
        obp = load_blueprint(args.opponent_blueprint)
        print(f"opponent blueprint: {len(obp):,} infosets ({args.opponent_blueprint})")
    if args.opponent_buckets:
        obk = for_play(load_bucketer(args.opponent_buckets))
    for opp in args.opponents.split(","):
        t = time.perf_counter()
        if args.agent == "search":
            hero = CoreSearchAgent(res, cfg, seed=1, name="hero")
        else:
            hero = BlueprintAgent(bp, bk_play, spec.grid, seed=1, name="hero")
        if opp == "gridrandom":
            vils = [GridRandomAgent(spec.grid, name=f"{opp}{i}", seed=10 + i) for i in range(n - 1)]
        elif opp == "blueprint":
            vils = [BlueprintAgent(obp, obk, spec.grid, name=f"{opp}{i}", seed=10 + i) for i in range(n - 1)]
        elif opp == "overbettor":
            vils = [ValueOverbettor(bp, bk_play, spec.grid, mult=args.overbet_mult, name=f"{opp}{i}", seed=10 + i) for i in range(n - 1)]
        else:
            vils = [make_agent(opp, seed=10 + i, label=f"{opp}{i}") for i in range(n - 1)]
        luck = n == 2 and not args.no_luck and (args.agent == "search" or args.luck)
        if args.agent == "blueprint" and not args.log and not args.log_hands and not args.progress and not luck:
            res_m = duplicate_match(hero, vils, n_deals=args.deals, seed=args.seed, sb=spec.sb, bb=spec.bb,
                                    stack_bb=spec.stack_bb, max_street=spec.max_street)
            print(f"  vs {opp:>8}: {res_m.bb100:+8.1f} bb/100  (95% CI +/-{res_m.ci95:.1f}, {res_m.n_hands} hands, "
                  f"off-map {hero.fallback_rate:.1%}, {time.perf_counter() - t:.0f}s)", flush=True)
            if opp == "overbettor":
                print(f"             overbets made: {sum(v.n_overbets for v in vils):,}", flush=True)
            continue
        log = open(args.log, "a", encoding="utf-8") if args.log else None

        def on_deal(d, r, recs, opp=opp, t0=t):
            if log is not None:
                row = {"opponent": opp, "agent": args.agent, "deal": d, "raw": round(r.raw[-1], 4),
                       "hero_s": round(sum(r.hero_seconds[-len(recs):]), 3), "hand_s": round(sum(r.hand_seconds[-len(recs):]), 3),
                       "events": [len(x.events) for x in recs]}
                if r.corrected:
                    row["corrected"] = round(r.corrected[-1], 4)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if args.progress and r.n_deals % args.progress == 0:
                extra = f"; searches {hero.stats.summary()}" if args.agent == "search" else ""
                print(f"    {opp} deal {r.n_deals}: {r.line()}, {time.perf_counter() - t0:.0f}s{extra}", flush=True)

        hand_log = open(args.log_hands, "a", encoding="utf-8") if args.log_hands else None

        def on_hand(d, seat, rec, luck_bb, infos, opp=opp):
            row = {"opponent": opp, "agent": args.agent, "deal": d, "hero_seat": seat, "button": rec.button,
                   "holes": rec.hole_cards, "board": rec.board,
                   "events": [[int(e.street), e.seat, int(e.action.type), int(e.action.amount)] for e in rec.events],
                   "net_bb": rec.net[seat] / spec.bb, "luck_bb": None if luck_bb is None else round(luck_bb, 4),
                   "hero": [{k: v for k, v in i.items() if k in ("street", "s", "action", "played", "reason", "iterations",
                                                                   "inserted", "off_map")} for i in infos]}
            hand_log.write(json.dumps(row) + "\n")
            hand_log.flush()

        r = duplicate_duel(hero, vils, n_deals=args.deals, seed=args.seed, sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb,
                           max_street=spec.max_street, luck=luck, first_deal=args.first_deal, on_deal=on_deal,
                           on_hand=on_hand if hand_log is not None else None)
        if log is not None:
            log.close()
        if hand_log is not None:
            hand_log.close()
        print(f"  vs {opp:>8}: {r.line()}, off-map {hero.fallback_rate:.1%}, {time.perf_counter() - t:.0f}s", flush=True)
        if args.agent == "search":
            print(f"             {hero.stats.summary()}", flush=True)
        if opp == "overbettor":
            print(f"             overbets made: {sum(v.n_overbets for v in vils):,}", flush=True)


if __name__ == "__main__":
    main()
