"""A saved blueprint against the archetype bots (and random), duplicate deals, bb/100 +/- 95% CI.

    python scripts/eval_archetypes.py --spec 2p_100bb_river --max-raises 2 \
        --blueprint data/blueprint_X.json --buckets data/buckets_X.json --deals 100000

    # the real-time search agent (C++ core) instead of the blueprint agent, against the blueprint itself
    python scripts/eval_archetypes.py --spec 2p_200bb_river --preflop-fracs 0.5,1.0,3.0 \
        --postflop-fracs 0.5,1.0,2.0,4.0 --blueprint data/blueprint_hunl200w3_pot16_s0.bin \
        --buckets data/buckets_hunl200w3_pot16_s0.json --cache data/bucketcache_hunl200w3_pot16_s0.bin \
        --agent search --search-budget 0.5 --opponents blueprint --deals 1000 --log data/duel.jsonl

    # the same, the subgame's turn and river on 64 exact buckets of its own (the blueprint keeps its 16)
    ... --agent search --search-buckets data/buckets_hunl200w3_pot64x_s0.json --search-tables data/bucket_tables

    # the depth grid (one blueprint per stack depth, picked per hand) at 50bb against the single 200bb blueprint
    python scripts/eval_archetypes.py --spec 2p_50bb_river --preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 \
        --agent grid --grid-manifest data/stack_grid_pot16_s0.json --tables data/bucket_tables --opponents blueprint \
        --blueprint data/blueprint_hunl200w3_pot16_s0.bin --buckets data/buckets_hunl200w3_pot16_s0.json --log-hands h.jsonl

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
match can be read while it runs; --progress prints every N deals.  A search that fails (e.g. out of memory
growing its node table) is played by the blueprint: the summary line counts them ("search errors N (fallback to
the blueprint)", next to "off the map"), and --log-hands marks such a decision played "blueprint after a search
error" with its error.
"grid" (negpluribus/agents/stack_grid.py): a StackGridAgent over the blueprint agents of the points of
--grid-manifest (one bucketer for all, on --tables; at most --grid-max-loaded points in memory), the point
picked per hand by the effective stack; --blueprint / --buckets are then only the opponents' (blueprint,
overbettor).  --log-hands writes the point and its blueprint with every hero decision, and with the first
one the effective stack and the review's alternatives (docs/stack_grid_design.md 6.4); every hand line has
the starting stacks ("stacks", chips).
The "blueprint" opponents count their decisions whose lookup gave the probability of raise sizes that are the
all-in at this stack to the all-in (a blueprint deeper than the table, H4 in docs/stack_grid_design.md 8): per
hand "opp_all_in" in --log-hands (only when non-zero), in total a line after the result.
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
from negpluribus.agents.stack_grid import StackGrid  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval import duplicate_match  # noqa: E402
from negpluribus.eval.duel import duplicate_duel  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402

STREETS = {"preflop": Street.PREFLOP, "flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}
# what --log-hands keeps of each hero decision (decision_info() of the agent plus the duel's street, seconds, action)
HERO_KEYS = ("street", "s", "action", "played", "reason", "iterations", "inserted", "off_map", "error",
             "point", "blueprint", "eff_bb", "eff_min_bb", "eff_median_bb", "point_min", "point_median")


def memory_line() -> str:
    """This process's memory (Windows: working set, its peak and private bytes; elsewhere the peak RSS)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = Counters()
            c.cb = ctypes.sizeof(c)
            psapi, kernel32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return "memory: unknown"
            mb = 1 << 20
            return (f"memory: working set {c.WorkingSetSize / mb:,.0f} MB (peak {c.PeakWorkingSetSize / mb:,.0f}), "
                    f"private {c.PagefileUsage / mb:,.0f} MB (peak {c.PeakPagefileUsage / mb:,.0f})")
        import resource

        return f"memory: peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:,.0f} MB"
    except Exception as e:  # a report line only
        return f"memory: unknown ({type(e).__name__})"


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
    ap.add_argument("--agent", choices=("blueprint", "search", "grid"), default="blueprint", help="the hero")
    ap.add_argument("--grid-manifest", default=None, help="--agent grid: the depth grid's manifest (data/stack_grid_*.json)")
    ap.add_argument("--grid-max-loaded", type=int, default=4, help="--agent grid: points kept in memory at once (LRU)")
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
                                   presample_seed=args.seed if args.presample else None,
                                   search_buckets_path=args.search_buckets, search_tables=args.search_tables,
                                   search_cache_path=args.search_cache)
        bp = res.blueprint
        cfg = search_config_from_args(args)
        print(f"search agent: {cfg}")
        print(f"resources loaded in {res.load_seconds:.1f}s (bucket cache: {res.cache_loaded})")
        if args.search_buckets:
            print(f"{res.describe_buckets()} ({args.search_buckets}; its cache: {res.search_cache_loaded})")
    else:
        bp = load_blueprint(args.blueprint)
    print(f"game: {spec.describe()}")
    what = "blueprint of the opponents" if args.agent == "grid" else "blueprint"
    print(f"{what}: {len(bp):,} infosets ({args.blueprint})")

    def for_play(bucketer):  # the bucketer a Python agent uses at the table
        if not args.tables:
            return bucketer
        from negpluribus.fast.tables import tabulated
        from negpluribus.fast.trainer import core_bucketer
        return tabulated(core_bucketer(bucketer), args.tables)

    bk_play = for_play(bk)
    grid_def = grid_make = None
    if args.agent == "grid":
        from negpluribus.fast.trainer import core_bucketer

        if not args.grid_manifest:
            ap.error("--agent grid needs --grid-manifest")
        grid_def = StackGrid.load(args.grid_manifest)
        if (grid_def.preflop_fracs, grid_def.postflop_fracs, grid_def.max_raises) != (spec.preflop_fracs, spec.postflop_fracs,
                                                                                      spec.max_raises_per_street):
            ap.error(f"the grid's bets {grid_def.preflop_fracs} / {grid_def.postflop_fracs} / {grid_def.max_raises} raises "
                     f"differ from this game's {spec.preflop_fracs} / {spec.postflop_fracs} / {spec.max_raises_per_street}")
        grid_bk = load_bucketer(grid_def.resolve(grid_def.bucketer))
        grid_make = grid_def.blueprint_factory(for_play(grid_bk), spec.grid, name="hero",
                                               bucketer_identity=core_bucketer(grid_bk, (0, 0, 0), tables="").identity)
        print(grid_def.describe())
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
        elif args.agent == "grid":
            hero = grid_def.agent(grid_make, max_loaded=args.grid_max_loaded, fallback_stack_bb=spec.stack_bb, seed=1,
                                  name="hero")
        else:
            hero = BlueprintAgent(bp, bk_play, spec.grid, seed=1, name="hero")
        if opp == "gridrandom":
            vils = [GridRandomAgent(spec.grid, name=f"{opp}{i}", seed=10 + i) for i in range(n - 1)]
        elif opp == "blueprint":  # counting the lookups that give collapsed raise sizes to the all-in (report, hand log)
            vils = [BlueprintAgent(obp, obk, spec.grid, name=f"{opp}{i}", seed=10 + i, count_all_in=True) for i in range(n - 1)]
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
            print_all_in(vils)
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
                extra += f"; {hero.summary()}" if args.agent == "grid" else ""
                print(f"    {opp} deal {r.n_deals}: {r.line()}, {time.perf_counter() - t0:.0f}s{extra}", flush=True)

        hand_log = open(args.log_hands, "a", encoding="utf-8") if args.log_hands else None
        all_in_seen = [0]  # the opponents' decisions so far whose lookup moved collapsed sizes to the all-in

        def on_hand(d, seat, rec, luck_bb, infos, opp=opp, vils=vils, all_in_seen=all_in_seen):
            row = {"opponent": opp, "agent": args.agent, "deal": d, "hero_seat": seat, "button": rec.button,
                   "stacks": list(rec.starting_stacks), "holes": rec.hole_cards, "board": rec.board,
                   "events": [[int(e.street), e.seat, int(e.action.type), int(e.action.amount)] for e in rec.events],
                   "net_bb": rec.net[seat] / spec.bb, "luck_bb": None if luck_bb is None else round(luck_bb, 4),
                   "hero": [{k: v for k, v in i.items() if k in HERO_KEYS} for i in infos]}
            moved = sum(getattr(v, "n_all_in", 0) for v in vils)
            if moved > all_in_seen[0]:  # only then: at the blueprints' own depth the logs keep their old bytes
                row["opp_all_in"] = moved - all_in_seen[0]
            all_in_seen[0] = moved
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
        if args.agent == "grid":
            print(f"             {hero.summary()}; {memory_line()}", flush=True)
        if opp == "overbettor":
            print(f"             overbets made: {sum(v.n_overbets for v in vils):,}", flush=True)
        print_all_in(vils)


def print_all_in(vils) -> None:
    """The line of the opponents' decisions whose lookup moved collapsed raise sizes to the all-in (if any)."""
    n_all_in = sum(getattr(v, "n_all_in", 0) for v in vils)
    if n_all_in:
        print(f"             opponent decisions with raise sizes played as the all-in (collapsed at this stack): "
              f"{n_all_in:,} of {sum(getattr(v, 'n_decisions', 0) for v in vils):,}", flush=True)


if __name__ == "__main__":
    main()
