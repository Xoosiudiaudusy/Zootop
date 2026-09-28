"""Speed and exact fingerprints of the C++ subgame search at flop, turn and river roots (optimisation work).

    python scripts/search_bench.py --blueprint B.bin --buckets K.json --threads 1 --scale 0.1 --out fp.json
    python scripts/search_bench.py ... --compare fp_ref.json      # the same spots on another build: bit for bit?

The HU 200bb game (grid preflop 0.5/1/3, postflop 0.5/1/2/4, 3 raises), the agent's depth rule ("pluribus") and
the agent's iterations per street (flop 350k, turn 1M, river 3.5M) times --scale.  Spots as in
scripts/search_timing.py, --hands random hands each (fixed --seed).  Per search: the build seconds (ranges,
bucket tables), the solve seconds, iterations per second, and a fingerprint: every float of "final", "average"
and the likelihood() of both seats, as hex.  With --threads 1 and a fixed seed the search is deterministic, so
two builds with the same logic give the same fingerprints; --compare reports the first difference.
A warm-up search per hand runs first (bucket caches), so the timed searches measure the solver; --cold skips it, and
"solve" then includes the bucket work of a new board (every hand has one), as in play.
--search-buckets (with --search-tables / --search-cache) gives the subgame an abstraction of its own for the rounds after
the root's; --infosets counts the subgame's infosets of those rounds per street (and the largest bucket keyed).
"""
from __future__ import annotations

import argparse
import json
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
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

core = fast.core()
PRE = ["r1", "c"]
FLOP_LINE = PRE + ["c", "r0.5", "c"]
SPOTS = {
    "flop, BB first to act": (PRE, 350_000),
    "flop, SB faces a half-pot bet": (PRE + ["r0.5"], 350_000),
    "turn, BB first to act": (FLOP_LINE, 1_000_000),
    "turn, BB faces a pot bet": (FLOP_LINE + ["c", "r1"], 1_000_000),
    "river, BB first to act": (FLOP_LINE + ["c", "c"], 3_500_000),
    "river, BB faces a pot bet": (FLOP_LINE + ["c", "c", "c", "r1"], 3_500_000),
}


def hexes(xs):
    if isinstance(xs, (list, tuple)):
        return [hexes(x) for x in xs]
    if isinstance(xs, dict):
        return {k: hexes(v) for k, v in xs.items()}
    if isinstance(xs, float):
        return xs.hex()
    return xs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.1, help="x the agent's iterations per street")
    ap.add_argument("--hands", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spots", default="", help="comma-separated prefixes of spot names")
    ap.add_argument("--depth", default="pluribus")
    ap.add_argument("--cold", action="store_true", help="no warm-up search: the timed search pays the board's bucket work, as in play")
    ap.add_argument("--cache", default=None, help="saved bucket cache of --buckets to load first")
    ap.add_argument("--search-buckets", default=None, help="the subgame's own buckets for the rounds after the root's (JSON)")
    ap.add_argument("--search-tables", default=None, help="bucket-table directory for --search-buckets (default $NEGPLURIBUS_BUCKET_TABLES)")
    ap.add_argument("--search-cache", default=None, help="saved bucket cache of --search-buckets")
    ap.add_argument("--infosets", action="store_true", help="count the subgame's infosets after the root's round per street")
    ap.add_argument("--out", default=None, help="write the fingerprints (JSON)")
    ap.add_argument("--compare", default=None, help="fingerprints of another build to compare with")
    args = ap.parse_args()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                    postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets,
                    bucket_kind=getattr(bk, "kind", "ehs"))
    bp = load_blueprint(args.blueprint)
    cbk = core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000))
    if args.cache:
        t = time.perf_counter()
        print(f"bucket cache {args.cache}: {cbk.load_cache(args.cache)} in {time.perf_counter() - t:.1f}s", flush=True)
    scbk = None
    if args.search_buckets:
        t = time.perf_counter()
        scbk = core_bucketer(load_bucketer(args.search_buckets), (8_000_000, 64_000_000, 4_000_000), tables=args.search_tables)
        if args.search_cache:
            getattr(scbk, "inner", scbk).load_cache(args.search_cache)
        print(f"search buckets {args.search_buckets}: {scbk.identity['n_buckets']} ({type(scbk).__name__}) in "
              f"{time.perf_counter() - t:.1f}s", flush=True)
    game = core.SearchGame(spec_to_dict(spec), cbk, bp.lookup, search_bucketer=scbk)
    rng = random.Random(args.seed)
    wanted = [s.strip() for s in args.spots.split(",") if s.strip()]
    fps = {}
    totals = {}
    print(f"{args.threads} threads, scale {args.scale}, depth {args.depth}, module {core.__file__}", flush=True)
    for spot, (line, iters) in SPOTS.items():
        if wanted and not any(spot.startswith(w) for w in wanted):
            for _ in range(args.hands):
                rng.shuffle(list(range(52)))  # keep the hands of the other spots the same
            continue
        n_it = max(1, int(iters * args.scale))
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

            def run(seed):
                t = time.perf_counter()
                s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                       iterations=n_it, time_budget=0.0, threads=args.threads, seed=seed, depth=args.depth)
                tb = time.perf_counter() - t
                r = s.solve()
                return s, r, tb

            if not args.cold:
                run(12345)  # warm-up: bucket caches of this board
            s, r, tb = run(7)
            key = f"{spot} #{h}"
            fp = {"final": hexes(r["final"]), "average": hexes(r["average"]), "iterations": r["iterations"],
                  "likelihood": [hexes(s.likelihood(seat)) for seat in (0, 1)]}
            fps[key] = fp
            it_s = r["iterations"] / max(r["seconds"], 1e-9)
            tt = totals.setdefault(spot.split(",")[0], [0, 0.0, 0.0])
            tt[0] += r["iterations"]
            tt[1] += r["seconds"]
            tt[2] += tb
            extra = ""
            if args.infosets:
                names = {1: "flop", 2: "turn", 3: "river"}
                extra = "  later infosets " + ", ".join(f"{names[k]} {v[0]:,} (max bucket {v[1]})" for k, v in sorted(s._later_infosets().items()))
            print(f"{key:36s} build {tb * 1000:7.1f} ms  solve {r['seconds']:7.3f}s  {r['iterations']:>9,} it  {it_s:>12,.0f} it/s  "
                  f"table {r['table_size']:,}{extra}", flush=True)
    print()
    for street, (it, sec, tb) in totals.items():
        print(f"{street:6s}: {it / max(sec, 1e-9):>12,.0f} it/s over {it:,} iterations; build {tb / max(1, args.hands * 2) * 1000:.1f} ms per search")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(fps, f)
    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        bad = [k for k in fps if k in ref and fps[k] != ref[k]]
        missing = [k for k in fps if k not in ref]
        print(f"\ncompare with {args.compare}: {len(fps) - len(bad) - len(missing)} identical, {len(bad)} different, {len(missing)} missing")
        for k in bad[:3]:
            for field in ("iterations", "final", "average", "likelihood"):
                if fps[k][field] != ref[k][field]:
                    print(f"  {k}: first difference in {field}")
                    break
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
