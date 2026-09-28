"""Vector Linear CFR (SubgameSearch vector_cfr=True) against the MCCFR at equal time: exact exploitability of the
subgame (bb per deal; the best responder deviates on the root's street and after), turn and river roots.

    python scripts/vector_cfr_bench.py --blueprint B.bin --buckets K.json --threads 1 --seconds 0.5,1,2,4 --hands 3

HU 200bb game of the blueprint.  Turn spots of scripts/search_quality.py and two river spots; per root and budget:
the MCCFR (seeds --seeds) and the vector CFR, average and final-iteration profiles, iterations.
"""
import argparse
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

core = fast.core()
FLOP_LINE = ["r1", "c", "c", "r0.5", "c"]
SPOTS = {
    "turn, BB first to act": FLOP_LINE,
    "turn, BB faces a pot bet": FLOP_LINE + ["c", "r1"],
    "river, BB first to act": FLOP_LINE + ["c", "c"],
    "river, BB faces a pot bet": FLOP_LINE + ["c", "c", "c", "r1"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--blueprint", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--seconds", default="0.5,1,2,4")
ap.add_argument("--hands", type=int, default=2)
ap.add_argument("--seeds", type=int, default=2, help="MCCFR seeds per budget")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--spots", default="")
args = ap.parse_args()
bk = load_bucketer(args.buckets)
spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets,
                bucket_kind=getattr(bk, "kind", "ehs"))
bp = load_blueprint(args.blueprint)
game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000)), bp.lookup)
budgets = [float(x) for x in args.seconds.split(",")]
wanted = [w.strip() for w in args.spots.split(",") if w.strip()]
rng = random.Random(args.seed)
summary = {}
print(f"{args.threads} threads; exploitability in bb per deal of the subgame (avg / final profile)", flush=True)
for spot, line in SPOTS.items():
    for h in range(args.hands):
        order = list(range(52))
        rng.shuffle(order)
        if wanted and not any(spot.startswith(w) for w in wanted):
            continue
        st = spec.new_hand(order, button=0)
        acts = []
        for name in line:
            obs = st.observe(st.current_player)
            a = spec.grid.to_concrete(obs, name)
            acts.append((int(a.type), int(a.amount)))
            st.apply(a)
        obs = st.observe(st.current_player)
        br = int(st.street)
        print(f"\n[{spot} #{h}] hole {obs.hole} board {obs.board}", flush=True)

        def run(sec, seed, vec):
            s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                   iterations=0, time_budget=sec, threads=args.threads, seed=seed, depth="pluribus", vector_cfr=vec)
            r = s.solve()
            return s.subgame_exploitability(0, br)[0], s.subgame_exploitability(1, br)[0], r["iterations"], s.vector_eligible

        for sec in budgets:
            mc = [run(sec, sd, False) for sd in range(1, args.seeds + 1)]
            vx = run(sec, 1, True)
            assert vx[3], "vector CFR not eligible here"
            ma = statistics.fmean(x[0] for x in mc)
            mf = statistics.fmean(x[1] for x in mc)
            print(f"  {sec:5.2f}s  MCCFR {ma:7.3f} / {mf:7.3f} ({mc[0][2]:>9,} it)   vector {vx[0]:7.3f} / {vx[1]:7.3f} ({vx[2]:>6,} it)", flush=True)
            summary.setdefault((spot.split(",")[0], sec), []).append((ma, mf, vx[0], vx[1]))
print("\nmean over roots: street, seconds: MCCFR avg / final, vector avg / final")
for (street, sec), rows in summary.items():
    m = [statistics.fmean(r[i] for r in rows) for i in range(4)]
    print(f"  {street:6s} {sec:5.2f}s  MCCFR {m[0]:7.3f} / {m[1]:7.3f}   vector {m[2]:7.3f} / {m[3]:7.3f}")
