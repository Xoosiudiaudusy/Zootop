"""The C++ subgame search next to the Python reference search (cfr/search.py) on flop spots.

    python scripts/search_compare.py --tag 2p_20bb_flop --hands 8 --cpp-seconds 1 --threads 8

Same blueprint, same hands, same decision; the two searches differ by design (docs/search_design.md):
root at the start of the round vs at the decision, exact ranges vs 200 particles, lossless flop
classes vs the blueprint's buckets, final iteration vs average, no warm start vs a blueprint prior.
So the comparison is of distributions, not numbers: for every spot and hand it prints the blueprint,
the Python search (depth to the end of the hand), and the C++ search (final iteration and average),
then the mean total-variation distances and the average probability of fold / check-call / bet-raise.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.abstraction.infoset import infoset_key  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.cfr.search import SearchConfig, SubgameSolver  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
core = fast.core()

# flop spots after "SB raises pot, BB calls": the actions on the flop before our decision
SPOTS = {
    "BB first to act": [],
    "SB vs check": ["c"],
    "SB vs half-pot bet": ["r0.5"],
    "BB vs bet after check": ["c", "r0.5"],
}


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def kinds(names, probs):
    out = {"fold": 0.0, "check/call": 0.0, "bet/raise": 0.0}
    for n, p in zip(names, probs):
        out["fold" if n == "f" else "check/call" if n == "c" else "bet/raise"] += p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="2p_20bb_flop")
    ap.add_argument("--data", default=DATA, help="where blueprint_<tag>.json and buckets_<tag>.json are")
    ap.add_argument("--hands", type=int, default=8, help="hands per spot")
    ap.add_argument("--py-iters", type=int, default=800)
    ap.add_argument("--py-prior", type=int, default=300, help="Python search: blueprint prior in iterations (0 = off)")
    ap.add_argument("--cpp-seconds", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    disable_power_throttling()
    bk = load_bucketer(os.path.join(args.data, f"buckets_{args.tag}.json"))
    spec = GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
    bp_dict = load_blueprint(os.path.join(args.data, f"blueprint_{args.tag}.json"), backend="python")
    bp_cpp = load_blueprint(os.path.join(args.data, f"blueprint_{args.tag}.json"))
    game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk), bp_cpp.lookup)
    cfg = SearchConfig(iterations=args.py_iters, depth=None, prior_iters=args.py_prior)
    print(f"game: {spec.describe()}; blueprint {len(bp_dict)} infosets; Python search {cfg}; C++ search "
          f"{args.cpp_seconds}s x {args.threads} threads")
    rng = random.Random(args.seed)
    summary = {}
    for spot, flop_actions in SPOTS.items():
        rows = []
        for h in range(args.hands):
            order = list(range(52))
            rng.shuffle(order)
            st = spec.new_hand(order, button=0)
            acts = []
            ok = True
            for name in ["r1", "c"] + flop_actions:
                obs = st.observe(st.current_player)
                if name not in spec.grid.abstract_actions(obs):
                    ok = False
                    break
                a = spec.grid.to_concrete(obs, name)
                acts.append((int(a.type), int(a.amount)))
                st.apply(a)
            if not ok or st.is_terminal or st.street != Street.FLOP:
                continue
            obs = st.observe(st.current_player)
            legal = spec.grid.abstract_actions(obs)
            key = infoset_key(obs, bk, spec.grid)
            bpp = bp_dict.policy(key, legal) or [1.0 / len(legal)] * len(legal)
            t = time.perf_counter()
            py = SubgameSolver(spec, bp_dict, bk, cfg, seed=args.seed + h).solve(obs)
            t_py = time.perf_counter() - t
            s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                   iterations=0, time_budget=args.cpp_seconds, threads=args.threads, seed=args.seed + h)
            r = s.solve()
            assert r["actions"] == legal
            rows.append((legal, bpp, py, r["final"], r["average"]))
            fmt = lambda p: " ".join(f"{x:.2f}" for x in p) if p else "none"  # noqa: E731
            print(f"[{spot}] hole {obs.hole} board {obs.board} actions {legal}\n"
                  f"   blueprint {fmt(bpp)} | python {fmt(py)} ({t_py:.1f}s) | c++ final {fmt(r['final'])} "
                  f"avg {fmt(r['average'])} ({r['iterations']:,} it, {r['seconds']:.1f}s)", flush=True)
        summary[spot] = rows
    print("\nsummary: mean total-variation distance, and mean probabilities (fold / check-call / bet-raise)")
    for spot, rows in summary.items():
        rows = [r for r in rows if r[2] is not None]
        if not rows:
            continue
        n = len(rows)
        d = {
            "python-blueprint": sum(tv(r[2], r[1]) for r in rows) / n,
            "c++final-blueprint": sum(tv(r[3], r[1]) for r in rows) / n,
            "c++avg-blueprint": sum(tv(r[4], r[1]) for r in rows) / n,
            "c++final-python": sum(tv(r[3], r[2]) for r in rows) / n,
            "c++avg-python": sum(tv(r[4], r[2]) for r in rows) / n,
        }
        mix = {}
        for label, idx in (("blueprint", 1), ("python", 2), ("c++ final", 3), ("c++ avg", 4)):
            acc = {"fold": 0.0, "check/call": 0.0, "bet/raise": 0.0}
            for r in rows:
                for k, v in kinds(r[0], r[idx]).items():
                    acc[k] += v / n
            mix[label] = acc
        print(f"  {spot} ({n} hands): " + ", ".join(f"TV {k} {v:.2f}" for k, v in d.items()))
        for label, acc in mix.items():
            print(f"      {label:<10} fold {acc['fold']:.2f}  check/call {acc['check/call']:.2f}  bet/raise {acc['bet/raise']:.2f}")


if __name__ == "__main__":
    main()
