"""Regret-based pruning in the subgame search (SearchParams::prune_mode, type b): is a pruned search better at equal time?

    python scripts/search_pruning_bench.py --blueprint B.bin --buckets K.json --street turn --threads 4 \\
        --seconds 2,4,8 --hands 2 --seeds 3 --configs off,r0.05,r0.2,t5

Configs: off | r<f> (relative: regret < -f x sum |regret| of the node) | t<C> (regret < -C x t) | a<C> (regret < -C),
a suffix "n" (e.g. r0.2n) = never on river nodes (as Pluribus on the last street),
each with --prune-prob of the iterations pruning, not in the last --prune-stop share of the budget, never on the real
path.  The HU 200bb game of scripts/search_quality.py, depth "pluribus", the spots' roots with --hands random hands each.

Turn and river roots (--street turn / river): the exact exploitability of the subgame (subgame_exploitability, the best
responder deviating from the root's street; bb per deal), average and final iteration, per config and time budget.
Flop roots (--street flop, no exact exploitability there): the distance to a reference solution -- a search without
pruning given --ref-mult x the largest budget (seed 999) -- as the total variation of the average strategy on the real
path, weighted over our range at the decision (and for our actual hole); "off" against the reference with other seeds
is the scale of the noise.  Also iterations per second and the share of the traverser's actions pruned.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
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
    "flop": {"flop, BB first to act": PRE, "flop, SB faces a half-pot bet": PRE + ["r0.5"]},
    "turn": {"turn, BB first to act": FLOP_LINE, "turn, BB faces a pot bet": FLOP_LINE + ["c", "r1"]},
    "river": {"river, BB first to act": FLOP_LINE + ["c", "c"], "river, BB faces a pot bet": FLOP_LINE + ["c", "c", "c", "r1"]},
}
STREET = {"flop": 1, "turn": 2, "river": 3}


def parse(cfg: str) -> dict:
    if cfg == "off":
        return {}
    kw = {}
    if cfg.endswith("n"):  # suffix n: never on river nodes
        kw["prune_river"] = False
        cfg = cfg[:-1]
    mode = {"a": 1, "t": 2, "r": 3}[cfg[0]]
    return dict(kw, prune_mode=mode, prune_below=float(cfg[1:]))


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def range_tv(ra, rb, w):
    num = den = 0.0
    for c in range(len(w)):
        if w[c] > 0.0 and ra[c] is not None and rb[c] is not None:
            num += w[c] * tv(ra[c], rb[c])
            den += w[c]
    return num / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--street", choices=tuple(SPOTS), default="turn")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", default="2,4,8")
    ap.add_argument("--hands", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7, help="the deals")
    ap.add_argument("--configs", default="off,r0.05,r0.2,t5")
    ap.add_argument("--prune-prob", type=float, default=0.95)
    ap.add_argument("--prune-stop", type=float, default=0.1)
    ap.add_argument("--ref-mult", type=float, default=8.0, help="flop: the reference's budget, x the largest")
    args = ap.parse_args()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                    postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets,
                    bucket_kind=getattr(bk, "kind", "ehs"))
    game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000)), load_blueprint(args.blueprint).lookup)
    budgets = [float(x) for x in args.seconds.split(",")]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    rng = random.Random(args.seed)
    flop = args.street == "flop"
    table = {}  # (config, seconds, metric) -> [values over roots x seeds]
    print(f"{args.street} roots, {args.threads} threads, seeds {args.seeds}, prune prob {args.prune_prob}, stop {args.prune_stop}; "
          + ("TV to the reference (x" + f"{args.ref_mult:g} budget, no pruning)" if flop else "exact exploitability, bb per deal"), flush=True)
    for spot, line in SPOTS[args.street].items():
        for h in range(args.hands):
            order = list(range(52))
            rng.shuffle(order)
            st = spec.new_hand(order, button=0)
            acts = []
            for name in line:
                a = spec.grid.to_concrete(st.observe(st.current_player), name)
                acts.append((int(a.type), int(a.amount)))
                st.apply(a)
            obs = st.observe(st.current_player)

            def search(sec, seed, **kw):
                return core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                          iterations=0, time_budget=sec, threads=args.threads, seed=seed, depth="pluribus",
                                          prune_prob=args.prune_prob, prune_stop=args.prune_stop, **kw)
            print(f"\n[{spot} #{h}] board {obs.board} hole {obs.hole}", flush=True)
            ref = None
            if flop:
                t = time.perf_counter()
                s = search(args.ref_mult * max(budgets), 999)
                r = s.solve()
                k = len(s.path())
                w = [a * b for a, b in zip(s.ranges()[obs.seat], s.likelihood(obs.seat)[0])]
                ref = (s.path_strategies(k, 0), r["average"], k, w)
                print(f"  reference: {r['iterations']:,} iterations ({time.perf_counter() - t:.0f}s)", flush=True)
            for sec in budgets:
                cells = []
                for cfg in configs:
                    kw = parse(cfg)
                    vals = {}
                    for sd in range(1, args.seeds + 1):
                        s = search(sec, sd, **kw)
                        r = s.solve()
                        touched = r["nodes_touched"]
                        vals.setdefault("it/s", []).append(r["iterations"] / max(r["seconds"], 1e-9))
                        vals.setdefault("pruned", []).append(r["pruned"] / max(1, r["pruned"] + touched))
                        if flop:
                            vals.setdefault("tv range", []).append(range_tv(s.path_strategies(ref[2], 0), ref[0], ref[3]))
                            vals.setdefault("tv hole", []).append(tv(r["average"], ref[1]))
                        else:
                            vals.setdefault("eps avg", []).append(s.subgame_exploitability(0, STREET[args.street])[0])
                            vals.setdefault("eps final", []).append(s.subgame_exploitability(1, STREET[args.street])[0])
                    for m, v in vals.items():
                        table.setdefault((cfg, sec, m), []).extend(v)
                    main_m = "tv range" if flop else "eps avg"
                    cells.append(f"{cfg} {statistics.fmean(vals[main_m]):.3f} ({statistics.fmean(vals['it/s']):,.0f}/s)")
                print(f"  {sec:5.1f}s  " + "  ".join(cells), flush=True)
    metrics = ["tv range", "tv hole"] if flop else ["eps avg", "eps final"]
    metrics += ["it/s", "pruned"]
    print("\nmean over roots x seeds (+- the standard error)")
    for m in metrics:
        print(f"\n{m}:")
        print("  seconds " + "".join(f"{c:>22s}" for c in configs))
        for sec in budgets:
            row = []
            for c in configs:
                v = table[(c, sec, m)]
                se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
                row.append(f"{statistics.fmean(v):12.4g} +- {se:<7.2g}")
            print(f"  {sec:7.1f} " + "".join(f"{x:>22s}" for x in row))


if __name__ == "__main__":
    main()
