"""Depth-limited search against the same search to the end of the hand, measured exactly.

    python scripts/search_quality.py --blueprint data/blueprint_hunl200w3_pot16_s0.bin \
        --buckets data/buckets_hunl200w3_pot16_s0.json --cache data/bucketcache_hunl200w3_pot16_s0.bin \
        --threads 14 --seconds 2 --hands 5 --presample

Turn roots of the HU 200bb game.  Each hand is solved with the same budget: A to the end of the hand
(river infosets on the blueprint's buckets, solved in the subgame) and B with leaves at the start
of the river (each player's continuation mix, 3 rollouts per leaf); A2 and B2 are the same with
another seed (the noise between seeds); Aeq is A stopped at B's number of iterations; with
--presample, Bp is B on a game whose rollouts play pre-sampled actions (same seed as B).

In play the agent would not play B's river continuations: it searches again on the river.  That
is what the frozen searches measure: X~ = a search to the end in which both players play the turn
exactly as X says (frozen, average or final iteration) and only the river is learned
(SubgameSearch.freeze_round), for --resolve-seconds.

Measured per hand, everything in bb per deal of the subgame:
* the distance between root strategies: total variation for our actual hole, and averaged over our
  whole range at the decision (weights = our root range x the likelihood of our path actions);
* the exact exploitability (2 players, the river card a chance node, all hole pairs with card
  removal; SubgameSearch.subgame_exploitability): "subgame" = the best responder deviates on the
  turn and the river, "river" = on the river only, with the turn played as the profile says,
  "model" (B's only) = inside B's own model: deviations on the turn and in the continuation choice
  at a leaf, the river played by the continuations (how far B is from solving what it solves).
  Profiles: A, A2, Aeq, the blueprint (as the agent plays it), B, B2, Bp (river = continuation
  mix), A~, B~, B2~, Bp~ (turn frozen, river re-solved), each average and final iteration (a frozen
  final profile is the final turn iteration with the final iteration of the river re-solve).
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
from negpluribus.fast.power import disable_power_throttling  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

core = fast.core()
FLOP_LINE = ["r1", "c", "c", "r0.5", "c"]
SPOTS = {"turn, BB first to act": FLOP_LINE, "turn, BB faces a pot bet": FLOP_LINE + ["c", "r1"]}
KINDS = ((0, "avg"), (1, "final"))


def fracs(s: str):
    return tuple(float(x) for x in s.split(",") if x.strip())


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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--stack", type=int, default=200)
    ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--resolve-seconds", type=float, default=4.0, help="budget of the frozen-turn river re-solves")
    ap.add_argument("--hands", type=int, default=3, help="hands per spot")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--presample", action="store_true", help="also Bp: B with pre-sampled continuation actions")
    ap.add_argument("--sweep", default="", help="instead: iteration counts (e.g. 30000,100000,300000,1000000); per hand, "
                    "B and A at each count, exploitability inside B's model, of the subgame and of the river play")
    args = ap.parse_args()
    disable_power_throttling()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=args.stack, max_street=Street.RIVER, preflop_fracs=fracs(args.preflop_fracs),
                    postflop_fracs=fracs(args.postflop_fracs), max_raises_per_street=args.max_raises,
                    n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
    bp = load_blueprint(args.blueprint)
    cbk = core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000))
    if args.cache:
        cbk.load_cache(args.cache)
    game = core.SearchGame(spec_to_dict(spec), cbk, bp.lookup)
    game_p = None
    if args.presample:
        game_p = core.SearchGame(spec_to_dict(spec), cbk, bp.lookup)
        t = time.perf_counter()
        game_p.presample(args.seed)
        print(f"pre-sampled continuation actions (Bp's game): {game_p.presampled_bytes:,} bytes in {time.perf_counter() - t:.1f}s")
    print(f"game: {spec.describe()}; searches {args.seconds}s, river re-solves {args.resolve_seconds}s, {args.threads} threads; "
          f"exploitability in bb per deal of the subgame")
    rng = random.Random(args.seed)
    rows = []
    for spot, line in SPOTS.items():
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
            if args.sweep:
                print(f"\n[{spot}] hole {obs.hole} board {obs.board}", flush=True)
                for iters in (int(x) for x in args.sweep.split(",") if x.strip()):
                    for depth in ("next_street", "end"):
                        s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                               iterations=iters, time_budget=0.0, threads=args.threads, seed=1, depth=depth)
                        r = s.solve()
                        out = f"   {depth:<11} {iters:>9,} it ({r['seconds']:5.1f}s):"
                        for kind, kname in KINDS:
                            model = s.subgame_exploitability(kind, 7)[0] if depth == "next_street" else float("nan")
                            out += (f"  {kname}: model {model:6.3f} subgame {s.subgame_exploitability(kind, 2)[0]:6.3f} "
                                    f"river {s.subgame_exploitability(kind, 3)[0]:6.3f};")
                        print(out, flush=True)
                continue
            t_hand = time.perf_counter()

            def search(depth, seed, g=game, seconds=args.seconds):
                return core.SubgameSearch(g, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                          iterations=0, time_budget=seconds, threads=args.threads, seed=seed, depth=depth)

            runs = {}  # label -> (search, result)
            plan = [("A", "end", h, game), ("A2", "end", h + 1000, game), ("B", "next_street", h, game),
                    ("B2", "next_street", h + 1000, game)]
            if game_p is not None:
                plan.append(("Bp", "next_street", h, game_p))
            for label, depth, seed, g in plan:
                s = search(depth, seed, g)
                runs[label] = (s, s.solve())
            # A with B's number of iterations: the depth rule's effect apart from the iterations it costs
            s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                   iterations=runs["B"][1]["iterations"], time_budget=0.0, threads=args.threads, seed=h, depth="end")
            runs["Aeq"] = (s, s.solve())
            for base in ("A", "B", "B2") + (("Bp",) if game_p is not None else ()):
                src, _ = runs[base]
                for kind, kname in KINDS:
                    s = search("end", h + 2000 + kind, game_p if base == "Bp" else game, args.resolve_seconds)
                    s.freeze_round(src, kind)
                    runs[f"{base}~{kname}"] = (s, s.solve())
            t_search = time.perf_counter() - t_hand
            t = time.perf_counter()
            ex = {}
            nan = float("nan")
            for label in ("A", "A2", "Aeq", "B", "B2") + (("Bp",) if game_p is not None else ()):
                s = runs[label][0]
                for kind, kname in KINDS:
                    model = s.subgame_exploitability(kind, 7)[0] if label.startswith("B") else nan
                    ex[f"{label} {kname}"] = (s.subgame_exploitability(kind, 2)[0], s.subgame_exploitability(kind, 3)[0], model)
            ex["blueprint"] = (runs["A"][0].subgame_exploitability(2, 2)[0], runs["A"][0].subgame_exploitability(2, 3)[0], nan)
            for base in ("A", "B", "B2") + (("Bp",) if game_p is not None else ()):
                for kind, kname in KINDS:
                    s = runs[f"{base}~{kname}"][0]
                    ex[f"{base}~ {kname}"] = (s.subgame_exploitability(kind, 2)[0], s.subgame_exploitability(kind, 3)[0], nan)
            t_eval = time.perf_counter() - t
            sa = runs["A"][0]
            k = len(sa.path())
            w_reach = sa.ranges()[obs.seat]
            lik, _ = sa.likelihood(obs.seat)
            w = [a * b for a, b in zip(w_reach, lik)]
            strat = {label: {kind: runs[label][0].path_strategies(k, kind) for kind, _ in KINDS} for label in runs if "~" not in label}
            dist = {}
            for x, y in (("A", "B"), ("A", "A2"), ("B", "B2")) + ((("B", "Bp"),) if game_p is not None else ()):
                for kind, kname in KINDS:
                    ours = tv(runs[x][1]["final" if kind else "average"], runs[y][1]["final" if kind else "average"])
                    dist[f"{x}-{y} {kname}"] = (ours, range_tv(strat[x][kind], strat[y][kind], w))
            pot = sa.root_info()["pot"] / spec.bb
            row = dict(spot=spot, pot=pot, ex=ex, dist=dist, it={lb: r["iterations"] for lb, (_, r) in runs.items()})
            rows.append(row)
            fmt = lambda p: " ".join(f"{x:.2f}" for x in p)  # noqa: E731
            ra, rb = runs["A"][1], runs["B"][1]
            print(f"\n[{spot}] hole {obs.hole} board {obs.board}, pot {pot:.0f}bb, actions {ra['actions']} "
                  f"[searches {t_search:.0f}s, evaluations {t_eval:.0f}s]")
            print("   iterations: " + ", ".join(f"{lb} {r['iterations']:,}" for lb, (_, r) in runs.items()))
            print(f"   our hole: A final {fmt(ra['final'])} avg {fmt(ra['average'])}; B final {fmt(rb['final'])} avg {fmt(rb['average'])}")
            print("   TV (our hole / our range): " + "; ".join(f"{key} {a:.3f} / {b:.3f}" for key, (a, b) in dist.items()))
            print("   exploitability subgame / river / model: " + "; ".join(f"{key} {a:.3f} / {b:.3f} / {m:.3f}" for key, (a, b, m) in ex.items()),
                  flush=True)
    n = len(rows)
    if not n:
        return
    print(f"\n==== means over {n} hands (pot {statistics.mean(r['pot'] for r in rows):.1f}bb), bb per deal; "
          f"+- = standard error of the mean over hands")

    def ms(vals):
        if any(v != v for v in vals):  # not measured for this profile
            return "-"
        m = statistics.mean(vals)
        se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
        return f"{m:.3f} +- {se:.3f}"

    print("exploitability          subgame              river play           own model (leaf game)")
    for key in rows[0]["ex"]:
        print(f"   {key:<14} " + " ".join(f"{ms([r['ex'][key][j] for r in rows]):>20}" for j in range(3)))
    print("distance (TV)           our hole             our range")
    for key in rows[0]["dist"]:
        print(f"   {key:<14} {ms([r['dist'][key][0] for r in rows]):>20} {ms([r['dist'][key][1] for r in rows]):>20}")
    print("paired differences of the subgame exploitability (mean +- s.e. over hands):")
    pairs = [("B~ avg", "A~ avg"), ("B~ final", "A~ final"), ("A~ avg", "A avg"), ("B2~ avg", "B~ avg"), ("A2 avg", "A avg"),
             ("Aeq avg", "A avg"), ("B~ avg", "Aeq avg"), ("B avg", "blueprint"), ("B~ avg", "blueprint"), ("A avg", "blueprint")]
    if game_p is not None:
        pairs += [("Bp avg", "B avg"), ("Bp~ avg", "B~ avg"), ("Bp~ final", "B~ final")]
    for x, y in pairs:
        print(f"   {x} - {y}: {ms([r['ex'][x][0] - r['ex'][y][0] for r in rows])}")
    print("iterations (mean): " + ", ".join(f"{lb} {statistics.mean(r['it'][lb] for r in rows):,.0f}" for lb in rows[0]["it"]))


if __name__ == "__main__":
    main()
