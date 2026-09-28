"""AIVAT on logged hands: bb/100 with 95% CIs for the raw nets, the card-luck correction and AIVAT
(docs/aivat.md; heuristic v1, fixed before any evaluation log was read).

    # our blueprint vs Slumbot (the known player: our blueprint agent)
    python scripts/aivat_eval.py --slumbot data/slumbot/hunl200w3_pot16_s0.jsonl --out data/aivat/slumbot.jsonl \
        --blueprint data/blueprint_hunl200w3_pot16_s0.bin --buckets data/buckets_hunl200w3_pot16_s0.json \
        --cache data/bucketcache_hunl200w3_pot16_s0.bin --threads 4

    # a duplicate duel log of scripts/eval_archetypes.py --log-hands (known player: the hero, a blueprint agent)
    python scripts/aivat_eval.py --duel hands.jsonl --out data/aivat/duel.jsonl ...

    # the hero a depth grid (eval_archetypes --agent grid) at fixed 50bb stacks: the known player's blueprint per hand
    python scripts/aivat_eval.py --duel hands.jsonl --known hero --blueprints data/stack_grid_pot16_s0.json \
        --stack-bb 50 --out out/grid50_aivat.jsonl --root-dir out/roots --threads 4 --limit 20000

Results are appended per hand to --out (one JSON line: hand id, deal, position, net, AIVAT value,
base, term sums, seconds), so an interrupted run resumes where it stopped; the summary is printed
at the end (and with --summary-only from --out alone).  Bucket tables of the abstraction are built
once into --tables (river from per-board batches, flop and turn through the warm --cache: the same
numbers as bucket()); the root table (u at the start of a hand) once into --root-cache.

--blueprints <manifest> (negpluribus/agents/stack_grid.py): the known player (the logged hero) played each hand with
the blueprint of one grid point, which the log names at the hero's first decision ("blueprint", else "point"; a hand
without a hero decision: the point the grid's rule gives at the hand's stacks).  One model per blueprint, made when a
hand first needs it, on the manifest's bucketer and bets; its root table in --root-dir (root_v1_<blueprint>_<S>bb.npz).
A root table is computed at fixed starting stacks (--stack-bb for both seats), so every hand's logged stacks must be
those: a duel with carried stacks is refused here (score it raw; docs/stack_grid_design.md 6.3).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval.aivat import AivatHand, hand_from_duel, hand_from_slumbot  # noqa: E402
from negpluribus.eval.aivat_fast import TERM_NAMES, FastAivat, hands_needed, make_game, root_table  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
V1 = {"rollouts": (4, 8, 8), "eq_samples": 2000, "root_rollouts": 256, "seed": 0}  # heuristic v1 (docs/aivat.md)


def fracs(s: str):
    return tuple(float(x) for x in s.split(",") if x.strip())


def read_complete_lines(path: str):
    """Every complete line of a JSONL file that may still be appended to (a partial last line is skipped)."""
    with open(path, "rb") as f:
        data = f.read()
    data = data[: data.rfind(b"\n") + 1]
    for ln in data.split(b"\n"):
        if ln.strip():
            yield json.loads(ln)


def load_hands(args, grid=None):
    """[(AivatHand, extra)] in log order; extra: position, deal, card-luck inputs; with a depth ``grid`` (--blueprints)
    also the point and the blueprint of the known player in that hand."""
    out, skipped = [], {}
    if args.slumbot:
        for rec in read_complete_lines(args.slumbot):
            if rec.get("type") != "hand":
                continue
            why = None
            if rec.get("status") != "ok" or rec.get("winnings") is None:
                why = "no result"
            elif not rec.get("bot_cards"):
                why = "Slumbot's cards unknown"
            elif rec.get("check") != "match":
                why = f"engine check {rec.get('check')}"
            h = None if why else hand_from_slumbot(rec)
            if h is None:
                skipped[why or "replay"] = skipped.get(why or "replay", 0) + 1
                continue
            if h.net != rec["winnings"]:
                raise RuntimeError(f"hand {rec['hand']}: replayed net {h.net} != winnings {rec['winnings']}")
            out.append((h, {"position": rec.get("position"), "slumbot": rec}))
    else:
        stack = args.stack_bb * 100
        for i, rec in enumerate(read_complete_lines(args.duel)):
            if len(rec["holes"]) != 2:
                raise ValueError("heads-up hands only")
            h = hand_from_duel(rec, stack=stack, hand_id=i, known=args.known)
            if h.stacks != (stack, stack):  # the root table is for these stacks (and the old lines had no others)
                raise SystemExit(f"hand {i} (deal {rec['deal']}): starting stacks {list(h.stacks)} are not the root "
                                 f"table's [{stack}, {stack}] (--stack-bb {args.stack_bb}); AIVAT here is at fixed stacks "
                                 f"only: score a duel with carried stacks raw")
            luck = rec.get("luck_bb")
            extra = {"position": "SB" if h.known_seat == h.button else "BB", "deal": int(rec["deal"]),
                     "luck_bb": None if luck is None else (luck if args.known == "hero" else -luck),
                     "opponent": rec.get("opponent")}
            if grid is not None:
                p = grid.point_for_hand(rec, args.stack_bb)
                extra.update(point=p.stack_bb, blueprint=p.blueprint)
            out.append((h, extra))
    return out, skipped


def card_luck_slumbot(entries, cache_path=None):
    """scripts/slumbot_luck.py's chance-node correction on these hands, in log order (its preflop
    equity is Monte-Carlo from random.Random(7), consumed hand by hand, so the values of a log's first n
    hands do not change when the log grows).  ``cache_path``: JSON {hand: corrected chips}, reused when it
    covers every hand, else recomputed from the first hand and rewritten."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if all(str(extra["slumbot"]["hand"]) in cached for _, extra in entries):
            return [cached[str(extra["slumbot"]["hand"])] for _, extra in entries]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import slumbot_luck as L
    from negpluribus.slumbot.protocol import parse_cards

    rng = random.Random(7)
    out = []
    for _, extra in entries:
        r = extra["slumbot"]
        h, o = parse_cards(r["hero_cards"]), parse_cards(r["bot_cards"])
        board = parse_cards(r.get("board") or [])
        pots, _, _ = L.pots_by_street(r["action"], r["client_pos"] == 1)
        eqs = [L.eq_preflop(h, o, rng)]
        for k, nb in ((1, 3), (2, 4), (3, 5)):
            if len(pots) > k and len(board) >= nb:
                eqs.append(L.eq_exact(h, o, board[:nb]))
        corr = (eqs[0] - 0.5) * pots[0]
        for k in range(1, len(eqs)):
            corr += (eqs[k] - eqs[k - 1]) * pots[k]
        out.append(r["winnings"] - corr)
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({str(extra["slumbot"]["hand"]): v for (_, extra), v in zip(entries, out)}, f)
    return out


def stats(xs_bb, groups=None):
    """bb/100, 95% half-width, sd per hand (bb), n; with ``groups`` a group (deal) is one sample."""
    x = np.asarray(xs_bb, dtype=float)
    sd_hand = float(x.std(ddof=1))
    if groups is None:
        n = len(x)
        return {"bb100": float(x.mean()) * 100, "ci95": 1.96 * sd_hand / math.sqrt(n) * 100, "sd": sd_hand, "n": n}
    g = {}
    for k, v in zip(groups, x):
        g.setdefault(k, []).append(v)
    per = np.array([sum(v) / len(v) for v in g.values()])  # mean per hand of a deal
    return {"bb100": float(per.mean()) * 100, "ci95": 1.96 * float(per.std(ddof=1)) / math.sqrt(len(per)) * 100,
            "sd": sd_hand, "sd_deal": float(per.std(ddof=1)), "n": len(x), "n_deals": len(per)}


def summary(rows, luck=None, groups=None, label=""):
    raw = [r["net"] / 100 for r in rows]
    av = [r["value"] / 100 for r in rows]
    lines = [f"{label}{len(rows):,} hands" + (f" ({len(set(groups)):,} duplicate deals: a deal is one sample)" if groups else "")]
    lines.append(f"{'estimate':26} {'bb/100':>9} {'95% CI':>8} {'sd/hand':>8} {'for +/-10':>11} {'for +/-5':>11}")
    series = [("raw", raw)] + ([("card luck (chance nodes)", luck)] if luck is not None else []) + [("AIVAT v1", av)]
    for name, xs in series:
        s = stats(xs, groups)
        sd_eff = s["sd"] if groups is None else s["sd_deal"] * math.sqrt(len(xs) / s["n_deals"])  # per-hand equivalent
        lines.append(f"{name:26} {s['bb100']:+9.1f} {s['ci95']:8.1f} {s['sd']:8.2f} {hands_needed(sd_eff, 10):11,.0f} {hands_needed(sd_eff, 5):11,.0f}")
    t = np.array([r["seconds"] for r in rows])
    lines.append(f"evaluator: {t.mean():.3f} s per hand on one thread (median {np.median(t):.3f}, max {t.max():.1f})")
    kinds = {}
    for r in rows:
        for k, v in r["terms"].items():
            kinds.setdefault(k, []).append(v / 100)
    lines.append("terms (bb, mean +/- 95% over hands, all hands; 0 where a hand has none): " + "; ".join(
        f"{k} {np.mean(v):+.3f} +/- {1.96 * np.std(v, ddof=1) / math.sqrt(len(v)):.3f}" for k, v in kinds.items()))
    base = np.array([r["base"] / 100 for r in rows])
    lines.append(f"base value {base.mean():+.4f} bb; AIVAT - raw {np.mean(np.array(av) - np.array(raw)):+.4f} bb per hand")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--slumbot", help="a Slumbot match log (negpluribus/slumbot/runner.py)")
    src.add_argument("--duel", help="hand lines of scripts/eval_archetypes.py --log-hands (heads-up)")
    ap.add_argument("--out", required=True, help="per-hand results (JSONL, appended; resumes)")
    ap.add_argument("--blueprint", default=os.path.join(DATA, "blueprint_hunl200w3_pot16_s0.bin"))
    ap.add_argument("--buckets", default=os.path.join(DATA, "buckets_hunl200w3_pot16_s0.json"))
    ap.add_argument("--cache", default=None, help="saved bucket cache (scripts/precompute_buckets.py): fast flop/turn tables")
    ap.add_argument("--tables", default=os.path.join(DATA, "bucket_tables"), help="bucket table directory")
    ap.add_argument("--root-cache", default=None, help="root table .npz (default: next to --out)")
    ap.add_argument("--blueprints", default=None,
                    help="--duel of a depth-grid hero (eval_archetypes --agent grid): the grid's manifest; the known player's "
                         "blueprint per hand from the log (see the module doc); --blueprint, --buckets and the bet flags are "
                         "then the manifest's")
    ap.add_argument("--root-dir", default=None,
                    help="--blueprints: directory of the root tables, one per blueprint at --stack-bb (default: next to --out)")
    ap.add_argument("--known", choices=("hero", "villain"), default="hero",
                    help="--duel: whose strategy is known (the blueprint agent): the logged hero, or the other seat "
                         "(a duel search vs blueprint: the numbers are then the blueprint's, minus the hero's result)")
    ap.add_argument("--stack-bb", type=int, default=200)
    ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="only the first N hands")
    ap.add_argument("--no-luck", action="store_true", help="skip the card-luck column")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--first", type=int, default=0, help="also summarize the first N hands (a comparison with an earlier count)")
    args = ap.parse_args()
    grid = None
    if args.blueprints:
        from negpluribus.agents.stack_grid import StackGrid

        if not args.duel or args.known != "hero":
            ap.error("--blueprints: a --duel log with --known hero (the log names the hero's grid points)")
        grid = StackGrid.load(args.blueprints)
        args.buckets = grid.resolve(grid.bucketer)
        args.preflop_fracs = ",".join(str(x) for x in grid.preflop_fracs)
        args.postflop_fracs = ",".join(str(x) for x in grid.postflop_fracs)
        args.max_raises = grid.max_raises
        print(grid.describe(), flush=True)

    t0 = time.time()
    entries, skipped = load_hands(args, grid)
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries):,} hands to score ({time.time() - t0:.0f}s); skipped: {skipped or 'none'}", flush=True)
    done = {}
    if os.path.exists(args.out):
        for r in read_complete_lines(args.out):
            done[r["hand_id"]] = r
    todo = [(h, e) for h, e in entries if h.hand_id not in done]
    if todo and not args.summary_only:
        from negpluribus import fast
        from negpluribus.fast.tables import table_name
        from negpluribus.fast.blueprint import load_blueprint
        from negpluribus.fast.trainer import core_bucketer

        core = fast.core()
        bk = load_bucketer(args.buckets)
        spec = GameSpec(n_players=2, stack_bb=args.stack_bb, max_street=Street.RIVER, preflop_fracs=fracs(args.preflop_fracs),
                        postflop_fracs=fracs(args.postflop_fracs), max_raises_per_street=args.max_raises,
                        n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
        cbk = core_bucketer(bk)
        tpath = os.path.join(args.tables, table_name(cbk))
        tables = core.BucketTables()
        if os.path.exists(tpath):
            tables.load(tpath, cbk)
        else:
            if args.cache:
                cbk.load_cache(args.cache)
            t = time.time()
            tables, info = core.aivat_build_tables(cbk, args.threads, [3, 1, 2])
            os.makedirs(args.tables, exist_ok=True)
            tables.save(tpath)
            print(f"bucket tables built in {time.time() - t:.0f}s ({info}): {tpath}", flush=True)
        tb = core.TabulatedBucketer(cbk, tables)
        models = {}  # blueprint (the manifest's path; None: --blueprint) -> FastAivat on it

        def model(key):
            """The known player's model on one blueprint (built when a hand first needs it): its game and root table."""
            ev = models.get(key)
            if ev is not None:
                return ev
            if key is None:
                bp = load_blueprint(args.blueprint, backend="cpp", n_players=2)
                rc = args.root_cache or os.path.splitext(args.out)[0] + f".root_k{V1['root_rollouts']}_s{V1['seed']}.npz"
                name = os.path.basename(args.blueprint)
            else:
                point = grid.find(key)
                bp = load_blueprint(grid.resolve(key), backend="cpp", n_players=2)
                grid.check_blueprint(point, bp, cbk.identity)
                name = os.path.basename(key)
                stem = os.path.splitext(name)[0]
                stem = stem[len("blueprint_"):] if stem.startswith("blueprint_") else stem
                root_dir = args.root_dir or os.path.dirname(os.path.abspath(args.out))
                rc = os.path.join(root_dir, f"root_v1_{stem}_{args.stack_bb}bb.npz")
            game = make_game(spec, tb, bp)
            t = time.time()
            rt = root_table(game, V1["root_rollouts"], seed=V1["seed"], threads=args.threads, cache_path=rc,
                            identity={"blueprint": name, "buckets": os.path.basename(args.buckets),
                                      "grid": [args.preflop_fracs, args.postflop_fracs, args.max_raises, args.stack_bb]})
            print(f"root table{'' if key is None else ' of ' + name}: {rt.n_classes:,} classes, means {rt.mean[0]:+.2f} / "
                  f"{rt.mean[1]:+.2f} chips (SB / BB), {time.time() - t:.0f}s ({rc})", flush=True)
            ev = models[key] = FastAivat(game, V1["rollouts"], V1["eq_samples"], V1["seed"], rt)
            return ev

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        t_start = time.time()
        n_done = 0
        with open(args.out, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.chunk):
                part = todo[i: i + args.chunk]
                by_model = {}
                for j, (_, e) in enumerate(part):
                    by_model.setdefault(e.get("blueprint"), []).append(j)
                res = [None] * len(part)
                for key, idx in by_model.items():
                    for j, r in zip(idx, model(key).evaluate_many([part[j][0] for j in idx], args.threads)):
                        res[j] = r
                for (h, e), r in zip(part, res):
                    if r["trace"]:
                        raise RuntimeError(f"hand {h.hand_id}: {r['trace'][0][0]}")
                    if r["net"] != h.net:
                        raise RuntimeError(f"hand {h.hand_id}: evaluator net {r['net']} != logged {h.net}")
                    terms = {}
                    for k, _, _, v in r["terms"]:
                        terms[TERM_NAMES[k]] = terms.get(TERM_NAMES[k], 0.0) + v
                    row = {"hand_id": h.hand_id, "position": e["position"], "net": h.net, "value": r["value"], "base": r["base"],
                           "terms": terms, "seconds": r["seconds"], "rollouts": r["rollouts"]}
                    if "deal" in e:
                        row["deal"] = e["deal"]
                    if "point" in e:
                        row["point"], row["blueprint"] = e["point"], e["blueprint"]
                    f.write(json.dumps(row) + "\n")
                    done[h.hand_id] = row
                f.flush()
                n_done += len(part)
                el = time.time() - t_start
                print(f"  {n_done:,}/{len(todo):,} hands, {el:.0f}s, {el / n_done * args.threads:.3f} thread-s per hand, "
                      f"eta {el / n_done * (len(todo) - n_done) / 60:.0f} min", flush=True)
    rows = [done[h.hand_id] for h, _ in entries if h.hand_id in done]
    if len(rows) < len(entries):
        print(f"only {len(rows):,} of {len(entries):,} hands scored so far")
        entries = [(h, e) for h, e in entries if h.hand_id in done]
    for r in rows:
        for k in TERM_NAMES.values():
            r["terms"].setdefault(k, 0.0)
    luck = None
    groups = None
    if args.duel:
        groups = [e["deal"] for _, e in entries]
        if all(e["luck_bb"] is not None for _, e in entries):
            luck = [h.net / 100 - e["luck_bb"] for h, e in entries]
    elif not args.no_luck:
        t = time.time()
        luck = [x / 100 for x in card_luck_slumbot(entries, os.path.splitext(args.out)[0] + ".luck.json")]
        print(f"card-luck correction: {time.time() - t:.0f}s")
    print(summary(rows, luck, groups))
    if args.first:
        k = min(args.first, len(rows))
        print()
        print(summary(rows[:k], None if luck is None else luck[:k], None if groups is None else groups[:k],
                      label=f"the first {k:,} hands: "))
    by_pos = {}
    for r in rows:
        by_pos.setdefault(r["position"], []).append(r)
    for pos, rs in sorted(by_pos.items()):
        s1, s2 = stats([r["net"] / 100 for r in rs]), stats([r["value"] / 100 for r in rs])
        print(f"  as {pos}: {len(rs):,} hands, raw {s1['bb100']:+.1f} +/- {s1['ci95']:.1f}, AIVAT {s2['bb100']:+.1f} +/- {s2['ci95']:.1f}")
    if grid is not None:  # per grid point (hands, not deals: the two hands of a deal may sit at different points)
        by_point = {}
        for r in rows:
            by_point.setdefault(r.get("point"), []).append(r)
        for p, rs in sorted(by_point.items(), key=lambda x: (x[0] is None, x[0] or 0)):
            s1, s2 = stats([r["net"] / 100 for r in rs]), stats([r["value"] / 100 for r in rs])
            print(f"  at the {p}bb point: {len(rs):,} hands, raw {s1['bb100']:+.1f} +/- {s1['ci95']:.1f}, "
                  f"AIVAT {s2['bb100']:+.1f} +/- {s2['ci95']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
