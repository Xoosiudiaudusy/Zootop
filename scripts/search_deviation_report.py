"""Where the search agent left the blueprint's line, and what it cost: a duel hand log of the search
agent (scripts/eval_archetypes.py --agent search --log-hands) next to the blueprint agent's log on the
same deals (E0), both scored by AIVAT (scripts/aivat_eval.py, per-hand values keyed by (deal, position)).

For every hero decision the blueprint's policy at the hero's key is looked up; a decision is a "clear
deviation" when the blueprint gives the search's action less than --clear probability.  Hands are
grouped by the street of the first clear deviation (none / flop / turn / river); per group: hands, the
hero's AIVAT mean, the paired difference against the blueprint hero on the same (deal, seat), and the
group's contribution to the total paired difference (contributions add up to the total).

    python scripts/search_deviation_report.py --search-hands H1 --search-aivat A1 --base-hands H0 --base-aivat A0 \
        --blueprint ... --buckets ... [--tables DIR] [--clear 0.1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.abstraction.infoset import infoset_key  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.engine import Action, ActionType, Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402

STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def load_hands(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[(r["deal"], r["hero_seat"])] = r
    return out


def load_aivat(path, hands, known="villain"):
    """The hero's AIVAT value per (deal, hero seat) in bb.  The evaluator's lines carry the deal and the
    KNOWN player's position and value; with the villain known (a duel search vs blueprint) the hero's
    value is minus the villain's, and the villain's position is the other one (heads-up: the button
    posts the small blind)."""
    out = {}
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by_deal = defaultdict(list)
    for r in rows:
        by_deal[r["deal"]].append(r)
    for (deal, seat), h in hands.items():
        hero_pos = "SB" if seat == h["button"] else "BB"
        pos = hero_pos if known == "hero" else ("BB" if hero_pos == "SB" else "SB")
        sign = 1.0 if known == "hero" else -1.0
        for r in by_deal.get(deal, []):
            if r["position"] == pos:
                out[(deal, seat)] = sign * r["value"] / 100.0  # chips -> bb (bb = 100)
    return out


def ci(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return 1.96 * sd / math.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-hands", required=True)
    ap.add_argument("--search-aivat", required=True)
    ap.add_argument("--base-hands", required=True)
    ap.add_argument("--base-aivat", required=True)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--clear", type=float, default=0.1, help="a deviation is clear below this blueprint probability")
    ap.add_argument("--known", choices=("villain", "hero"), default="villain", help="the AIVAT files' known player")
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
    if args.tables:
        from negpluribus.fast.tables import tabulated
        from negpluribus.fast.trainer import core_bucketer
        bk = tabulated(core_bucketer(bk), args.tables)
    bp = load_blueprint(args.blueprint)
    grid = spec.grid
    sh, bh = load_hands(args.search_hands), load_hands(args.base_hands)
    sa, ba = load_aivat(args.search_aivat, sh, args.known), load_aivat(args.base_aivat, bh, args.known)
    groups = defaultdict(list)       # first clear deviation street -> [(search value, base value)]
    kinds = defaultdict(list)        # "street: taken kind (blueprint's most likely kind)" -> the same pairs

    def kind(name):
        return "fold" if name.startswith("f") else ("check/call" if name == "c" else "bet/raise")
    per_decision = defaultdict(lambda: [0, 0, 0.0])  # street -> [decisions, clear deviations, sum of bp prob of the action]
    for key, h in sh.items():
        if key not in bh or key not in sa or key not in ba:
            continue
        seat = h["hero_seat"]
        holes, board = h["holes"], h["board"]
        used = set(holes[0]) | set(holes[1]) | set(board)
        order = list(holes[0]) + list(holes[1]) + list(board) + [c for c in range(52) if c not in used]
        st = spec.new_hand(order, button=h["button"])
        first = None
        for street, actor, t, amount in h["events"]:
            if actor == seat and street > 0:
                obs = st.observe(seat)
                legal = grid.abstract_actions(obs)
                k = infoset_key(obs, bk, grid)
                probs = bp.policy(k, legal)
                taken = Action(ActionType(t), amount)
                name = next((nm for nm in legal if grid.to_concrete(obs, nm) == taken), None)  # the search plays grid actions
                p = None
                if probs is not None and name is not None:
                    p = probs[legal.index(name)]
                per_decision[street][0] += 1
                if p is not None:
                    per_decision[street][2] += p
                    if p < args.clear:
                        per_decision[street][1] += 1
                        if first is None:
                            first = street
                            best = legal[max(range(len(legal)), key=lambda i: probs[i])]
                            first_kind = f"{STREETS[street]}: {kind(name)} (blueprint {kind(best)} {max(probs):.2f})"
            st.apply(Action(ActionType(t), amount))
        groups["none" if first is None else STREETS[first]].append((sa[key], ba[key]))
        if first is not None:
            kinds[first_kind].append((sa[key], ba[key]))
    total = [s - b for g in groups.values() for s, b in g]
    n_all = len(total)
    print(f"paired hands {n_all:,}; search AIVAT {100 * sum(s for g in groups.values() for s, _ in g) / n_all:+.1f} bb/100; "
          f"blueprint hero {100 * sum(b for g in groups.values() for _, b in g) / n_all:+.1f}; "
          f"paired difference {100 * sum(total) / n_all:+.1f} +/- {100 * ci(total):.1f}")
    print(f"{'first clear deviation':22} {'hands':>6} {'search, bb/100':>15} {'paired diff':>12} {'95%':>7} {'share of total':>15}")
    for g in ("none", "flop", "turn", "river"):
        xs = groups.get(g, [])
        if not xs:
            continue
        d = [s - b for s, b in xs]
        print(f"{g:22} {len(xs):6,} {100 * sum(s for s, _ in xs) / len(xs):+15.1f} {100 * sum(d) / len(d):+12.1f} {100 * ci(d):7.1f} "
              f"{100 * sum(d) / n_all:+15.1f}")
    print(f"{'first clear deviation, by kind':58} {'hands':>6} {'paired diff':>12} {'95%':>7} {'share':>7}")
    for g, xs in sorted(kinds.items(), key=lambda kv: sum(s - b for s, b in kv[1])):
        d = [s - b for s, b in xs]
        if len(d) >= 10:
            print(f"{g:58} {len(xs):6,} {100 * sum(d) / len(d):+12.1f} {100 * ci(d):7.1f} {100 * sum(d) / n_all:+7.1f}")
    print("per street: hero decisions, clear deviations, mean blueprint probability of the search's action")
    for s in (1, 2, 3):
        n, c, sp = per_decision[s]
        if n:
            print(f"  {STREETS[s]:6} {n:6,} {c:6,} ({100 * c / n:.1f}%)  {sp / n:.3f}")


if __name__ == "__main__":
    main()
