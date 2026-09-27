"""Where a hero wins or loses: the per-hand logs of scripts/eval_archetypes.py --log-hands, by category.

    python scripts/hand_log_report.py data/overbet_search_hands.jsonl [--paired data/overbet_blueprint_hands.jsonl]

Categories: the street of the opponent's first off-grid overbet (a raise, not all-in, whose raise
increment is above --overbet-frac x the largest grid size of its street), else "no overbet"; and,
for the search agent, whether the hero searched preflop.  Per category: hands, the hero's bb/100
raw and without its card luck (the chance-node correction), with 95% CIs (hands as samples).
--paired: the same categories for another log of the same deals and seeds (for example the blueprint
agent): per category the mean difference over the hands both logs have, per hand (same deal, same
seat), which removes the card luck the two share.  The category is the first log's.

--breakdown (with --paired): the paired difference split where it can arise.  With the same seeds
the search agent plays its unsearched preflop decisions exactly as the blueprint agent (tested), so a
pair of hands is identical up to the search agent's first search; the hands are split by that first
search (none, preflop, from the flop after a preflop overbet the blueprint translated, from the flop
without one) and by the street where the search agent's hand was decided (the last action: a fold on
that street, or the showdown).  Per part: hands, the mean difference per 100 of its hands, and its
share of the total per 100 hands of the log (the shares add up to the total), each with a 95%
interval (hands as samples), raw and luck-corrected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.cards import Deck  # noqa: E402
from negpluribus.engine import Action, ActionType, HandState, Street  # noqa: E402

STREETS = ["preflop", "flop", "turn", "river"]


def read(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def overbet_street(row: dict, top: Tuple[float, float], frac: float, stack: int, sb: int, bb: int) -> Optional[int]:
    """The street of the opponent's first off-grid overbet in this hand, None if there was none."""
    st = HandState([stack, stack], row["button"], sb, bb, 0, deck=Deck.from_order(list(range(52))), max_street=Street.RIVER)
    hero = row["hero_seat"]
    for street, seat, typ, amount in row["events"]:
        obs = st.observe(seat)
        if seat != hero and typ == int(ActionType.RAISE) and amount < obs.max_raise_to:
            inc = amount - obs.street_bets_max()
            if inc / (obs.pot + obs.to_call) > frac * (top[0] if street == 0 else top[1]):
                return street
        st.apply(Action(ActionType(typ), amount))
    return None


def first_search(row: dict, overbet: Optional[int]) -> str:
    first = next((h for h in row.get("hero", []) if h.get("played") == "search"), None)
    if first is None:
        return "no search (identical to the pair)"
    if first.get("street") == 0:
        return "preflop search"
    if overbet == 0:
        return "search from the flop, after a preflop overbet"
    if overbet is None:
        return "search from the flop, no overbet at all"
    return "search from the flop, overbet after the preflop"


def decided(row: dict) -> str:
    if not row["events"]:
        return "?"
    street, _, typ, _ = row["events"][-1]
    return f"{STREETS[street]} {'fold' if typ == int(ActionType.FOLD) else 'showdown'}"


def breakdown(rows: List[dict], other: Dict[Tuple[str, int, int], dict], top, frac, stack, sb, bb) -> None:
    pairs = []
    for r in rows:
        o = other.get((r["opponent"], r["deal"], r["hero_seat"]))
        if o is None:
            continue
        ob = overbet_street(r, top, frac, stack, sb, bb)
        raw = r["net_bb"] - o["net_bb"]
        cor = None
        if r.get("luck_bb") is not None and o.get("luck_bb") is not None:
            cor = (r["net_bb"] - r["luck_bb"]) - (o["net_bb"] - o["luck_bb"])
        pairs.append((first_search(r, ob), decided(r), raw, cor))
    n = len(pairs)
    has_cor = all(p[3] is not None for p in pairs)

    def part(label: str, pick) -> str:
        sel = [p for p in pairs if pick(p)]
        m_raw, c_raw = ci([p[2] for p in sel])
        s_raw, sc_raw = ci([p[2] if pick(p) else 0.0 for p in pairs])
        out = f"  {label:<52} {len(sel):>5} hands | per 100 of them {m_raw:+8.1f} +/-{c_raw:6.1f} | share {s_raw:+7.1f} +/-{sc_raw:5.1f}"
        if has_cor:
            m_c, c_c = ci([p[3] for p in sel])
            s_c, sc_c = ci([p[3] if pick(p) else 0.0 for p in pairs])
            out += f" || luck-corrected: per 100 {m_c:+8.1f} +/-{c_c:6.1f} | share {s_c:+7.1f} +/-{sc_c:5.1f}"
        return out

    print(f"paired per hand: {n} hands")
    print(part("all", lambda p: True))
    kinds = sorted({p[0] for p in pairs}, key=lambda k: -sum(1 for p in pairs if p[0] == k))
    print("by the search agent's first search:")
    for k in kinds:
        print(part(k, lambda p, k=k: p[0] == k))
    print("by the street where the search agent's hand was decided:")
    for d in sorted({p[1] for p in pairs}, key=lambda d: (STREETS.index(d.split()[0]) if d != "?" else 9, d)):
        print(part(d, lambda p, d=d: p[1] == d))
    print("both (parts with 20 hands or more):")
    for k in kinds:
        for d in sorted({p[1] for p in pairs if p[0] == k}, key=lambda d: (STREETS.index(d.split()[0]) if d != "?" else 9, d)):
            if sum(1 for p in pairs if p[0] == k and p[1] == d) >= 20:
                print(part(f"{k}; {d}", lambda p, k=k, d=d: p[0] == k and p[1] == d))


def ci(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m * 100, float("inf")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m * 100, 1.96 * sd / math.sqrt(n) * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--paired", default=None)
    ap.add_argument("--top", default="3.0,4.0", help="largest grid size preflop,postflop (pot fractions)")
    ap.add_argument("--overbet-frac", type=float, default=1.5, help="an overbet: above this x the largest grid size")
    ap.add_argument("--stack", type=int, default=20000)
    ap.add_argument("--sb", type=int, default=50)
    ap.add_argument("--bb", type=int, default=100)
    ap.add_argument("--breakdown", action="store_true", help="with --paired: split the paired difference (see above)")
    args = ap.parse_args()
    top = tuple(float(x) for x in args.top.split(","))
    rows = read(args.log)
    other: Dict[Tuple[str, int, int], dict] = {}
    if args.paired:
        for r in read(args.paired):
            other[(r["opponent"], r["deal"], r["hero_seat"])] = r
    if args.breakdown:
        if not other:
            ap.error("--breakdown needs --paired")
        breakdown(rows, other, top, args.overbet_frac, args.stack, args.sb, args.bb)
        return 0
    cats: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        s = overbet_street(r, top, args.overbet_frac, args.stack, args.sb, args.bb)
        cat = "no overbet" if s is None else f"overbet {STREETS[s]}"
        pre = [h for h in r.get("hero", []) if h.get("street") == 0 and h.get("played") == "search"]
        if pre:
            cat += ", hero searched preflop"
        cats[cat].append(r)
    total = [r["net_bb"] for r in rows]
    m, c = ci(total)
    print(f"{args.log}: {len(rows)} hands, {m:+.1f} bb/100 +/-{c:.1f}", end="")
    if all(r.get("luck_bb") is not None for r in rows):
        mc, cc = ci([r["net_bb"] - r["luck_bb"] for r in rows])
        print(f"; without card luck {mc:+.1f} +/-{cc:.1f}", end="")
    print()
    for cat in sorted(cats, key=lambda k: -len(cats[k])):
        rs = cats[cat]
        m, c = ci([r["net_bb"] for r in rs])
        line = f"  {cat:<44} {len(rs):>6} hands: {m:+9.1f} +/-{c:7.1f}"
        if all(r.get("luck_bb") is not None for r in rs):
            mc, cc = ci([r["net_bb"] - r["luck_bb"] for r in rs])
            line += f" | no luck {mc:+9.1f} +/-{cc:7.1f}"
        # share of the whole: the category's contribution to the overall bb/100
        line += f" | contributes {sum(r['net_bb'] for r in rs) / max(1, len(rows)) * 100:+.1f} bb/100"
        if other:
            pairs = [(r, other.get((r["opponent"], r["deal"], r["hero_seat"]))) for r in rs]
            pairs = [(a, b) for a, b in pairs if b is not None]
            if pairs:
                md, cd = ci([a["net_bb"] - b["net_bb"] for a, b in pairs])
                line += f" | paired diff {md:+8.1f} +/-{cd:6.1f} ({len(pairs)} hands; contributes " \
                        f"{sum(a['net_bb'] - b['net_bb'] for a, b in pairs) / max(1, len(rows)) * 100:+.1f})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
