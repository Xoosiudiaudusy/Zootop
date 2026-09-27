"""How a hero answered the opponent's postflop overbets, next to the blueprint agent on the same hands.

    python scripts/overbet_answers.py data/hands_search.jsonl --paired data/hands_blueprint.jsonl [--seed 0]

Both logs come from scripts/eval_archetypes.py --log-hands with the same --seed (the same decks and
agent seeds).  The hands: those where the first log's hero faced the opponent's first postflop
overbet (a raise, not all-in, above --overbet-frac x the largest grid size of its street) and made
its first search after the preflop.  For each such hand, in each log: the hero's answer to that
overbet (fold, call, raise) and whether it was behind the opponent's actual hand then (exact equity
below 1/2); for the blueprint agent also the grid size its translation read the overbet as (the
agent's randomized pseudo-harmonic translation, replayed from its seed: the largest size or the
all-in).  In the paired log the hand can have gone another way before the overbet (the search agent
searched earlier on the flop): then "no overbet faced".  Last: the paired difference of the hero's
results in these hands, raw and luck-corrected, per 100 of them and as a share of all hands of the
log, 95% intervals (hands as samples).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import zlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import BetGrid  # noqa: E402
from negpluribus.cards import Deck  # noqa: E402
from negpluribus.engine import Action, ActionType, HandState, Street  # noqa: E402
from negpluribus.fast import core  # noqa: E402

BOARD = {1: 3, 2: 4, 3: 5}


def read(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def first_postflop_overbet(row: dict, grid: BetGrid, frac: float, stack: int, sb: int, bb: int):
    """(event index, street, Event) of the opponent's first postflop overbet, and the hero's answer
    (type) to it, or None."""
    st = HandState([stack, stack], row["button"], sb, bb, 0, deck=Deck.from_order(list(range(52))), max_street=Street.RIVER)
    hero = row["hero_seat"]
    found = None
    for i, (street, seat, typ, amount) in enumerate(row["events"]):
        obs = st.observe(seat)
        ev = st.apply(Action(ActionType(typ), amount))
        if found is None and street >= 1 and seat != hero and typ == int(ActionType.RAISE) and amount < obs.max_raise_to:
            inc = amount - obs.street_bets_max()
            if inc / (obs.pot + obs.to_call) > frac * max(grid.fracs_for(Street(street))):
                found = (i, street, ev)
                continue
        if found is not None and seat == hero:
            return found + (typ,)
    return found + (None,) if found is not None else None


def blueprint_reading(row: dict, i: int, ev, grid: BetGrid, seed: int) -> str:
    """The name the blueprint agent's translation gave event i in this hand (its seeds replayed)."""
    s = (seed * 7_919 + row["deal"] * 31 + row["hero_seat"]) ^ zlib.crc32(b"hero")
    nonce = random.Random(s).getrandbits(32)
    return grid.from_concrete(ev, ev.all_in, random.Random(nonce * 1_000_003 + i))


def ci(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] * 100 if xs else float("nan")), float("inf")
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m * 100, 1.96 * sd / math.sqrt(n) * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--paired", required=True, help="the blueprint agent's log of the same deals and seeds")
    ap.add_argument("--seed", type=int, default=0, help="the --seed of both runs")
    ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--overbet-frac", type=float, default=1.5)
    ap.add_argument("--stack", type=int, default=20000)
    ap.add_argument("--sb", type=int, default=50)
    ap.add_argument("--bb", type=int, default=100)
    args = ap.parse_args()
    grid = BetGrid(preflop_fracs=tuple(float(x) for x in args.preflop_fracs.split(",")),
                   postflop_fracs=tuple(float(x) for x in args.postflop_fracs.split(",")),
                   max_raises_per_street=args.max_raises)
    c = core()
    names = {int(ActionType.FOLD): "fold", int(ActionType.CALL): "call", int(ActionType.RAISE): "raise", None: "(no answer)"}
    a_rows = read(args.log)
    b_rows = {(r["deal"], r["hero_seat"]): r for r in read(args.paired)}
    a_ans: Counter = Counter()
    b_ans: Counter = Counter()
    b_read: Counter = Counter()
    diffs_raw, diffs_cor, n_all = [], [], len(a_rows)
    for r in a_rows:
        f = first_postflop_overbet(r, grid, args.overbet_frac, args.stack, args.sb, args.bb)
        if f is None:
            continue
        first = next((h for h in r.get("hero", []) if h.get("played") == "search"), None)
        if first is not None and first.get("street") == 0:
            continue  # a preflop search: another part (the hands diverged before the flop)
        i, street, ev, answer = f
        hs = r["hero_seat"]
        eq = c.equity_vs_hand(r["holes"][hs], r["holes"][1 - hs], r["board"][: BOARD[street]])
        a_ans[(names[answer], "behind" if eq < 0.5 else "ahead")] += 1
        o = b_rows.get((r["deal"], hs))
        if o is None:
            continue
        g = first_postflop_overbet(o, grid, args.overbet_frac, args.stack, args.sb, args.bb)
        if g is None:
            b_ans[("no overbet faced", "")] += 1
        else:
            j, st2, ev2, ans2 = g
            eq2 = c.equity_vs_hand(o["holes"][hs], o["holes"][1 - hs], o["board"][: BOARD[st2]])
            b_ans[(names[ans2], "behind" if eq2 < 0.5 else "ahead")] += 1
            b_read[(blueprint_reading(o, j, ev2, grid, args.seed), names[ans2])] += 1
        diffs_raw.append(r["net_bb"] - o["net_bb"])
        if r.get("luck_bb") is not None and o.get("luck_bb") is not None:
            diffs_cor.append((r["net_bb"] - r["luck_bb"]) - (o["net_bb"] - o["luck_bb"]))

    def show(title: str, cnt: Counter) -> None:
        tot = sum(cnt.values())
        print(f"{title} ({tot} hands): " + ", ".join(f"{k[0]}{' ' + k[1] if k[1] else ''} {v}" for k, v in sorted(cnt.items())))

    show(f"{os.path.basename(args.log)}: its answer to the first postflop overbet", a_ans)
    show(f"{os.path.basename(args.paired)} in the same hands", b_ans)
    print("   the blueprint agent's reading of the overbet -> its answer: "
          + ", ".join(f"{k[0]} -> {k[1]} {v}" for k, v in sorted(b_read.items())))
    m, cc = ci(diffs_raw)
    share = sum(diffs_raw) / max(1, n_all) * 100
    line = f"paired difference in these {len(diffs_raw)} hands: {m:+.1f} +/-{cc:.1f} per 100 of them (share {share:+.1f} of the log's bb/100)"
    if diffs_cor and len(diffs_cor) == len(diffs_raw):
        mc, ccc = ci(diffs_cor)
        line += f"; luck-corrected {mc:+.1f} +/-{ccc:.1f} (share {sum(diffs_cor) / max(1, n_all) * 100:+.1f})"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
