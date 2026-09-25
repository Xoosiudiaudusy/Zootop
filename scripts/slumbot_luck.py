"""Card-luck correction on the Slumbot log (Slumbot shows both hands every hand).

For our value v = equity(our hand vs Slumbot's hand | board) x pot - our contribution, every card
deal is a chance node with E[v after | before] = v before (equity is a martingale over the next
cards), so the correction (eq_after - eq_before) x pot_at_the_deal has mean zero and removing it
keeps the estimate unbiased (the chance-node part of AIVAT, Burch et al. 2018; no model of either
player's strategy is needed).  Compared: raw winnings; the all-in-before-river adjustment only;
the full card-luck correction.
"""
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from negpluribus.slumbot.protocol import parse_cards, split_action  # noqa: E402

try:
    from negpluribus import _fastcore as _fc
    evaluate = _fc.evaluate
except Exception:  # pragma: no cover
    from negpluribus.evaluator import evaluate

STACK, SB, BB = 20000, 50, 100


def eq_exact(h, o, board):
    """Our equity vs the known opponent hand, averaging over all remaining board cards (turn/river)."""
    dead = set(h) | set(o) | set(board)
    rest = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    tot = 0.0
    n = 0
    if need == 0:
        a, b = evaluate(list(h) + board), evaluate(list(o) + board)
        return 1.0 if a > b else (0.5 if a == b else 0.0)
    if need == 1:
        for c in rest:
            bd = board + [c]
            a, b = evaluate(list(h) + bd), evaluate(list(o) + bd)
            tot += 1.0 if a > b else (0.5 if a == b else 0.0)
            n += 1
        return tot / n
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            bd = board + [rest[i], rest[j]]
            a, b = evaluate(list(h) + bd), evaluate(list(o) + bd)
            tot += 1.0 if a > b else (0.5 if a == b else 0.0)
            n += 1
    return tot / n


def eq_preflop(h, o, rng, samples=4000):
    dead = set(h) | set(o)
    rest = [c for c in range(52) if c not in dead]
    tot = 0.0
    for _ in range(samples):
        bd = rng.sample(rest, 5)
        a, b = evaluate(list(h) + bd), evaluate(list(o) + bd)
        tot += 1.0 if a > b else (0.5 if a == b else 0.0)
    return tot / samples


def pots_by_street(action, hero_is_sb):
    """Pot (both players' chips) at the start of each street that was dealt, and whether both were
    all-in before the river (and on which street), from Slumbot's action string."""
    streets = split_action(action)
    total = {"sb": SB, "bb": BB}   # chips in the pot per player, including the current street
    street_bet = {"sb": SB, "bb": BB}
    pots = [SB + BB]               # pot when the hole cards are dealt
    allin_street = None
    for si, moves in enumerate(streets):
        if si > 0:
            pots.append(total["sb"] + total["bb"])   # pot when this street's cards were dealt
            street_bet = {"sb": 0, "bb": 0}
        actor = "sb" if si == 0 else "bb"
        for m in moves:
            other = "bb" if actor == "sb" else "sb"
            if m.code == "c":
                add = min(street_bet[other] - street_bet[actor], STACK - total[actor])
                street_bet[actor] += add
                total[actor] += add
            elif m.code == "b":
                add = m.amount - street_bet[actor]
                street_bet[actor] = m.amount
                total[actor] += add
            elif m.code == "f":
                return pots, None, total
            actor = other
        if total["sb"] >= STACK and total["bb"] >= STACK and allin_street is None and si < 3:
            allin_street = si
    return pots, allin_street, total


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    hands = [r for r in rows if "winnings" in r and r.get("bot_cards")]
    rng = random.Random(7)
    raw, allin_adj, luck_adj = [], [], []
    n_allin = 0
    for r in hands:
        h, o = parse_cards(r["hero_cards"]), parse_cards(r["bot_cards"])
        board = parse_cards(r.get("board") or [])
        hero_is_sb = r["client_pos"] == 1
        me = "sb" if hero_is_sb else "bb"
        w = r["winnings"]
        pots, allin_street, total = pots_by_street(r["action"], hero_is_sb)
        raw.append(w)
        # equities at each deal that happened: preflop (hole cards), flop, turn, river
        eqs = [eq_preflop(h, o, rng)]
        for k, nb in ((1, 3), (2, 4), (3, 5)):
            if len(pots) > k and len(board) >= nb:
                eqs.append(eq_exact(h, o, board[:nb]))
        # card-luck correction: deal of the hole cards (expected equity 0.5), then each street
        corr = (eqs[0] - 0.5) * pots[0]
        for k in range(1, len(eqs)):
            corr += (eqs[k] - eqs[k - 1]) * pots[k]
        luck_adj.append(w - corr)
        # all-in-before-river adjustment only
        if allin_street is not None:
            n_allin += 1
            eq_at = eqs[allin_street] if allin_street < len(eqs) else eqs[-1]
            pot = total["sb"] + total["bb"]
            allin_adj.append(eq_at * pot - total[me])
        else:
            allin_adj.append(w)
    n = len(raw)
    print(f"hands with both hands known: {n}; all-in before the river: {n_allin}")
    print(f"{'estimate':34} {'bb/100':>9} {'sd per hand, bb':>16} {'95% CI, bb/100':>15} {'hands for +/-10':>16} {'for +/-5':>10}")
    for name, xs in (("raw winnings", raw), ("all-in before river adjusted", allin_adj), ("card-luck corrected (chance nodes)", luck_adj)):
        m = st.mean(xs)  # chips per hand = bb/100
        sd = st.stdev(xs) / 100
        ci = 1.96 * sd / n ** 0.5 * 100
        need10 = (1.96 * sd * 100 / 10) ** 2
        need5 = (1.96 * sd * 100 / 5) ** 2
        print(f"{name:34} {m:+9.1f} {sd:16.2f} {ci:15.1f} {need10:16,.0f} {need5:10,.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/slumbot/hunl200w_pot16_s0.jsonl")
