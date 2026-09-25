"""Vanilla CFR on Kuhn poker: the smallest complete example of the Pluribus core idea.

Kuhn poker: 3 cards (J, Q, K), each player antes 1, one card each.
P1 may check or bet 1; facing a bet you may fold or call; check-check goes to
showdown.  The Nash value for P1 is -1/18 ≈ -0.0556.

Counterfactual Regret Minimization in one paragraph
---------------------------------------------------
Walk the game tree.  At each information set (what the acting player knows:
own card + betting history) keep a table of *regrets*: how much better each
action would have done than the strategy we actually used, weighted by how
likely the *opponents* (and chance) were to bring us here (counterfactual
reach).  The current strategy is "regret matching": play each action
proportionally to its positive regret.  Average the strategies over
iterations; the average converges to a Nash equilibrium.  Pluribus is this
idea + sampling (MCCFR) + abstraction + real-time search.

Run:  python -m negpluribus.cfr.kuhn
"""
from __future__ import annotations

import random
from typing import Dict, List

from .strategy import TabularStrategy

CARDS = ["J", "Q", "K"]
ACTIONS = ["p", "b"]  # pass(check/fold) / bet(call)


class Node:
    __slots__ = ("regret", "strategy_sum")

    def __init__(self) -> None:
        self.regret = [0.0, 0.0]
        self.strategy_sum = [0.0, 0.0]

    def strategy(self, reach: float) -> List[float]:
        pos = [max(0.0, r) for r in self.regret]
        s = sum(pos)
        strat = [p / s for p in pos] if s > 0 else [0.5, 0.5]
        for i in range(2):
            self.strategy_sum[i] += reach * strat[i]
        return strat

    def average(self) -> List[float]:
        s = sum(self.strategy_sum)
        return [x / s for x in self.strategy_sum] if s > 0 else [0.5, 0.5]


def is_terminal(h: str) -> bool:
    return h in ("pp", "bb", "bp", "pbp", "pbb")


def payoff(h: str, cards: List[str]) -> float:
    """Utility for the player who acted first (P1)."""
    p1_wins = CARDS.index(cards[0]) > CARDS.index(cards[1])
    if h == "pp":
        return 1.0 if p1_wins else -1.0
    if h == "bp":  # P1 bet, P2 folded
        return 1.0
    if h == "pbp":  # P1 checked, P2 bet, P1 folded
        return -1.0
    return 2.0 if p1_wins else -2.0  # bb / pbb : showdown for 2


def cfr(nodes: Dict[str, Node], cards: List[str], h: str, p1: float, p2: float) -> float:
    """Returns expected utility for P1 at history h, updating regrets on the way."""
    player = len(h) % 2
    if is_terminal(h):
        return payoff(h, cards)
    info = cards[player] + h
    node = nodes.setdefault(info, Node())
    strat = node.strategy(p1 if player == 0 else p2)
    util = [0.0, 0.0]
    node_util = 0.0
    for i, a in enumerate(ACTIONS):
        if player == 0:
            util[i] = cfr(nodes, cards, h + a, p1 * strat[i], p2)
        else:
            util[i] = cfr(nodes, cards, h + a, p1, p2 * strat[i])
        node_util += strat[i] * util[i]
    for i in range(2):
        regret = util[i] - node_util
        if player == 0:
            node.regret[i] += p2 * regret
        else:
            node.regret[i] += p1 * (-regret)  # P2 utility is -P1 utility
    return node_util


def train(iterations: int = 20000, seed: int = 0) -> tuple[TabularStrategy, float]:
    rng = random.Random(seed)
    nodes: Dict[str, Node] = {}
    total = 0.0
    for _ in range(iterations):
        cards = CARDS[:]
        rng.shuffle(cards)
        total += cfr(nodes, cards[:2], "", 1.0, 1.0)
    strat = TabularStrategy(ACTIONS, {k: n.average() for k, n in nodes.items()})
    return strat, total / iterations


def best_response_value(strat: TabularStrategy, br_player: int) -> float:
    """Exact best-response value (utility for P1) when ``br_player`` best-responds to ``strat``.

    The best responder picks, at each of its information sets (own card + history),
    the action that maximises value summed over the opponent's possible cards,
    each weighted by how likely the opponent's strategy is to reach that history.
    """

    def reach(h: str, opp_card: str) -> float:
        pr = 1.0
        for i in range(len(h)):
            if i % 2 != br_player:
                probs = strat.policy(opp_card + h[:i])
                pr *= probs[ACTIONS.index(h[i])]
        return pr

    def infoset_value(h: str, my_card: str) -> float:
        player = len(h) % 2
        if is_terminal(h):
            tot = 0.0
            for oc in CARDS:
                if oc == my_card:
                    continue
                cards = [my_card, oc] if br_player == 0 else [oc, my_card]
                tot += payoff(h, cards) * reach(h, oc)
            return tot
        if player == br_player:
            vals = [infoset_value(h + a, my_card) for a in ACTIONS]
            return max(vals) if br_player == 0 else min(vals)
        return sum(infoset_value(h + a, my_card) for a in ACTIONS)

    total = sum(infoset_value("", c) for c in CARDS)
    return total / 6.0  # 6 equally likely deals


def exploitability(strat: TabularStrategy) -> float:
    """How much a best responder gains over the game value, averaged over both seats."""
    game_value = -1.0 / 18.0
    br_vs_p1 = best_response_value(strat, br_player=1)  # P2 best-responds: P1 utility goes down
    br_vs_p2 = best_response_value(strat, br_player=0)  # P1 best-responds: P1 utility goes up
    return ((game_value - br_vs_p1) + (br_vs_p2 - game_value)) / 2.0


def main() -> None:
    import time

    t = time.perf_counter()
    strat, avg = train(50000)
    dt = time.perf_counter() - t
    print(f"trained 50k iterations in {dt:.2f}s; average P1 utility during training {avg:+.4f} (Nash: {-1/18:+.4f})")
    print(f"exploitability of average strategy: {exploitability(strat):.4f}  (0 = exact Nash)")
    print("strategy (P(bet) by infoset):")
    for k in sorted(strat.table):
        print(f"  {k:>4}: bet {strat.table[k][1]:.3f}")
    print("\nNotice: with J, P1 bluffs ~1/3 as often as it value-bets with K; with Q it never bets. That's GTO.")


if __name__ == "__main__":
    main()
