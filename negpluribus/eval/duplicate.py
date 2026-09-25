"""Duplicate-deal ("paired seed") evaluation in bb/100 with confidence intervals.

Why not just play N hands and average?  6-max NLHE has a standard deviation of
roughly 80-100 bb/100, so distinguishing a 5 bb/100 edge at 95% confidence
needs ~150k hands.  Two variance-reduction tricks are used here:

1. **Duplicate deals.**  Each deck is played ``n`` times; the hero sits in
   every seat once while the villain line-up keeps its relative order.  Card
   luck largely cancels within a deal (as in duplicate bridge / ACPC).
2. **Paired comparison.**  ``compare_heroes`` plays hero A and hero B on the
   *same* deals and reports the difference, which is what StratFormer calls
   "gain over GTO".

Confidence intervals treat one *deal* (all rotations) as one sample, so the
within-deal correlation does not fool the statistics.

AIVAT (the estimator used for Pluribus / GTO Wizard benchmark) is the next
step up: it also corrects for the hero's own card luck and chance outcomes.
It needs a value function; we will add it once a blueprint strategy exists.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..agents.base import Agent
from ..cards import Deck
from ..engine import Street, position_name
from ..stats import StatsTracker
from ..table import play_hand


@dataclass
class EvalResult:
    hero: str
    villains: List[str]
    n_deals: int
    n_rotations: int
    bb: int
    per_deal_bb: List[float] = field(default_factory=list)  # hero net (in bb) summed over rotations
    by_position_bb: Dict[str, List[float]] = field(default_factory=dict)
    hands: int = 0

    @property
    def n_hands(self) -> int:
        return self.n_deals * self.n_rotations

    @property
    def bb100(self) -> float:
        return sum(self.per_deal_bb) / max(1, self.n_hands) * 100.0

    @property
    def se_bb100(self) -> float:
        n = len(self.per_deal_bb)
        if n < 2:
            return float("inf")
        mean = sum(self.per_deal_bb) / n
        var = sum((x - mean) ** 2 for x in self.per_deal_bb) / (n - 1)
        se_deal = math.sqrt(var / n)  # SE of per-deal mean
        return se_deal / self.n_rotations * 100.0

    @property
    def ci95(self) -> float:
        return 1.96 * self.se_bb100

    def position_table(self) -> Dict[str, float]:
        out = {}
        for pos, xs in self.by_position_bb.items():
            out[pos] = sum(xs) / max(1, len(xs)) * 100.0
        return out

    def __str__(self) -> str:
        lines = [
            f"{self.hero} vs [{', '.join(self.villains)}]  hands={self.n_hands}  "
            f"{self.bb100:+.2f} bb/100  (95% CI +/-{self.ci95:.2f})"
        ]
        pt = self.position_table()
        if pt:
            lines.append("  by position: " + "  ".join(f"{p}:{v:+.1f}" for p, v in pt.items()))
        return "\n".join(lines)


def _seat_lineup(lineup: Sequence[Agent], rotation: int) -> List[Agent]:
    n = len(lineup)
    seats: List[Optional[Agent]] = [None] * n
    for i, a in enumerate(lineup):
        seats[(rotation + i) % n] = a
    return seats  # type: ignore[return-value]


def duplicate_match(
    hero: Agent,
    villains: Sequence[Agent],
    n_deals: int = 200,
    seed: int = 0,
    sb: int = 50,
    bb: int = 100,
    stack_bb: int = 100,
    tracker: Optional[StatsTracker] = None,
    ids: Optional[Sequence[str]] = None,
    max_street: Street = Street.RIVER,
) -> EvalResult:
    """Hero takes each seat once per deal; villains keep relative order."""
    lineup = [hero] + list(villains)
    n = len(lineup)
    names = [a.name for a in lineup]
    ids = list(ids) if ids else names
    res = EvalResult(hero=hero.name, villains=list(names[1:]), n_deals=n_deals, n_rotations=n, bb=bb)
    stacks = [stack_bb * bb] * n
    for d in range(n_deals):
        deal_rng = random.Random(seed * 1_000_003 + d)
        deck_order = list(range(52))
        deal_rng.shuffle(deck_order)
        button = d % n
        total = 0.0
        for r in range(n):
            seats = _seat_lineup(lineup, r)
            for a in lineup:
                # the hero's stream is keyed by its *slot*, not its name, so two heroes compared
                # on the same deals share random draws and differ only where their policies differ
                tag = b"hero" if a is hero else a.name.encode()
                a.reset(seed=(seed * 7_919 + d * 31 + r) ^ zlib.crc32(tag))
            rec = play_hand(seats, stacks, button, sb, bb, deck=Deck.from_order(deck_order), max_street=max_street)
            hero_seat = r
            x = rec.net[hero_seat] / bb
            total += x
            pos = position_name(hero_seat, button, n)
            res.by_position_bb.setdefault(pos, []).append(x)
            if tracker is not None:
                seat_ids = [ids[lineup.index(seats[s])] for s in range(n)]
                tracker.observe_hand(rec, seat_ids)
        res.per_deal_bb.append(total)
    return res


def compare_heroes(
    hero_a: Agent,
    hero_b: Agent,
    villains: Sequence[Agent],
    n_deals: int = 200,
    seed: int = 0,
    **kw,
) -> Tuple[EvalResult, EvalResult, float, float]:
    """Play A and B on identical deals.  Returns (res_a, res_b, gain_bb100, ci95_of_gain)."""
    ra = duplicate_match(hero_a, villains, n_deals, seed, **kw)
    rb = duplicate_match(hero_b, villains, n_deals, seed, **kw)
    diffs = [a - b for a, b in zip(ra.per_deal_bb, rb.per_deal_bb)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)
    se = math.sqrt(var / n) / ra.n_rotations * 100.0
    gain = mean / ra.n_rotations * 100.0
    return ra, rb, gain, 1.96 * se
