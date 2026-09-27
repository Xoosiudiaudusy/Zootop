"""The value overbettor: a test opponent for action translation (docs/scale_4street.md, 2026-09-24).

It plays exactly like the blueprint, except that when the blueprint raises and its hand is strong it
overbets instead: ``mult`` times the grid's largest raise size of the street (as a pot fraction),
capped at 80% of the way to its all-in.  The size is off the grid, above the largest abstract size,
so a translating agent reads it as that size or as the all-in, and the bet is pure value.  On the
narrow HU 100bb grid the old clamping translation lost 54 bb/100 to this opponent (x2.5).

"Strong": preflop, a hole class in the top ``strong_preflop`` of ``hole_percentile``; afterwards, an
equity against one random hand of at least ``strong_equity`` (exact on the river, ``samples`` Monte
Carlo draws before).
"""
from __future__ import annotations

from typing import Optional

from ..engine import Action, ActionType, HandRecord, Observation
from ..equity import hole_percentile
from .base import Agent
from .blueprint import BlueprintAgent


class ValueOverbettor(Agent):
    name = "overbettor"

    def __init__(self, strategy, bucketer, grid, mult: float = 2.5, strong_preflop: float = 0.85,
                 strong_equity: float = 0.8, samples: int = 400, name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name=name, seed=seed)
        self.inner = BlueprintAgent(strategy, bucketer, grid, name=f"{self.name}_bp", seed=seed)
        self.grid = grid
        self.mult = mult
        self.strong_preflop = strong_preflop
        self.strong_equity = strong_equity
        self.samples = samples
        self.n_overbets = 0

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.inner.reset(seed)

    def end_hand(self, record: HandRecord, my_seat: int) -> None:
        self.inner.end_hand(record, my_seat)

    def strong(self, obs: Observation) -> bool:
        if not obs.board:
            return hole_percentile(obs.hole[0], obs.hole[1]) >= self.strong_preflop
        from ..fast import core

        c = core()
        if len(obs.board) == 5:
            eq = c.river_equity_exact(list(obs.hole), list(obs.board))
        else:
            eq = c.equity_vs_random_seeded(list(obs.hole), list(obs.board), 1, self.samples, self.rng.getrandbits(32))
        return eq >= self.strong_equity

    def act(self, obs: Observation) -> Action:
        a = self.inner.act(obs)
        if a.type == ActionType.RAISE and obs.can_raise and a.amount < obs.max_raise_to and self.strong(obs):
            level = obs.street_bets_max()
            want = level + max(self.grid.fracs_for(obs.street)) * self.mult * (obs.pot + obs.to_call)
            want = int(round(min(want, level + 0.8 * (obs.max_raise_to - level))))
            if want > a.amount:  # an overbet only (a short stack can leave no room for one)
                self.n_overbets += 1
                return obs.clamp_raise(want)
        return a

    @property
    def fallback_rate(self) -> float:
        return self.inner.fallback_rate
