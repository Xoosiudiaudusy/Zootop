"""BlueprintAgent: plays a trained MCCFR strategy at the table.

Per hand it draws a *nonce*; the pseudo-harmonic translation of every opponent
raise is seeded from (nonce, event index), so each raise is mapped once and
the key stays consistent across streets (see ``abstraction/infoset.py``).

Unknown keys (spots never reached in training) fall back to check/call and are
counted in ``n_fallback`` so you can see how often the blueprint is off-map.
"""
from __future__ import annotations

import random
from typing import Optional

from ..abstraction import BetGrid, EquityBucketer, infoset_key
from ..cfr.strategy import BlueprintStrategy
from ..engine import Action, HandRecord, Observation
from .base import Agent


class BlueprintAgent(Agent):
    name = "blueprint"

    def __init__(
        self,
        strategy: BlueprintStrategy,
        bucketer: EquityBucketer,
        grid: BetGrid,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        randomize_translation: bool = True,
    ):
        super().__init__(name=name, seed=seed)
        self.strategy = strategy
        self.bucketer = bucketer
        self.grid = grid
        self.randomize_translation = randomize_translation
        self.n_decisions = 0
        self.n_fallback = 0
        self._nonce: Optional[int] = None
        self._new_hand = True

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._new_hand = True

    def end_hand(self, record: HandRecord, my_seat: int) -> None:
        self._new_hand = True

    def _event_rng(self, i: int) -> random.Random:
        return random.Random(self._nonce * 1_000_003 + i)

    def act(self, obs: Observation) -> Action:
        if self._new_hand or self._nonce is None:
            self._nonce = self.rng.getrandbits(32)
            self._new_hand = False
        legal = self.grid.abstract_actions(obs)
        key = infoset_key(obs, self.bucketer, self.grid, event_rng=self._event_rng if self.randomize_translation else None)
        probs = self.strategy.policy(key, legal)
        self.n_decisions += 1
        if probs is None:
            self.n_fallback += 1
            return self.grid.to_concrete(obs, "c")
        r = self.rng.random()
        acc = 0.0
        choice = legal[-1]
        for a, p in zip(legal, probs):
            acc += p
            if r < acc:
                choice = a
                break
        return self.grid.to_concrete(obs, choice)

    @property
    def fallback_rate(self) -> float:
        return self.n_fallback / self.n_decisions if self.n_decisions else 0.0
