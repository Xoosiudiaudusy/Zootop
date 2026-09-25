"""Agent interface.

An agent sees an ``Observation`` when it must act and returns an ``Action``.
After every hand it receives the public ``HandRecord`` (all actions, board,
showdown cards) so it can update opponent models.
"""
from __future__ import annotations

import random
from typing import Optional

from ..engine import Action, ActionType, HandRecord, Observation


class Agent:
    name: str = "agent"

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None):
        if name:
            self.name = name
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed internal randomness (called before each duplicate deal)."""
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:  # pragma: no cover - abstract
        raise NotImplementedError

    def end_hand(self, record: HandRecord, my_seat: int) -> None:
        """Hook for learning / opponent modelling.  Default: nothing."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class RandomAgent(Agent):
    """Uniform over {fold, call, random-size raise}.  A sanity baseline."""

    name = "random"

    def act(self, obs: Observation) -> Action:
        acts = obs.legal_actions()
        a = self.rng.choice(acts)
        if a.type == ActionType.RAISE:
            return obs.clamp_raise(self.rng.randint(obs.min_raise_to, obs.max_raise_to))
        return a


class CallingAgent(Agent):
    """Always check/call.  Useful for tests and as the most exploitable fish."""

    name = "caller"

    def act(self, obs: Observation) -> Action:
        from ..engine import CALL

        return CALL
