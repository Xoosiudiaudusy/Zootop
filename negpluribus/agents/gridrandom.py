"""GridRandomAgent: uniform over the abstract actions of a bet grid.

Every raise it makes is an on-grid size, so a blueprint on the same grid never has to translate
its actions.  Against it, acceptance measures the abstraction and the strategy alone; against the
plain ``RandomAgent`` (arbitrary chip amounts) it also measures action translation.  The pair
separates the two sources of error (proposed by the explainer session, 2026-09-24).
"""
from __future__ import annotations

from typing import Optional

from ..abstraction import BetGrid
from ..engine import Action, Observation
from .base import Agent


class GridRandomAgent(Agent):
    name = "gridrandom"

    def __init__(self, grid: BetGrid, name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name=name, seed=seed)
        self.grid = grid

    def act(self, obs: Observation) -> Action:
        return self.grid.to_concrete(obs, self.rng.choice(self.grid.abstract_actions(obs)))
