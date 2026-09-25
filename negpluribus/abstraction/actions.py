"""Action abstraction: a small grid of bet sizes instead of every chip amount.

Abstract actions are short names:

    "f"      fold
    "c"      check / call
    "r0.5"   raise by 0.5 x pot   (raise increment = 0.5 * (pot + amount to call))
    "r1.0"   raise by 1.0 x pot
    "a"      all-in

``BetGrid.abstract_actions(obs)`` lists the ones that are legal *and distinct*
right now (two fractions that clamp to the same chip amount collapse into one).
``to_concrete`` turns a name into an engine ``Action``.

The reverse direction matters just as much: opponents bet whatever they like.
``from_concrete`` maps an observed raise onto the grid with the
**pseudo-harmonic mapping** (Ganzfried & Sandholm 2013).  For an observed size
x between grid sizes A < x < B (all as pot fractions) it goes to A with probability

    f_A(x) = (B - x) * (1 + A) / ((B - A) * (1 + x))

and to B otherwise.  Compared with "nearest size" this is much harder to
exploit: a bet just above the small size is not always read as the small size.

A and B are the neighbouring ABSTRACT actions, and the all-in is one of them (the paper's own
example: A = pot, B = all-in).  So a bet above the largest grid size is translated between that
size and the actor's all-in, not clamped to the largest size: until 2026-09-24 it was clamped,
and a 60bb bet into a 2bb pot was read as a pot-size bet (the HUNL blueprint lost to a random
bettor because of it).  Sizes at or above the actor's all-in are the all-in in that spot.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..engine import CALL, FOLD, Action, ActionType, Event, Observation, Street

FOLD_NAME = "f"
CALL_NAME = "c"
ALL_IN = "a"


def raise_name(frac: float) -> str:
    return f"r{frac:g}"


@dataclass(frozen=True)
class BetGrid:
    preflop_fracs: Tuple[float, ...] = (0.5, 1.0)
    postflop_fracs: Tuple[float, ...] = (0.5, 1.0)
    allow_all_in: bool = True
    max_raises_per_street: int = 4  # after this many raises only call/fold/all-in remain
    forbid_open_limp: bool = False  # push/fold games: no limping into an unraised preflop pot

    def fracs_for(self, street: Street) -> Tuple[float, ...]:
        return self.preflop_fracs if street == Street.PREFLOP else self.postflop_fracs

    # ------------------------------------------------------ name -> chips
    @staticmethod
    def raise_to_for_frac(obs: Observation, frac: float) -> int:
        pot_after_call = obs.pot + obs.to_call
        increment = frac * pot_after_call
        return int(round(obs.street_bets_max() + increment))

    def abstract_actions(self, obs: Observation) -> List[str]:
        names: List[str] = []
        if obs.can_fold:
            names.append(FOLD_NAME)
        limp = obs.street == Street.PREFLOP and obs.raises_this_street == 0 and obs.to_call > 0
        if not (self.forbid_open_limp and limp and obs.can_raise):
            names.append(CALL_NAME)
        if not obs.can_raise:
            return names
        capped = obs.raises_this_street >= self.max_raises_per_street
        seen_amounts = set()
        if not capped:
            for frac in self.fracs_for(obs.street):
                amt = obs.clamp_raise(self.raise_to_for_frac(obs, frac)).amount
                if amt in seen_amounts:
                    continue
                if amt == obs.max_raise_to and self.allow_all_in:
                    continue  # a fraction that hits the whole stack *is* the all-in
                seen_amounts.add(amt)
                names.append(raise_name(frac))
        if self.allow_all_in and obs.max_raise_to not in seen_amounts:
            names.append(ALL_IN)
        return names

    def to_concrete(self, obs: Observation, name: str) -> Action:
        if name == FOLD_NAME:
            return FOLD if obs.can_fold else CALL
        if name == CALL_NAME:
            return CALL
        if not obs.can_raise:
            return CALL
        if name == ALL_IN:
            return obs.clamp_raise(obs.max_raise_to)
        frac = float(name[1:])
        return obs.clamp_raise(self.raise_to_for_frac(obs, frac))

    # ------------------------------------------------------ chips -> name
    @staticmethod
    def observed_frac(ev: Event) -> float:
        """Pot fraction of a raise seen in the log: increment / pot after the call."""
        increment = ev.paid - ev.to_call
        pot_after_call = ev.pot_before + ev.to_call
        return max(0.0, increment / pot_after_call) if pot_after_call > 0 else 0.0

    def from_concrete(
        self,
        ev: Event,
        all_in: bool,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Map a logged action onto the grid.  ``all_in``: did this raise put the actor all-in."""
        if ev.action.type == ActionType.FOLD:
            return FOLD_NAME
        if ev.action.type == ActionType.CALL:
            return CALL_NAME
        fracs = self.fracs_for(ev.street)
        if all_in and self.allow_all_in:
            return ALL_IN
        if not fracs:
            return ALL_IN  # only fold/call/all-in exist in this grid
        x = self.observed_frac(ev)
        x_allin = self.all_in_frac(ev)
        if self.allow_all_in and x_allin is not None:
            # Ganzfried & Sandholm (IJCAI 2013): translate between the two NEIGHBOURING abstract
            # actions, and the all-in is one of them (their example: A = pot, B = all-in).  Grid
            # sizes at or above the actor's all-in are the all-in in this spot, so they drop out.
            below = [f for f in sorted(fracs) if f < x_allin]
            if not below:
                return ALL_IN
            top = below[-1]
            if x > top:
                p_top = (x_allin - x) * (1 + top) / ((x_allin - top) * (1 + x))
                if rng is None:
                    return raise_name(top) if p_top >= 0.5 else ALL_IN
                return raise_name(top) if rng.random() < p_top else ALL_IN
            return raise_name(pseudo_harmonic(x, below, rng))
        # no stack information (old records) or no all-in in the grid: nearest grid sizes only
        return raise_name(pseudo_harmonic(x, fracs, rng))

    @staticmethod
    def all_in_frac(ev: Event) -> Optional[float]:
        """The actor's all-in raise as a pot fraction, in the same units as ``observed_frac``:
        (chips the actor had before acting - the call) / (pot after the call).  None if unknown."""
        if ev.stack_after < 0:
            return None
        pot_after_call = ev.pot_before + ev.to_call
        if pot_after_call <= 0:
            return None
        return (ev.paid + ev.stack_after - ev.to_call) / pot_after_call


def pseudo_harmonic(x: float, grid: Sequence[float], rng: Optional[random.Random] = None) -> float:
    """Return the grid size an observed pot-fraction ``x`` maps to."""
    grid = sorted(grid)
    if x <= grid[0]:
        return grid[0]
    if x >= grid[-1]:
        return grid[-1]
    for a, b in zip(grid, grid[1:]):
        if a <= x <= b:
            p_a = (b - x) * (1 + a) / ((b - a) * (1 + x))
            if rng is None:
                return a if p_a >= 0.5 else b
            return a if rng.random() < p_a else b
    return grid[-1]
