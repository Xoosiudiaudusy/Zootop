"""Information-set key = everything CFR is allowed to condition on.

    "<street>|<position>|<n_active>|b<bucket>|<abstract history>"

* street, position and number of live players describe *where* we are;
* the card bucket replaces our actual cards;
* the history is the sequence of abstract actions so far, streets separated
  by "/", e.g. ``f f r1 c f c/c r0.5`` — every raise already translated
  onto the grid, so two hands with slightly different opponent sizes share
  the same key (this is what makes the table finite).

Nothing about the opponents' *identity* is in the key: the blueprint is
opponent-agnostic by design.  Opponent stats enter later, in the search /
exploitation layer.

Where the pseudo-harmonic *probability* lives
--------------------------------------------
The key never contains a probability: it contains the *result* of the coin
flip.  The flip happens in the translator (``BetGrid.from_concrete``), and it
must be flipped **once per raise per hand** and remembered, otherwise the same
flop bet could read as ``r0.5`` on the turn and ``r1`` on the river and the
history in the key would drift between streets.  ``history_string`` therefore
takes ``event_rng(i)``: a function returning the RNG for the i-th event of the
hand, seeded from a per-hand nonce (see ``BlueprintAgent``).  With
``event_rng=None`` the mapping is deterministic (nearest in the harmonic
sense), which is what self-play training uses: every bet is on the grid anyway.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence

from ..engine import Event, Observation, Street
from .actions import BetGrid
from .buckets import EquityBucketer


def starting_stacks(obs: Observation) -> List[int]:
    """Each seat's stack before blinds/antes.  The engine reports it in the Observation;
    the fallback reconstruction (stacks + chips paid in events) misses the posted blinds,
    which broke all-in detection and subgame rebuilding, so prefer the reported value."""
    if obs.starting_stacks:
        return list(obs.starting_stacks)
    start = list(obs.stacks)
    for ev in obs.events:
        start[ev.seat] += ev.paid
    return start


def history_string(
    events: Sequence[Event],
    grid: BetGrid,
    event_rng: Optional[Callable[[int], random.Random]] = None,
) -> str:
    """Abstract action names of ``events``, streets separated by '/'.  Whether a raise was an
    all-in comes from the event itself (``Event.all_in``, set by the engine)."""
    parts: List[str] = []
    cur: Optional[Street] = None
    for i, ev in enumerate(events):
        if ev.street != cur:
            if cur is not None:
                parts.append("/")
            cur = ev.street
        parts.append(grid.from_concrete(ev, ev.all_in, event_rng(i) if event_rng else None))
    return " ".join(parts).replace(" / ", "/")


def infoset_key(
    obs: Observation,
    bucketer: EquityBucketer,
    grid: BetGrid,
    event_rng: Optional[Callable[[int], random.Random]] = None,
) -> str:
    b = bucketer.bucket(obs.hole, obs.board)
    hist = history_string(obs.events, grid, event_rng)
    return f"{obs.street.name[0]}|{obs.position}|{obs.n_active}|b{b}|{hist}"


def infoset_key_for_bucket(
    obs: Observation,
    bucket: int,
    grid: BetGrid,
    event_rng: Optional[Callable[[int], random.Random]] = None,
) -> str:
    """Same key as ``infoset_key`` but for a hypothetical bucket (used by best-response code
    that asks "what would the blueprint do here *with hand class b*?")."""
    hist = history_string(obs.events, grid, event_rng)
    return f"{obs.street.name[0]}|{obs.position}|{obs.n_active}|b{bucket}|{hist}"
