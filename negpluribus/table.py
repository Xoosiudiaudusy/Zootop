"""Table runner: wires agents to the engine for one hand or a session.

``play_hand`` is the single-hand primitive (used by evaluation with fixed
decks).  ``CashTable`` runs a cash-game session: button rotates, stacks are
reset to ``stack_bb`` big blinds each hand (standard for bot evaluation, so
results are not dominated by stack-depth drift).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .agents.base import Agent
from .cards import Deck
from .engine import HandRecord, HandState, Street

HandHook = Callable[[HandState, int], None]  # (state_before_action, seat) for logging/UI


def play_hand(
    agents: Sequence[Agent],
    stacks: Sequence[int],
    button: int,
    sb: int = 50,
    bb: int = 100,
    ante: int = 0,
    deck: Optional[Deck] = None,
    seed: Optional[int] = None,
    on_decision: Optional[HandHook] = None,
    max_street: Street = Street.RIVER,
) -> HandRecord:
    state = HandState(stacks, button, sb, bb, ante, deck=deck, seed=seed, max_street=max_street)
    while not state.is_terminal:
        seat = state.current_player
        assert seat is not None
        if on_decision:
            on_decision(state, seat)
        obs = state.observe(seat)
        action = agents[seat].act(obs)
        state.apply(action)
    record = state.record()
    for seat, agent in enumerate(agents):
        agent.end_hand(record, seat)
    return record


@dataclass
class SessionResult:
    records: List[HandRecord] = field(default_factory=list)
    net: List[int] = field(default_factory=list)  # cumulative chips per seat

    def bb_per_100(self, seat: int, bb: int) -> float:
        n = len(self.records)
        return 0.0 if n == 0 else self.net[seat] / bb / n * 100.0


class CashTable:
    def __init__(
        self,
        agents: Sequence[Agent],
        sb: int = 50,
        bb: int = 100,
        ante: int = 0,
        stack_bb: int = 100,
        reset_stacks: bool = True,
        seed: Optional[int] = None,
        max_street: Street = Street.RIVER,
    ):
        self.max_street = max_street
        self.agents = list(agents)
        self.n = len(agents)
        self.sb, self.bb, self.ante = sb, bb, ante
        self.stack_bb = stack_bb
        self.reset_stacks = reset_stacks
        self.rng = random.Random(seed)
        self.stacks = [stack_bb * bb] * self.n
        self.button = 0
        self.result = SessionResult(net=[0] * self.n)

    def play(self, n_hands: int, on_decision: Optional[HandHook] = None, verbose: bool = False) -> SessionResult:
        for _ in range(n_hands):
            if self.reset_stacks:
                self.stacks = [self.stack_bb * self.bb] * self.n
            else:
                # rebuy busted players
                self.stacks = [s if s > 0 else self.stack_bb * self.bb for s in self.stacks]
            deck = Deck(rng=self.rng)
            rec = play_hand(
                self.agents, self.stacks, self.button, self.sb, self.bb, self.ante, deck=deck,
                on_decision=on_decision, max_street=self.max_street,
            )
            self.stacks = [s + d for s, d in zip(self.stacks, rec.net)]
            for i, d in enumerate(rec.net):
                self.result.net[i] += d
            self.result.records.append(rec)
            if verbose:
                print(format_record(rec, [a.name for a in self.agents]))
            self.button = (self.button + 1) % self.n
        return self.result


def format_record(rec: HandRecord, names: Optional[Sequence[str]] = None) -> str:
    from .cards import cards_to_str
    from .engine import STREET_NAMES

    names = names or [f"seat{i}" for i in range(rec.n_players)]
    lines = [f"--- hand: button={names[rec.button]}  board=[{cards_to_str(rec.board)}]"]
    cur = None
    for e in rec.events:
        if e.street != cur:
            cur = e.street
            lines.append(f"  [{STREET_NAMES[cur]}]")
        lines.append(f"    {names[e.seat]:>10} ({rec.position(e.seat)}): {e.action}  (to_call={e.to_call}, pot={e.pot_before})")
    for s in range(rec.n_players):
        shown = cards_to_str(rec.hole_cards[s]) if s in rec.showdown_seats else "-- --"
        lines.append(f"  {names[s]:>10}: {shown}  net={rec.net[s]:+d}{'  WIN' if s in rec.winners else ''}")
    return "\n".join(lines)
