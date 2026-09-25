"""No-Limit Texas Hold'em engine for 2..6 players (cash-game hand).

Design goals
------------
* **Correct rules**: blinds/antes, heads-up button rule, min-raise tracking,
  all-ins, side pots, uncalled-bet return, odd-chip distribution.
* **Deterministic**: give it a deck order and the hand replays exactly (needed
  for duplicate/paired evaluation).
* **Inspectable**: every action is appended to ``events`` with the context the
  actor faced (street, to_call, pot) so a HUD/stats layer can be built on top
  without re-simulating.

Actions are a small closed set:

    Action(FOLD)                – only legal when facing a bet
    Action(CALL)                – check when to_call == 0, otherwise call (capped by stack)
    Action(RAISE, amount=to)    – "raise to" ``to`` chips committed on this street
                                   (a bet when nobody has bet yet); all-in allowed
                                   even below the min-raise.

Chips are ints.  Convention used throughout the project: sb=50, bb=100,
stacks 10_000 (= 100bb), so bb/100 = chips / 100 / hands * 100.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

from .cards import Deck, cards_to_str
from .evaluator import evaluate


class ActionType(IntEnum):
    FOLD = 0
    CALL = 1  # check or call
    RAISE = 2  # bet or raise (to amount)


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


STREET_NAMES = ["preflop", "flop", "turn", "river", "showdown"]
BOARD_CARDS_BY_STREET = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}

POSITION_NAMES_6 = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]


@dataclass(frozen=True)
class Action:
    type: ActionType
    amount: int = 0  # RAISE only: total chips committed on this street after acting

    def __str__(self) -> str:
        if self.type == ActionType.FOLD:
            return "fold"
        if self.type == ActionType.CALL:
            return "call"
        return f"raise_to {self.amount}"


FOLD = Action(ActionType.FOLD)
CALL = Action(ActionType.CALL)


def raise_to(amount: int) -> Action:
    return Action(ActionType.RAISE, int(amount))


@dataclass
class PlayerState:
    seat: int
    stack: int
    hole: List[int] = field(default_factory=list)
    street_bet: int = 0  # chips committed on the current street
    invested: int = 0  # chips committed in the whole hand
    folded: bool = False
    all_in: bool = False
    acted: bool = False  # acted since the last raise on this street

    @property
    def active(self) -> bool:
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in


@dataclass(frozen=True)
class Event:
    """One decision, with the context the actor saw."""

    street: Street
    seat: int
    action: Action
    to_call: int  # chips needed to call before acting
    pot_before: int  # pot (incl. all street bets) before acting
    facing_raise: bool  # someone has bet/raised on this street before us
    raises_this_street: int  # number of bets/raises on this street before acting
    paid: int  # chips this action put in
    all_in: bool = False  # the actor has no chips left after this action
    stack_after: int = -1  # the actor's remaining stack after this action (-1 = unknown, old records)

    @property
    def is_aggressive(self) -> bool:
        return self.action.type == ActionType.RAISE

    @property
    def is_voluntary_money(self) -> bool:
        return self.action.type != ActionType.FOLD and self.paid > 0


@dataclass
class HandRecord:
    """Everything that happened in a hand; what agents/stat-trackers see afterwards."""

    n_players: int
    button: int
    sb: int
    bb: int
    ante: int
    starting_stacks: List[int]
    hole_cards: List[List[int]]  # all players' cards (public post-hoc; hide as needed)
    board: List[int]
    events: List[Event]
    net: List[int]  # chips won/lost per seat
    showdown_seats: List[int]  # seats whose cards were revealed at showdown
    winners: List[int]
    saw_flop: List[bool]

    def position(self, seat: int) -> str:
        return position_name(seat, self.button, self.n_players)


def position_name(seat: int, button: int, n: int) -> str:
    """Seat's position label relative to the button (6-max labels)."""
    rel = (seat - button) % n
    if n == 2:
        return ["BTN/SB", "BB"][rel]
    labels = ["BTN", "SB", "BB"] + ["UTG", "HJ", "CO"][: n - 3]
    if n < 6:  # e.g. 5-handed: BTN SB BB UTG CO
        labels = ["BTN", "SB", "BB"] + ["UTG", "HJ", "CO"][6 - n :]
    return labels[rel]


@dataclass
class Observation:
    """What an agent is allowed to know when it must act."""

    seat: int
    hole: List[int]
    board: List[int]
    street: Street
    pot: int
    to_call: int
    stack: int
    stacks: List[int]
    street_bets: List[int]
    folded: List[bool]
    all_in: List[bool]
    button: int
    n_players: int
    bb: int
    min_raise_to: int
    max_raise_to: int
    can_raise: bool
    can_fold: bool
    events: List[Event]  # this hand so far
    facing_raise: bool
    raises_this_street: int
    aggressor: Optional[int]  # seat of the last bettor/raiser on this street (or previous street if none)
    starting_stacks: List[int] = field(default_factory=list)  # stacks before blinds/antes were posted

    @property
    def position(self) -> str:
        return position_name(self.seat, self.button, self.n_players)

    @property
    def n_active(self) -> int:
        return sum(1 for f in self.folded if not f)

    def street_bets_max(self) -> int:
        """Current bet level on this street (0 if nobody has bet)."""
        return max(self.street_bets) if self.street_bets else 0

    @property
    def pot_odds(self) -> float:
        """Fraction of the final pot we must contribute to call (0 if free)."""
        if self.to_call == 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    def legal_actions(self) -> List[Action]:
        acts = []
        if self.can_fold:
            acts.append(FOLD)
        acts.append(CALL)
        if self.can_raise:
            acts.append(raise_to(self.min_raise_to))
            if self.max_raise_to != self.min_raise_to:
                acts.append(raise_to(self.max_raise_to))
        return acts

    def clamp_raise(self, amount: int) -> Action:
        """Turn any desired raise-to size into a legal action (or a call if we can't raise)."""
        if not self.can_raise:
            return CALL
        amount = int(round(amount))
        amount = max(self.min_raise_to, min(self.max_raise_to, amount))
        return raise_to(amount)


class HandState:
    """Mutable state of a single hand.  Drive it with ``apply(action)`` until ``is_terminal``."""

    def __init__(
        self,
        stacks: Sequence[int],
        button: int,
        sb: int = 50,
        bb: int = 100,
        ante: int = 0,
        deck: Optional[Deck] = None,
        seed: Optional[int] = None,
        max_street: Street = Street.RIVER,
    ):
        n = len(stacks)
        if n < 2 or n > 9:
            raise ValueError("2..9 players")
        if any(s <= 0 for s in stacks):
            raise ValueError("all stacks must be > 0 (sit out players by omitting them)")
        self.n = n
        self.button = button % n
        self.sb, self.bb, self.ante = sb, bb, ante
        # last street with betting; afterwards the board is run out and hands go to
        # showdown.  RIVER = full game; FLOP/PREFLOP give the reduced games used for
        # fast CFR experiments.
        self.max_street = Street(min(int(max_street), int(Street.RIVER)))
        self.starting_stacks = list(stacks)
        self.players = [PlayerState(seat=i, stack=int(s)) for i, s in enumerate(stacks)]
        self.deck = deck if deck is not None else Deck(seed=seed)
        self.board: List[int] = []
        self.events: List[Event] = []
        self.street = Street.PREFLOP
        self.current_bet = 0
        self.min_raise = bb
        self.raises_this_street = 0
        self.last_aggressor: Optional[int] = None
        self.to_act: Optional[int] = None
        self.terminal = False
        self.winners: List[int] = []
        self.showdown_seats: List[int] = []
        self.saw_flop = [False] * n

        # deal
        for p in self.players:
            p.hole = self.deck.draw(2)

        # antes + blinds
        if ante:
            for p in self.players:
                self._commit(p, ante)
        if n == 2:
            sb_seat, bb_seat = self.button, (self.button + 1) % n
        else:
            sb_seat, bb_seat = (self.button + 1) % n, (self.button + 2) % n
        self._commit(self.players[sb_seat], sb)
        self._commit(self.players[bb_seat], bb)
        self.current_bet = max(self.players[sb_seat].street_bet, self.players[bb_seat].street_bet)
        self.min_raise = bb
        self.to_act = self._next_can_act((bb_seat) % n)
        self._check_round_end(initial=True)

    # ------------------------------------------------------------------ helpers
    def _commit(self, p: PlayerState, amount: int) -> int:
        amount = min(amount, p.stack)
        p.stack -= amount
        p.street_bet += amount
        p.invested += amount
        if p.stack == 0:
            p.all_in = True
        return amount

    @property
    def pot(self) -> int:
        return sum(p.invested for p in self.players)

    @property
    def is_terminal(self) -> bool:
        return self.terminal

    @property
    def current_player(self) -> Optional[int]:
        return None if self.terminal else self.to_act

    def _next_can_act(self, after: int) -> Optional[int]:
        for k in range(1, self.n + 1):
            q = self.players[(after + k) % self.n]
            if q.can_act:
                return q.seat
        return None

    def _active(self) -> List[PlayerState]:
        return [p for p in self.players if not p.folded]

    def to_call_for(self, seat: int) -> int:
        p = self.players[seat]
        return min(self.current_bet - p.street_bet, p.stack)

    def raise_bounds(self, seat: int) -> Tuple[bool, int, int]:
        """(can_raise, min_raise_to, max_raise_to) for ``seat``."""
        p = self.players[seat]
        to_call = self.current_bet - p.street_bet
        if p.stack <= to_call:
            return False, 0, 0
        others = [q for q in self.players if q.seat != seat and q.can_act]
        if not others:
            return False, 0, 0
        max_to = p.street_bet + p.stack
        min_to = self.current_bet + self.min_raise if self.current_bet > 0 else self.bb
        if max_to < min_to:
            min_to = max_to  # only an all-in "raise" is possible
        return True, min_to, max_to

    # ------------------------------------------------------------- observation
    def observe(self, seat: Optional[int] = None) -> Observation:
        seat = self.to_act if seat is None else seat
        assert seat is not None
        p = self.players[seat]
        can_raise, min_to, max_to = self.raise_bounds(seat)
        to_call = self.to_call_for(seat)
        return Observation(
            seat=seat,
            hole=list(p.hole),
            board=list(self.board),
            street=self.street,
            pot=self.pot,
            to_call=to_call,
            stack=p.stack,
            stacks=[q.stack for q in self.players],
            street_bets=[q.street_bet for q in self.players],
            folded=[q.folded for q in self.players],
            all_in=[q.all_in for q in self.players],
            button=self.button,
            n_players=self.n,
            bb=self.bb,
            min_raise_to=min_to,
            max_raise_to=max_to,
            can_raise=can_raise,
            can_fold=to_call > 0,
            events=list(self.events),
            facing_raise=self.raises_this_street > 0,
            raises_this_street=self.raises_this_street,
            aggressor=self.last_aggressor,
            starting_stacks=list(self.starting_stacks),
        )

    # ------------------------------------------------------------------ clone
    def clone(self) -> "HandState":
        """Independent copy (same future cards).  Used by CFR to branch on actions."""
        new = HandState.__new__(HandState)
        new.__dict__.update(self.__dict__)
        new.players = [copy.copy(p) for p in self.players]
        new.board = list(self.board)
        new.events = list(self.events)
        new.saw_flop = list(self.saw_flop)
        new.winners = list(self.winners)
        new.showdown_seats = list(self.showdown_seats)
        d = Deck.__new__(Deck)
        d.cards, d._pos, d.rng = self.deck.cards, self.deck._pos, self.deck.rng  # card order is never mutated
        new.deck = d
        return new

    # ------------------------------------------------------------------ apply
    def apply(self, action: Action) -> Event:
        if self.terminal or self.to_act is None:
            raise RuntimeError("hand is over")
        seat = self.to_act
        p = self.players[seat]
        to_call = self.to_call_for(seat)
        pot_before = self.pot
        facing = self.raises_this_street > 0
        n_raises = self.raises_this_street
        paid = 0

        if action.type == ActionType.FOLD:
            if to_call == 0:
                raise ValueError("cannot fold when checking is free")
            p.folded = True
        elif action.type == ActionType.CALL:
            paid = self._commit(p, to_call)
        elif action.type == ActionType.RAISE:
            can_raise, min_to, max_to = self.raise_bounds(seat)
            if not can_raise:
                raise ValueError("raise not allowed here")
            to = action.amount
            if to < min_to or to > max_to:
                raise ValueError(f"raise_to {to} outside [{min_to}, {max_to}]")
            raise_size = to - self.current_bet
            paid = self._commit(p, to - p.street_bet)
            if raise_size >= self.min_raise:
                self.min_raise = raise_size
            self.current_bet = to
            self.raises_this_street += 1
            self.last_aggressor = seat
            for q in self.players:
                if q.seat != seat:
                    q.acted = False
        else:
            raise ValueError(action)

        p.acted = True
        ev = Event(self.street, seat, action, to_call, pot_before, facing, n_raises, paid, all_in=p.all_in, stack_after=p.stack)
        self.events.append(ev)
        self._check_round_end()
        return ev

    # ------------------------------------------------------- round transitions
    def _check_round_end(self, initial: bool = False) -> None:
        active = self._active()
        if len(active) == 1:
            self._finish(active)
            return
        can_act = [p for p in active if p.can_act]
        if initial:
            # preflop: the blinds still get to act (option), so nothing to do unless
            # posting blinds already put everyone all-in
            if len(can_act) <= 1 and all(p.street_bet == self.current_bet or p.all_in for p in active):
                self._run_out_and_showdown()
            elif not can_act:
                self._run_out_and_showdown()
            return
        done = all(p.acted and p.street_bet == self.current_bet for p in can_act)
        if not done:
            self.to_act = self._next_can_act(self.to_act)
            return
        if len(can_act) <= 1 or self.street >= self.max_street:
            self._run_out_and_showdown()
            return
        self._next_street()

    def _next_street(self) -> None:
        self.street = Street(self.street + 1)
        for p in self.players:
            p.street_bet = 0
            p.acted = False
        self.current_bet = 0
        self.min_raise = self.bb
        self.raises_this_street = 0
        need = BOARD_CARDS_BY_STREET[self.street] - len(self.board)
        if need > 0:
            self.board.extend(self.deck.draw(need))
        if self.street == Street.FLOP:
            for p in self.players:
                if not p.folded:
                    self.saw_flop[p.seat] = True
        self.to_act = self._next_can_act(self.button)

    def _run_out_and_showdown(self) -> None:
        if self.street < Street.FLOP:
            for p in self.players:
                if not p.folded:
                    self.saw_flop[p.seat] = True
        need = 5 - len(self.board)
        if need > 0:
            self.board.extend(self.deck.draw(need))
        self.street = Street.RIVER
        self._showdown()

    def _showdown(self) -> None:
        active = self._active()
        self.showdown_seats = [p.seat for p in active]
        self._finish(active)

    def _finish(self, active: List[PlayerState]) -> None:
        """Distribute the pot (with side pots) among ``active`` players and end the hand."""
        self.terminal = True
        self.to_act = None
        strengths: Dict[int, int] = {}
        if len(active) > 1:
            for p in active:
                strengths[p.seat] = evaluate(p.hole + self.board)
        winners_all: set = set()
        levels = sorted({p.invested for p in self.players if p.invested > 0})
        prev = 0
        for lvl in levels:
            portion = 0
            for p in self.players:
                portion += max(0, min(p.invested, lvl) - prev)
            eligible = [p for p in active if p.invested >= lvl]
            if not eligible:
                # everyone who contributed to this layer folded; give to best remaining active
                eligible = active
            if len(eligible) == 1:
                eligible[0].stack += portion
                winners_all.add(eligible[0].seat)
            else:
                best = max(strengths[p.seat] for p in eligible)
                ws = [p for p in eligible if strengths[p.seat] == best]
                share, odd = divmod(portion, len(ws))
                # odd chips go to the first winner left of the button
                ws.sort(key=lambda p: (p.seat - self.button - 1) % self.n)
                for i, p in enumerate(ws):
                    p.stack += share + (1 if i < odd else 0)
                    winners_all.add(p.seat)
            prev = lvl
        self.winners = sorted(winners_all)
        self.street = Street.SHOWDOWN

    # ----------------------------------------------------------------- record
    def record(self) -> HandRecord:
        assert self.terminal
        return HandRecord(
            n_players=self.n,
            button=self.button,
            sb=self.sb,
            bb=self.bb,
            ante=self.ante,
            starting_stacks=list(self.starting_stacks),
            hole_cards=[list(p.hole) for p in self.players],
            board=list(self.board),
            events=list(self.events),
            net=[p.stack - s for p, s in zip(self.players, self.starting_stacks)],
            showdown_seats=list(self.showdown_seats),
            winners=list(self.winners),
            saw_flop=list(self.saw_flop),
        )

    # ------------------------------------------------------------------ debug
    def pretty(self) -> str:
        lines = [f"street={STREET_NAMES[self.street]} board=[{cards_to_str(self.board)}] pot={self.pot}"]
        for p in self.players:
            tag = "F" if p.folded else ("A" if p.all_in else " ")
            lines.append(
                f"  seat{p.seat} {position_name(p.seat, self.button, self.n):>6} [{cards_to_str(p.hole)}] "
                f"stack={p.stack:>6} bet={p.street_bet:>5} {tag}{' <-' if p.seat == self.to_act else ''}"
            )
        return "\n".join(lines)
