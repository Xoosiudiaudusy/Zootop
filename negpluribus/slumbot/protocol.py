"""The Slumbot HTTP protocol, as far as it touches the cards and the chips.

Source: the official sample client https://www.slumbot.com/sample_api.py (its header comment and
its ``ParseAction``), read 2026-09-24.  What is only inferred, not verified, is listed in
docs/slumbot.md.

* Heads-up No-Limit Hold'em, blinds 50/100, both stacks 20,000 (200bb), reset every hand.
* ``client_pos`` 0: we are the big blind (second to act preflop, first postflop);
  1: we are the small blind.  Slumbot's own positions use the same numbers (0 = BB, 1 = SB).
* The action string: ``k`` check, ``c`` call, ``f`` fold, ``bN`` bet or raise where N is the
  chips the bettor has put in *on this street only* after the bet (preflop the blind counts: an SB
  raise to 2bb is ``b200``).  ``/`` separates streets.  After an all-in is called before the
  river, either no slashes follow or exactly the slashes that close the streets up to the river
  (``b20000c///``).
* Bets follow the usual rules: a bet is at least the big blind, a raise at least the previous
  bet/raise increment, an all-in is always allowed.

Our engine's layout for a Slumbot hand: two seats, the button is seat 0.  In heads-up the button
posts the small blind and acts first preflop, so Slumbot position ``p`` sits in engine seat
``1 - p`` (SB -> seat 0, BB -> seat 1).  The engine's raise amount ("total committed on this
street after acting", blinds included) is exactly Slumbot's ``N``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..cards import card_from_str, card_to_str
from ..engine import Action, ActionType, Observation

NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20_000

BUTTON_SEAT = 0  # engine seat of the button (= small blind in heads-up)
POSITION_NAMES = {0: "BB", 1: "SB"}  # Slumbot position / client_pos -> name


class ProtocolError(Exception):
    """A response that does not fit the protocol or the rules (bad action string, wrong turn...)."""


def seat_of(slumbot_pos: int) -> int:
    """Slumbot position (0 = big blind, 1 = small blind) -> engine seat (button 0 posts the SB)."""
    if slumbot_pos not in (0, 1):
        raise ProtocolError(f"position must be 0 or 1, got {slumbot_pos!r}")
    return 1 - slumbot_pos


def hero_seat(client_pos: int) -> int:
    """Our engine seat for a hand with this ``client_pos``."""
    return seat_of(client_pos)


def bot_seat(client_pos: int) -> int:
    """Slumbot's engine seat for a hand with this ``client_pos``."""
    return seat_of(1 - client_pos)


@dataclass(frozen=True)
class Move:
    """One token of the action string."""

    code: str  # "k" check, "c" call, "f" fold, "b" bet/raise
    amount: int = 0  # "b" only: the bettor's chips on this street after the bet

    def __str__(self) -> str:
        return f"b{self.amount}" if self.code == "b" else self.code


def split_action(action: str) -> List[List[Move]]:
    """Syntax only: ``"b200c/kb400"`` -> ``[[b200, c], [k, b400]]`` (one list per street).

    Whether the moves are legal, and whether each ``/`` sits where a street really ends, is
    checked by replaying them in the engine (``adapter.replay``)."""
    if not isinstance(action, str):
        raise ProtocolError(f"action must be a string, got {action!r}")
    streets = action.split("/")
    if len(streets) > NUM_STREETS:
        raise ProtocolError(f"more than {NUM_STREETS} streets in {action!r}")
    out: List[List[Move]] = []
    for s in streets:
        moves: List[Move] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in "kcf":
                moves.append(Move(ch))
                i += 1
            elif ch == "b":
                j = i + 1
                while j < len(s) and "0" <= s[j] <= "9":
                    j += 1
                if j == i + 1:
                    raise ProtocolError(f"bet without a size in {action!r}")
                moves.append(Move("b", int(s[i + 1 : j])))
                i = j
            else:
                raise ProtocolError(f"unexpected character {ch!r} in {action!r}")
        out.append(moves)
    return out


def parse_incr(incr: str) -> Move:
    """A single move as sent to ``/act`` (``"k"``, ``"c"``, ``"f"``, ``"b600"``)."""
    streets = split_action(incr)
    if len(streets) != 1 or len(streets[0]) != 1:
        raise ProtocolError(f"incr must be exactly one move, got {incr!r}")
    return streets[0][0]


def format_incr(action: Action, obs: Observation) -> str:
    """Our engine action -> the ``incr`` string, after checking that it is legal in ``obs``.

    CALL is a check when there is nothing to call.  RAISE to ``x`` is ``b<x>`` because the
    engine's amount and Slumbot's N are the same number (this street's chips, blinds included),
    so all-ins and min-raises need nothing special: ``max_raise_to`` / ``min_raise_to`` are
    already in Slumbot's units."""
    if action.type == ActionType.FOLD:
        if not obs.can_fold:
            raise ValueError("fold when checking is free")
        return "f"
    if action.type == ActionType.CALL:
        return "k" if obs.to_call == 0 else "c"
    if action.type == ActionType.RAISE:
        if not obs.can_raise:
            raise ValueError("raise not allowed here")
        if not obs.min_raise_to <= action.amount <= obs.max_raise_to:
            raise ValueError(f"raise to {action.amount} outside [{obs.min_raise_to}, {obs.max_raise_to}]")
        return f"b{action.amount}"
    raise ValueError(f"unknown action {action!r}")


def parse_cards(cards: Optional[Sequence[str]]) -> List[int]:
    """``["Ac", "9d"]`` -> engine card ints; ``None`` -> ``[]``."""
    if cards is None:
        return []
    if isinstance(cards, str) or not isinstance(cards, (list, tuple)):
        raise ProtocolError(f"cards must be a list of strings, got {cards!r}")
    try:
        return [card_from_str(c) for c in cards]
    except (ValueError, AttributeError, TypeError) as exc:
        raise ProtocolError(f"bad card list {cards!r}: {exc}") from None


def cards_str(cards: Sequence[int]) -> List[str]:
    return [card_to_str(c) for c in cards]
