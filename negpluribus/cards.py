"""Card primitives.

A card is an int in ``range(52)``:  ``rank = card // 4`` (0 = deuce … 12 = ace),
``suit = card % 4`` (0=c, 1=d, 2=h, 3=s).  Keeping cards as small ints makes
evaluation and hashing cheap, which matters once we start running millions of
hands for CFR / evaluation.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
NUM_CARDS = 52


def make_card(rank: int, suit: int) -> int:
    return rank * 4 + suit


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def card_from_str(s: str) -> int:
    """'As' -> 51, 'Td' -> 33 …"""
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"bad card string: {s!r}")
    r = RANKS.index(s[0].upper())
    u = SUITS.index(s[1].lower())
    return make_card(r, u)


def cards_from_str(s: str) -> List[int]:
    """'As Kd' or 'AsKd' -> [51, 47]"""
    s = s.replace(",", " ").strip()
    if " " in s:
        parts = s.split()
    else:
        parts = [s[i : i + 2] for i in range(0, len(s), 2)]
    return [card_from_str(p) for p in parts]


def card_to_str(card: int) -> str:
    return RANKS[rank_of(card)] + SUITS[suit_of(card)]


def cards_to_str(cards: Iterable[int]) -> str:
    return " ".join(card_to_str(c) for c in cards)


def hole_class(c1: int, c2: int) -> str:
    """Canonical 169-class name of a starting hand: 'AKs', 'T9o', 'QQ'."""
    r1, r2 = rank_of(c1), rank_of(c2)
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    suited = suit_of(c1) == suit_of(c2)
    return RANKS[r1] + RANKS[r2] + ("s" if suited else "o")


ALL_HOLE_CLASSES: List[str] = []
for _i in range(12, -1, -1):
    for _j in range(12, -1, -1):
        if _i == _j:
            ALL_HOLE_CLASSES.append(RANKS[_i] * 2)
        elif _i > _j:
            ALL_HOLE_CLASSES.append(RANKS[_i] + RANKS[_j] + "s")
            ALL_HOLE_CLASSES.append(RANKS[_i] + RANKS[_j] + "o")
assert len(ALL_HOLE_CLASSES) == 169


class Deck:
    """A shuffled deck backed by a private RNG so hands are reproducible."""

    __slots__ = ("cards", "_pos", "rng")

    def __init__(self, seed: int | None = None, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random(seed)
        self.cards: List[int] = list(range(NUM_CARDS))
        self.rng.shuffle(self.cards)
        self._pos = 0

    def draw(self, n: int = 1) -> List[int]:
        if self._pos + n > NUM_CARDS:
            raise RuntimeError("deck exhausted")
        out = self.cards[self._pos : self._pos + n]
        self._pos += n
        return out

    def remaining(self) -> Sequence[int]:
        return self.cards[self._pos :]

    @classmethod
    def from_order(cls, order: Sequence[int]) -> "Deck":
        d = cls.__new__(cls)
        d.rng = random.Random(0)
        d.cards = list(order)
        d._pos = 0
        return d
