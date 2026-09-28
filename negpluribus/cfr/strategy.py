"""Strategy interface: information-set key -> distribution over abstract actions."""
from __future__ import annotations

import json
import math
import re
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from ..abstraction.actions import ALL_IN


class Strategy:
    """Maps an information-set key to action probabilities."""

    actions: Sequence[str]

    def policy(self, infoset: Hashable) -> List[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, infoset: Hashable, rng) -> str:
        probs = self.policy(infoset)
        r = rng.random()
        acc = 0.0
        for a, p in zip(self.actions, probs):
            acc += p
            if r < acc:
                return a
        return self.actions[-1]


class TabularStrategy(Strategy):
    def __init__(self, actions: Sequence[str], table: Dict[Hashable, List[float]] | None = None):
        self.actions = list(actions)
        self.table: Dict[Hashable, List[float]] = table or {}

    def policy(self, infoset: Hashable) -> List[float]:
        p = self.table.get(infoset)
        if p is None:
            n = len(self.actions)
            return [1.0 / n] * n
        return p

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"actions": self.actions, "table": {str(k): v for k, v in self.table.items()}}, f)

    @classmethod
    def load(cls, path: str) -> "TabularStrategy":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["actions"], d["table"])

    def items(self) -> Iterable:
        return self.table.items()


_DECIMAL = re.compile(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def size_fraction(name) -> Optional[float]:
    """The pot fraction of a grid raise size ("r0.5" -> 0.5: the names ``BetGrid`` gives its sizes); None for "f",
    "c", "a" and any other name.  csrc/persist.h ``size_fraction`` (std::from_chars: the same decimal syntax, both
    correctly rounded)."""
    if not isinstance(name, str) or len(name) < 2 or name[0] != "r" or not _DECIMAL.fullmatch(name, 1):
        return None
    f = float(name[1:])
    return f if math.isfinite(f) else None


def policy_of_row(names: Sequence[str], probs: Sequence[float],
                  legal: Sequence[str]) -> Tuple[Optional[List[float]], float]:
    """``BlueprintStrategy.policy`` for one stored row (names, probs): the probabilities aligned with ``legal`` (None
    when they sum to zero or less), and the probability moved to the all-in.

    Every legal name gets its stored probability (0.0 when the row lacks it) and the result is renormalised, with one
    exception (H4).  A raise size of the row that ``legal`` lacks and that is larger than every raise size ``legal``
    shares with the row reaches the stack here: chips grow with the pot fraction, and ``BetGrid.abstract_actions``
    lists the all-in in place of every size that hits the stack.  That happens when a row is played at a shallower
    stack than it was trained for (the 200bb blueprint at a 50bb table, a depth-grid point above the table's stack) or
    after an off-grid bet.  The probability of such a size goes to "a", the nearest legal action by amount, when "a"
    is legal, instead of being renormalised away (a row with r3 = 80 % would otherwise call or fold instead of
    shoving, and a row entirely on such sizes would give None: check / call, "off the map").  Other missing names (a
    size clamped onto a smaller one, anything when the all-in is not legal) are renormalised away as before, and at
    the row's own node every stored name is legal, so nothing changes there.

    csrc/persist.h ``BlueprintTable::policy_at`` computes the same floats in the same order.  The one case the two
    tell apart differently: a C++ table none of whose rows has "a" cannot recognise a legal "a" (no index for it) and
    moves nothing."""
    lookup = dict(zip(names, probs))
    out = [lookup.get(a, 0.0) for a in legal]
    moved = _move_collapsed_sizes(names, probs, legal, out)
    s = sum(out)
    if s <= 0:
        return None, moved
    return [x / s for x in out], moved


def _move_collapsed_sizes(names: Sequence[str], probs: Sequence[float], legal: Sequence[str], out: List[float]) -> float:
    """Adds the probability of the row's sizes that collapsed into the all-in (see ``policy_of_row``) to the all-in's
    entry of ``out``, one by one in row order; returns the total moved."""
    in_legal = set(legal)
    if ALL_IN not in in_legal or all(n in in_legal for n in names):
        return 0.0  # no all-in to move to, or nothing missing (always so at the row's own node)
    in_row = set(names)
    top = -math.inf  # the largest raise size legal shares with the row
    for a in legal:
        f = size_fraction(a) if a in in_row else None
        if f is not None and f > top:
            top = f
    i = legal.index(ALL_IN)
    moved = 0.0
    for n, p in zip(names, probs):
        f = size_fraction(n)
        if f is None or f <= top or n in in_legal:
            continue
        out[i] += p
        moved += p
    return moved


class BlueprintStrategy:
    """Key -> (abstract action names, probabilities).  Each key carries its own action list
    because the legal grid differs from spot to spot (no fold when checking is free, no
    raise when capped, ...).  ``policy(key, legal)`` returns probabilities aligned with
    ``legal`` (``policy_of_row``: raise sizes that are the all-in at this stack give their
    probability to "a"); ``None`` when the key was never visited in training."""

    def __init__(self, table: Dict[str, Tuple[List[str], List[float]]] | None = None):
        self.table: Dict[str, Tuple[List[str], List[float]]] = table or {}

    def __len__(self) -> int:
        return len(self.table)

    def policy(self, key: str, legal: Sequence[str]) -> Optional[List[float]]:
        entry = self.table.get(key)
        if entry is None:
            return None
        return policy_of_row(entry[0], entry[1], legal)[0]

    def get(self, key: str) -> Optional[Tuple[List[str], List[float]]]:
        """(names, probs) of a key, or None (as ``CppBlueprint.get``)."""
        return self.table.get(key)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"table": {k: [names, [round(p, 5) for p in probs]] for k, (names, probs) in self.table.items()}},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BlueprintStrategy":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls({k: (v[0], v[1]) for k, v in d["table"].items()})
