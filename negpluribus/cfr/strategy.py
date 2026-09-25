"""Strategy interface: information-set key -> distribution over abstract actions."""
from __future__ import annotations

import json
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


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


class BlueprintStrategy:
    """Key -> (abstract action names, probabilities).  Each key carries its own action list
    because the legal grid differs from spot to spot (no fold when checking is free, no
    raise when capped, ...).  ``policy(key, legal)`` returns probabilities aligned with
    ``legal``; ``None`` when the key was never visited in training."""

    def __init__(self, table: Dict[str, Tuple[List[str], List[float]]] | None = None):
        self.table: Dict[str, Tuple[List[str], List[float]]] = table or {}

    def __len__(self) -> int:
        return len(self.table)

    def policy(self, key: str, legal: Sequence[str]) -> Optional[List[float]]:
        entry = self.table.get(key)
        if entry is None:
            return None
        names, probs = entry
        lookup = dict(zip(names, probs))
        out = [lookup.get(a, 0.0) for a in legal]
        s = sum(out)
        if s <= 0:
            return None
        return [x / s for x in out]

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
