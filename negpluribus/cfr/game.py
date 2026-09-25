"""GameSpec: which (reduced) game the blueprint is solved for.

The full 6-max game with 100bb stacks and four streets is out of reach for
tabular MCCFR in pure Python.  A ``GameSpec`` pins down a smaller game that
still uses the *real* engine, cards and evaluator:

* fewer players (2 or 3), shorter stacks (10-20bb);
* betting on preflop only, or preflop + flop; afterwards the board is run out
  to showdown (``max_street``);
* a coarse bet grid.

Everything downstream (trainer, blueprint agent, evaluation) takes a spec, so
scaling up later is a matter of changing numbers, not code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..abstraction import BUCKET_KINDS, BetGrid, EquityBucketer, make_bucketer
from ..cards import Deck
from ..engine import HandState, Street


@dataclass(frozen=True)
class GameSpec:
    n_players: int = 2
    stack_bb: int = 20
    sb: int = 50
    bb: int = 100
    ante: int = 0
    max_street: Street = Street.FLOP
    preflop_fracs: Tuple[float, ...] = (1.0,)
    postflop_fracs: Tuple[float, ...] = (0.5, 1.0)
    max_raises_per_street: int = 3
    n_buckets: int = 8
    forbid_open_limp: bool = False  # True + preflop_fracs=() = pure push/fold (the HRC game)
    bucket_kind: str = "ehs"  # "ehs" (EquityBucketer) or "potential" (PotentialAwareBucketer)

    def __post_init__(self) -> None:
        if self.bucket_kind not in BUCKET_KINDS:
            raise ValueError(f"bucket_kind must be one of {BUCKET_KINDS}, got {self.bucket_kind!r}")

    @property
    def grid(self) -> BetGrid:
        return BetGrid(
            preflop_fracs=self.preflop_fracs,
            postflop_fracs=self.postflop_fracs,
            allow_all_in=True,
            max_raises_per_street=self.max_raises_per_street,
            forbid_open_limp=self.forbid_open_limp,
        )

    @property
    def stacks(self) -> Tuple[int, ...]:
        return tuple([self.stack_bb * self.bb] * self.n_players)

    def new_hand(self, deck_order: Sequence[int], button: int) -> HandState:
        return HandState(
            list(self.stacks), button, self.sb, self.bb, self.ante,
            deck=Deck.from_order(deck_order), max_street=self.max_street,
        )

    def needs_buckets(self) -> bool:
        return self.max_street > Street.PREFLOP

    def make_bucketer(self, samples: Optional[int] = None) -> EquityBucketer:
        """An unfitted bucketer of ``bucket_kind`` (``samples``: 150 for E[HS], 100 runouts per
        next-street card for potential-aware, when not given)."""
        return make_bucketer(self.bucket_kind, self.n_buckets, samples)

    def describe(self) -> str:
        return (
            f"{self.n_players} players, {self.stack_bb}bb stacks, betting through {self.max_street.name.lower()}, "
            f"grid preflop {self.preflop_fracs} / postflop {self.postflop_fracs} + all-in, "
            f"{self.n_buckets} postflop buckets ({self.bucket_kind})"
        )
