"""HUD-style opponent statistics with Bayesian smoothing.

Two families of stats are kept per player:

1. **Classic HUD**: VPIP, PFR, 3-bet, fold-to-3-bet, c-bet, fold-to-c-bet,
   aggression factor, WTSD, W$SD.
2. **Bucket rates** (the StratFormer idea): fold/call/raise frequencies in a
   grid of contexts  ``{global, preflop, flop, turn, river} x {facing a bet/raise, not}``
   ("facing" means someone bet or raised on this street; preflop with only the
   blinds posted counts as *not* facing, so an open-raise and a call of a raise
   land in different cells).
   These are what an exploitative policy will condition on.

Every frequency is a ``Beta`` posterior: ``(hits + m*k) / (opps + k)`` where
``m`` is a population prior mean and ``k`` a pseudo-count.  With 0 observed
hands you get the population prior; after ~k observations the data dominates.
That is the answer to the "only 200 hands against my friends" problem:
early on the bot plays vs. the *typical* player, and sharpens as evidence
accumulates.  ``confidence()`` tells you how far along that is.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..engine import BOARD_CARDS_BY_STREET, ActionType, Event, HandRecord, Street

# population priors (roughly a 6-max online reg pool); tune from data later
PRIORS: Dict[str, Tuple[float, float]] = {
    # name: (prior mean, pseudo-count k)
    "vpip": (0.25, 8),
    "pfr": (0.18, 8),
    "3bet": (0.07, 12),
    "fold_to_3bet": (0.55, 6),
    "cbet_flop": (0.60, 6),
    "fold_to_cbet": (0.50, 6),
    "wtsd": (0.28, 8),
    "wsd": (0.52, 6),
    "limp": (0.08, 8),
}

STREET_KEYS = ["global", "preflop", "flop", "turn", "river"]


@dataclass
class Counter:
    hits: float = 0.0
    opps: float = 0.0

    def add(self, hit: bool) -> None:
        self.opps += 1
        if hit:
            self.hits += 1

    def raw(self) -> Optional[float]:
        return None if self.opps == 0 else self.hits / self.opps

    def smoothed(self, prior_mean: float, k: float) -> float:
        return (self.hits + prior_mean * k) / (self.opps + k)

    def confidence(self, k: float) -> float:
        return self.opps / (self.opps + k)


@dataclass
class BucketRates:
    """fold / call / raise counts in one context."""

    fold: float = 0.0
    call: float = 0.0
    raise_: float = 0.0

    @property
    def n(self) -> float:
        return self.fold + self.call + self.raise_

    def add(self, action: ActionType) -> None:
        if action == ActionType.FOLD:
            self.fold += 1
        elif action == ActionType.CALL:
            self.call += 1
        else:
            self.raise_ += 1

    def rates(self, prior: Tuple[float, float, float] = (0.3, 0.45, 0.25), k: float = 6.0) -> Tuple[float, float, float]:
        n = self.n + k
        return (
            (self.fold + prior[0] * k) / n,
            (self.call + prior[1] * k) / n,
            (self.raise_ + prior[2] * k) / n,
        )


@dataclass
class PlayerStats:
    hands: int = 0
    hud: Dict[str, Counter] = field(default_factory=lambda: {k: Counter() for k in PRIORS})
    postflop_bets_raises: int = 0
    postflop_calls: int = 0
    buckets: Dict[Tuple[str, bool], BucketRates] = field(
        default_factory=lambda: {(s, f): BucketRates() for s in STREET_KEYS for f in (True, False)}
    )
    # finer contexts for the opponent model: (street, kind-history on this street before the
    # decision), e.g. ("preflop", "") = first to act, ("preflop", "r") = facing a raise,
    # ("preflop", "a") = facing a shove, ("preflop", "c") = BB after a limp.  Kinds: f c r a.
    hist_buckets: Dict[Tuple[str, str], BucketRates] = field(default_factory=dict)
    # showdown-only evidence: for hands the player REVEALED, the strength (0..1) of the hand
    # behind each of its actions, per (street, kind-history, fold/call/raise): [sum, count].
    # A biased sample (only lines that reached showdown) - see exploit/model.py for how it is used.
    shown: Dict[Tuple[str, str, str], List[float]] = field(default_factory=dict)
    net: int = 0

    # ---- accessors
    def stat(self, name: str) -> float:
        m, k = PRIORS[name]
        return self.hud[name].smoothed(m, k)

    def raw(self, name: str) -> Optional[float]:
        return self.hud[name].raw()

    def confidence(self, name: str) -> float:
        return self.hud[name].confidence(PRIORS[name][1])

    @property
    def af(self) -> float:
        """Aggression factor (postflop bets+raises / calls), smoothed with 2/2 prior."""
        return (self.postflop_bets_raises + 2.0) / (self.postflop_calls + 2.0)

    def bucket(self, street: str, facing: bool) -> Tuple[float, float, float]:
        return self.buckets[(street, facing)].rates()

    @property
    def n_shown(self) -> float:
        return sum(v[1] for v in self.shown.values())

    def shown_mean(self, street: str, hist: str, kind: str) -> Optional[Tuple[float, float]]:
        """(mean strength, count) of revealed hands that took ``kind`` in this context, or None."""
        v = self.shown.get((street, hist, kind))
        return None if not v or v[1] == 0 else (v[0] / v[1], v[1])

    def feature_vector(self) -> List[float]:
        """15 bucket-rate numbers (5 contexts x fold/call/raise), StratFormer style,
        plus the classic HUD stats.  This is the input for an exploitative policy."""
        v: List[float] = []
        for s in STREET_KEYS:
            for facing in (True, False):
                v.extend(self.bucket(s, facing))
        v.extend(self.stat(k) for k in PRIORS)
        v.append(min(self.af, 10.0) / 10.0)
        return v

    def summary(self) -> str:
        parts = [f"n={self.hands}"]
        for k in ("vpip", "pfr", "3bet", "fold_to_3bet", "cbet_flop", "fold_to_cbet", "wtsd", "wsd"):
            r = self.raw(k)
            parts.append(f"{k}={self.stat(k) * 100:4.0f}%" + (f"({self.hud[k].opps:.0f})" if r is not None else "(-)"))
        parts.append(f"AF={self.af:.1f}")
        return " ".join(parts)


def default_strength(hole: Sequence[int], board: Sequence[int], rng: random.Random) -> float:
    """0..1 hand strength used for showdown evidence: preflop = 169-class percentile,
    postflop = Monte-Carlo equity vs one random hand on the board as it was on that street."""
    from ..equity import equity_vs_random, hole_percentile

    if not board:
        return hole_percentile(hole[0], hole[1])
    return equity_vs_random(hole, board, 1, 60, rng)


class StatsTracker:
    """Consumes ``HandRecord``s and maintains ``PlayerStats`` keyed by player id."""

    def __init__(self, strength_fn: Optional[Callable[[Sequence[int], Sequence[int], random.Random], float]] = None,
                 seed: int = 0) -> None:
        self.players: Dict[str, PlayerStats] = {}
        self.strength_fn = strength_fn or default_strength
        self.rng = random.Random(seed)

    def get(self, pid: str) -> PlayerStats:
        if pid not in self.players:
            self.players[pid] = PlayerStats()
        return self.players[pid]

    def observe_hand(self, rec: HandRecord, ids: List[str]) -> None:
        """``ids[seat]`` maps seats to persistent player identities."""
        n = rec.n_players
        st = [self.get(ids[s]) for s in range(n)]
        for s in range(n):
            st[s].hands += 1
            st[s].net += rec.net[s]

        vpip = [False] * n
        pfr = [False] * n
        limped = [False] * n
        pf_aggressor: Optional[int] = None
        street_aggressor: Optional[int] = None
        cur_street = Street.PREFLOP
        pf_raiser_seen_3bet = set()
        bb_seat = (rec.button + (1 if n == 2 else 2)) % n

        hist_kinds = ""  # kind-history of the current street: f / c / r / a
        for e in rec.events:
            if e.street != cur_street:
                cur_street = e.street
                street_aggressor = None
                hist_kinds = ""
            p = st[e.seat]
            sk = ["preflop", "flop", "turn", "river"][e.street]
            facing = e.facing_raise  # a bet/raise on this street; a posted blind alone is not "facing"
            p.buckets[(sk, facing)].add(e.action.type)
            p.buckets[("global", facing)].add(e.action.type)
            p.hist_buckets.setdefault((sk, hist_kinds), BucketRates()).add(e.action.type)
            hist_kinds += event_kind(e)

            if e.street == Street.PREFLOP:
                if e.action.type != ActionType.FOLD and e.paid > 0:
                    vpip[e.seat] = True
                if e.action.type == ActionType.RAISE:
                    pfr[e.seat] = True
                    if e.raises_this_street == 1:
                        p.hud["3bet"].add(True)
                    pf_aggressor = e.seat
                elif e.raises_this_street == 1:
                    p.hud["3bet"].add(False)
                if e.raises_this_street == 0 and e.action.type == ActionType.CALL and e.seat != bb_seat:
                    limped[e.seat] = True
                # fold to 3bet: I was the (only) raiser, now facing a 3-bet
                if e.raises_this_street == 2 and e.seat not in pf_raiser_seen_3bet and _was_first_raiser(rec, e):
                    pf_raiser_seen_3bet.add(e.seat)
                    p.hud["fold_to_3bet"].add(e.action.type == ActionType.FOLD)
            else:
                if e.action.type == ActionType.RAISE:
                    p.postflop_bets_raises += 1
                elif e.action.type == ActionType.CALL and e.to_call > 0:
                    p.postflop_calls += 1
                if e.street == Street.FLOP:
                    if e.seat == pf_aggressor and e.raises_this_street == 0 and e.to_call == 0:
                        p.hud["cbet_flop"].add(e.action.type == ActionType.RAISE)
                    if (
                        e.raises_this_street == 1
                        and street_aggressor == pf_aggressor
                        and e.seat != pf_aggressor
                        and e.to_call > 0
                    ):
                        p.hud["fold_to_cbet"].add(e.action.type == ActionType.FOLD)
                if e.action.type == ActionType.RAISE:
                    street_aggressor = e.seat

        for s in range(n):
            st[s].hud["vpip"].add(vpip[s])
            st[s].hud["pfr"].add(pfr[s])
            st[s].hud["limp"].add(limped[s])
            if rec.saw_flop[s]:
                went = s in rec.showdown_seats
                st[s].hud["wtsd"].add(went)
                if went:
                    st[s].hud["wsd"].add(s in rec.winners)

        # showdown evidence: revealed hands only (a folder never shows, so folds carry no entry)
        if rec.showdown_seats:
            hist_kinds = ""
            cur = None
            for e in rec.events:
                if e.street != cur:
                    cur = e.street
                    hist_kinds = ""
                if e.seat in rec.showdown_seats:
                    hole = rec.hole_cards[e.seat]
                    board = rec.board[: BOARD_CARDS_BY_STREET[e.street]]
                    strength = self.strength_fn(hole, board, self.rng)
                    kind = "f" if e.action.type == ActionType.FOLD else ("c" if e.action.type == ActionType.CALL else "r")
                    sk = ["preflop", "flop", "turn", "river"][e.street]
                    cell = st[e.seat].shown.setdefault((sk, hist_kinds, kind), [0.0, 0.0])
                    cell[0] += strength
                    cell[1] += 1
                hist_kinds += event_kind(e)

    def report(self, ids: Optional[List[str]] = None) -> str:
        keys = ids or list(self.players)
        width = max(len(k) for k in keys) if keys else 6
        return "\n".join(f"{k:>{width}}: {self.players[k].summary()}" for k in keys if k in self.players)


def event_kind(e: Event) -> str:
    """One letter per action for kind-histories: f, c, r (bet/raise), a (all-in raise)."""
    if e.action.type == ActionType.FOLD:
        return "f"
    if e.action.type == ActionType.CALL:
        return "c"
    return "a" if e.all_in else "r"


def _was_first_raiser(rec: HandRecord, ev: Event) -> bool:
    for e in rec.events:
        if e is ev:
            return False
        if e.street == Street.PREFLOP and e.action.type == ActionType.RAISE:
            return e.seat == ev.seat
    return False
