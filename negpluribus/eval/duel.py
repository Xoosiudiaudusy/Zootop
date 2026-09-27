"""Duplicate matches with the card-luck correction and timing, for slow agents (the search agent).

``duplicate_duel`` deals and seeds exactly as ``duplicate_match`` (the same decks, buttons, seats and
agent seeds for the same ``seed``, so its raw numbers are the same), and adds per hand:

* the **chance-node correction** (heads-up): at every card deal of the hand (the hole cards, then
  each street the hand reached), (equity after - equity before) x the pot at the deal, for the hero
  against the other player's known hand.  Equity is a martingale over the cards still to come, so
  this has mean zero whatever the players do, and removing it keeps the estimate unbiased (the
  chance-node part of AIVAT, Burch et al. 2018; scripts/slumbot_luck.py applies it to Slumbot logs).
  Equities are exact (``_fastcore.equity_vs_hand`` enumerates every board completion).
* the **seconds** the hero spent in ``act`` and the wall time of the hand.

Results per deal (all of the hero's seats of one deck), in big blinds; ``DuelResult.bb100(corrected)``
gives bb/100 with a 95% CI that treats a deal as one sample.  ``on_deal`` is called after every deal
(progress, logging).
"""
from __future__ import annotations

import math
import random
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..agents.base import Agent
from ..cards import Deck
from ..engine import Action, HandRecord, HandState, Observation, Street
from ..table import play_hand


def _equity_fn(threads: int = 8):
    from ..fast import core

    c = core()
    if c is not None and hasattr(c, "equity_vs_hand"):
        def equity(h, o, board):  # the preflop enumeration (1.7M boards) on threads; exact either way
            return c.equity_vs_hand(h, o, board, threads if not board else 1)

        return equity
    from ..evaluator import evaluate
    import itertools

    def equity(h, o, board):  # the same enumeration in Python (slow preflop; tests only)
        dead = set(h) | set(o) | set(board)
        rest = [x for x in range(52) if x not in dead]
        won = n = 0.0
        for extra in itertools.combinations(rest, 5 - len(board)):
            b = list(board) + list(extra)
            a1, a2 = evaluate(list(h) + b), evaluate(list(o) + b)
            won += 1.0 if a1 > a2 else 0.5 if a1 == a2 else 0.0
            n += 1
        return won / n

    return equity


_PREFLOP_MEMO: Dict[Tuple[int, int, int, int], float] = {}


def deal_pots(record: HandRecord) -> List[int]:
    """Heads-up: the pot (matched chips of both players) when each card deal of the hand happened:
    [hole cards, flop, turn, river], as far as the hand dealt cards (a fold stops the deals; an
    all-in runs the board out)."""
    st = HandState(list(record.starting_stacks), record.button, record.sb, record.bb, record.ante,
                   deck=Deck.from_order(list(range(52))), max_street=Street.RIVER)
    contrib = [p.invested for p in st.players]
    pots = [sum(contrib)]                      # the hole cards are dealt with the blinds in
    street = Street.PREFLOP
    for ev in record.events:
        while ev.street > street:              # a new street began before this action
            street = Street(street + 1)
            pots.append(2 * min(contrib))
        contrib[ev.seat] += ev.paid
    n_deals = {0: 1, 3: 2, 4: 3, 5: 4}[len(record.board)]
    while len(pots) < n_deals:                 # streets dealt after the last action (all-in run-out)
        pots.append(2 * min(contrib))
    return pots[:n_deals]


def chance_correction(record: HandRecord, seat: int, equity=None) -> float:
    """Heads-up card luck of ``seat`` in chips: sum over the deals of (equity after - equity
    before) x the pot at the deal (equity before the hole cards: 1/2).  Subtract it from the net."""
    if record.n_players != 2:
        raise ValueError("the chance-node correction here is heads-up only")
    equity = equity or _equity_fn()
    h, o = list(record.hole_cards[seat]), list(record.hole_cards[1 - seat])
    pots = deal_pots(record)
    key = (min(h), max(h), min(o), max(o))
    e0 = _PREFLOP_MEMO.get(key)
    if e0 is None:
        e0 = equity(h, o, [])
        if len(_PREFLOP_MEMO) > 200_000:
            _PREFLOP_MEMO.clear()
        _PREFLOP_MEMO[key] = e0
        _PREFLOP_MEMO[(min(o), max(o), min(h), max(h))] = 1.0 - e0
    eqs = [e0] + [equity(h, o, list(record.board[:nb])) for nb in (3, 4, 5)][: len(pots) - 1]
    corr = (eqs[0] - 0.5) * pots[0]
    for k in range(1, len(pots)):
        corr += (eqs[k] - eqs[k - 1]) * pots[k]
    return corr


def _ci(xs: Sequence[float], per: float) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, float("inf")
    m = sum(xs) / n
    if n < 2:
        return m / per * 100.0, float("inf")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m / per * 100.0, 1.96 * math.sqrt(var / n) / per * 100.0


@dataclass
class DuelResult:
    hero: str
    villains: List[str]
    n_rotations: int
    bb: int
    raw: List[float] = field(default_factory=list)          # per deal, bb (hero's net over its seats)
    corrected: List[float] = field(default_factory=list)    # per deal, bb, card luck removed (heads-up)
    hero_seconds: List[float] = field(default_factory=list)  # per hand
    hand_seconds: List[float] = field(default_factory=list)  # per hand
    hero_decisions: int = 0

    @property
    def n_deals(self) -> int:
        return len(self.raw)

    @property
    def n_hands(self) -> int:
        return self.n_deals * self.n_rotations

    def bb100(self, corrected: bool = False) -> Tuple[float, float]:
        """(bb/100, 95% CI half-width), one deal = one sample."""
        return _ci(self.corrected if corrected else self.raw, self.n_rotations)

    def seconds_per_hand(self) -> Tuple[float, float]:
        """(hero's seconds in act per hand, wall seconds per hand)."""
        n = max(1, len(self.hand_seconds))
        return sum(self.hero_seconds) / n, sum(self.hand_seconds) / n

    def line(self) -> str:
        m, ci = self.bb100()
        out = f"{m:+8.1f} bb/100 (95% CI +/-{ci:.1f}"
        if self.corrected:
            mc, cic = self.bb100(corrected=True)
            out += f"; luck-corrected {mc:+.1f} +/-{cic:.1f}"
        hs, ws = self.seconds_per_hand()
        return out + f"; {self.n_hands} hands, {hs:.2f}s hero / {ws:.2f}s per hand)"


class _Timed(Agent):
    """Forwards to an agent and adds up the seconds of its ``act`` calls (and, for an agent with
    ``decision_info()``, keeps what it says about each decision of the current hand)."""

    def __init__(self, inner: Agent):
        self.inner = inner
        self.name = inner.name
        self.seconds = 0.0
        self.decisions = 0
        self.infos: List[dict] = []
        self._info = getattr(inner, "decision_info", None)

    def reset(self, seed: Optional[int] = None) -> None:
        self.inner.reset(seed)

    def act(self, obs: Observation) -> Action:
        t = time.perf_counter()
        a = self.inner.act(obs)
        dt = time.perf_counter() - t
        self.seconds += dt
        self.decisions += 1
        info = dict(self._info()) if callable(self._info) else {}
        info.update(street=int(obs.street), s=round(dt, 3), action=[int(a.type), int(a.amount)])
        self.infos.append(info)
        return a

    def end_hand(self, record: HandRecord, my_seat: int) -> None:
        self.inner.end_hand(record, my_seat)


def duplicate_duel(
    hero: Agent,
    villains: Sequence[Agent],
    n_deals: int = 200,
    seed: int = 0,
    sb: int = 50,
    bb: int = 100,
    stack_bb: int = 100,
    max_street: Street = Street.RIVER,
    luck: bool = True,
    first_deal: int = 0,
    on_deal: Optional[Callable[[int, DuelResult, List[HandRecord]], None]] = None,
    on_hand: Optional[Callable[[int, int, HandRecord, Optional[float], List[dict]], None]] = None,
) -> DuelResult:
    """``duplicate_match`` with the heads-up card-luck correction and timing (see the module doc).
    ``first_deal``: start at this deal index (to continue a match in pieces with the same decks).
    ``on_hand(deal, hero_seat, record, card_luck_bb or None, hero_decisions)`` after every hand;
    ``hero_decisions``: per hero decision its street, seconds, action and the agent's
    ``decision_info()`` when it has one."""
    lineup = [hero] + list(villains)
    n = len(lineup)
    names = [a.name for a in lineup]
    res = DuelResult(hero=hero.name, villains=names[1:], n_rotations=n, bb=bb)
    stacks = [stack_bb * bb] * n
    timed = _Timed(hero)
    do_luck = luck and n == 2
    equity = _equity_fn() if do_luck else None
    for d in range(first_deal, first_deal + n_deals):
        deal_rng = random.Random(seed * 1_000_003 + d)
        deck_order = list(range(52))
        deal_rng.shuffle(deck_order)
        button = d % n
        raw = corr = 0.0
        records = []
        for r in range(n):
            seats: List[Optional[Agent]] = [None] * n
            for i, a in enumerate(lineup):
                seats[(r + i) % n] = a
            for a in lineup:  # the same seeds as duplicate_match
                tag = b"hero" if a is hero else a.name.encode()
                a.reset(seed=(seed * 7_919 + d * 31 + r) ^ zlib.crc32(tag))
            seats[r] = timed
            s0, k0 = timed.seconds, timed.decisions
            timed.infos = []
            t = time.perf_counter()
            rec = play_hand(seats, stacks, button, sb, bb, deck=Deck.from_order(deck_order), max_street=max_street)
            res.hand_seconds.append(time.perf_counter() - t)
            res.hero_seconds.append(timed.seconds - s0)
            res.hero_decisions += timed.decisions - k0
            x = rec.net[r] / bb
            raw += x
            luck_bb = None
            if do_luck:
                luck_bb = chance_correction(rec, r, equity) / bb
                corr += x - luck_bb
            records.append(rec)
            if on_hand is not None:
                on_hand(d, r, rec, luck_bb, timed.infos)
        res.raw.append(raw)
        if do_luck:
            res.corrected.append(corr)
        if on_deal is not None:
            on_deal(d, res, records)
    return res
