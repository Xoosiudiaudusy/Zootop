"""AIVAT for heads-up no-limit hold'em: the Python reference (it defines the numbers; csrc/aivat.h
computes the same ones fast).

Source: N. Burch, M. Schmid, M. Moravcik, D. Morrill, M. Bowling, "AIVAT: A New Variance Reduction
Technique for Agent Evaluation in Imperfect Information Games", AAAI 2018, arXiv 1612.06915 (v2).
Section and equation names below are the paper's.

What is estimated
-----------------
The expected net chips of the KNOWN player x (an agent whose strategy we can query for every hole
it could hold) against an UNKNOWN opponent y (Slumbot, a test bot, a human).  One hand record
(both holes, the board, every action with its amount) gives one AIVAT value; the mean over hands
is an unbiased estimate of x's expected net whatever y does (Theorem 1).

The estimator (paper, "AIVAT Value Estimate", Eq. 1)
----------------------------------------------------
    AIVAT(z) = sum_{z' in W} pi_Pa(z') v(z') / sum_{z' in W} pi_Pa(z')  +  sum_{H in H} k_H(z)

* Pa = {chance, x}: the players whose strategies are known.  pi_Pa(h) = the product of chance's
  and x's probabilities along h (paper, "Background").
* The partition H (paper, "AIVAT Correction Terms", conditions 1-3; its HUNL choice, "Experimental
  Results": "Each H has states with identical betting, public board cards, and private hole cards
  for any players in Po"): a part is every state with the same public actions, the same board and
  the same hole of y, x's hole c free.  Its states have equal opponent reach (condition 1), none
  is a prefix of another (condition 2), and the action sets agree (condition 3: x's legal actions
  depend on the public state only).
* The correction term of a part H the hand passed through, with a_O the observed action or chance
  outcome there (paper, the displayed formula for k_H(z)):

      k_H(z) = sum_a sum_{h in H} pi_Pa(h a) u_h(a) / sum_{h in H} pi_Pa(h)
             - sum_{h in H} pi_Pa(h a_O) u_h(a_O) / sum_{h in H} pi_Pa(h a_O)

  Lemma 1: E_z[k_H(z)] = 0 for ANY functions u_h(a); so every term below has mean zero on its own.
* The base value (paper, "AIVAT Base Value"): imaginary observations of every hole x could hold,
  weighted by pi_Pa (Bowling et al. 2008, "Example 3: Private Information"); W = terminal states
  with the same public actions, board and y hole (the matching partition of the paper).

What that means here, per hand (all sums over the 1326 combos c of x; d = y's actual hole)
------------------------------------------------------------------------------------------
    R(c)            x's reach: the product of x's probabilities of its actions so far with hole c,
                    averaged over x's private randomness (the translation coins, below)
    w(c)            1[c disjoint from d and the board so far] * R(c)   (pi_Pa(h) up to a factor
                    common to the whole part: the card probabilities are uniform on disjoint deals)
    root term       C_pos - mean_{c disjoint from d} u_root(pos, c, d)          (y's hole deal)
                    + (C_SB + C_BB) / 2 - C_pos                                   (the seat deal)
                    The paper models the seat as a 50/50 chance event with its own term
                    ("Experimental Results"); C_pos = mean over all disjoint (c, d) of u_root.
    x decision      sum_a sum_c w(c) s(c,a) u(c, a) / sum_c w(c)
                    - sum_c w(c) s(c,a_O) u(c, a_O) / sum_c w(c) s(c,a_O)
    board deal f    sum_c w(c) E_f'[u(c, f')] / sum_c w(c)
                    - sum_{c disjoint from f_O} w(c) u(c, f_O) / sum_{c disjoint from f_O} w(c)
                    (flop, turn and river each one chance node, also when an all-in runs them out;
                    x's own hole deal gets no term: it is inside the imaginary observations, as in
                    the paper's Figure 1)
    base            sum_{c disjoint from d, board} R(c) net_x(c) / sum R(c)
    y decision      no term (y's strategy is unknown)

x's private randomness.  The blueprint agent translates every raise it sees with the
pseudo-harmonic coin (``BetGrid.from_concrete`` with ``event_rng``), one coin per event per hand,
reused at all its later decisions.  The coins are private chance of x, independent of its hole:
the states of a part differ in (c, coins), pi_Pa multiplies the coin probabilities in, and the
reach is R(c) = sum_t P(t) prod_k sigma(c, a_k | t) over the coin outcomes t (not a product of
per-decision averages: one coin feeds several decisions).  The value functions do not depend on
the coins, so sum_h pi_Pa(h a) u_h(a) = sum_c u(c, a) sum_t P(t) R_t(c) sigma(c, a | t).

Unbiasedness needs (i) sigma exactly as the agent samples (sampling law of ``BlueprintAgent.act``
including its check/call fallback on unknown keys, and the coin law of its translation, both to
2^-53, the resolution of ``random.random()``) and (ii) u identical in both halves of a term:
branch values are computed once per node and used in the expectation and in the taken branch.
Where the expectation over a chance node cannot be enumerated (the flop, the turn), its first half
is an unbiased Monte-Carlo estimate of E_f[u(c, f)] drawn independently of the dealt cards, which
keeps E[k_H] = 0 (docs/aivat.md, "Unbiased with noisy values").

Value functions: ``values`` objects with ``root(pos, d)``, ``root_mean(pos)`` and ``branch(...)``;
negpluribus/eval/aivat_values.py holds the blueprint self-play heuristic (the one fixed for the
evaluations, docs/aivat.md) and a check-down equity heuristic for tests.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..abstraction import BetGrid
from ..abstraction.actions import pseudo_harmonic
from ..cards import Deck
from ..engine import Action, ActionType, Event, HandState, Street, CALL, FOLD, raise_to

M64 = (1 << 64) - 1
TWO53 = float(1 << 53)

# hole combos (a < b) in the order of csrc/search.h ComboTable
COMBOS: List[Tuple[int, int]] = [(a, b) for a in range(52) for b in range(a + 1, 52)]
N_COMBOS = len(COMBOS)
COMBO_INDEX: Dict[Tuple[int, int], int] = {}
for _i, (_a, _b) in enumerate(COMBOS):
    COMBO_INDEX[(_a, _b)] = _i
    COMBO_INDEX[(_b, _a)] = _i
COMBO_MASK = np.array([(1 << a) | (1 << b) for a, b in COMBOS], dtype=np.int64)
COMBO_C0 = np.array([a for a, _ in COMBOS], dtype=np.int64)
COMBO_C1 = np.array([b for _, b in COMBOS], dtype=np.int64)


def card_mask(cards: Iterable[int]) -> int:
    m = 0
    for c in cards:
        m |= 1 << int(c)
    return m


def disjoint(cards: Iterable[int]) -> np.ndarray:
    """Per combo: True if it shares no card with ``cards``."""
    return (COMBO_MASK & np.int64(card_mask(cards))) == 0


def combo_index(hole: Sequence[int]) -> int:
    return COMBO_INDEX[(int(hole[0]), int(hole[1]))]


# ---------------------------------------------------------------------------------------- RNG
def mix64(x: int) -> int:
    """splitmix64 finalizer (Steele, Lea, Flood 2014); csrc/aivat.h ``aivat_mix64``."""
    x &= M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def stream_seed(*parts: int) -> int:
    """Seed of one random stream from integer parts (global seed, hand, node, combo, rollout)."""
    h = 0x6A09E667F3BCC908
    for p in parts:
        h = mix64(((h ^ (int(p) & M64)) + 0x9E3779B97F4A7C15) & M64)
    return h


class CounterRng:
    """splitmix64 stream: state += golden gamma, output = mix64(state); uniform = top 53 bits."""

    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = seed & M64

    def next64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        return mix64(self.state)

    def uniform(self) -> float:
        return (self.next64() >> 11) * (1.0 / TWO53)


# ------------------------------------------------------------------------------- hand records
@dataclass(frozen=True)
class AivatHand:
    """One finished heads-up hand as AIVAT needs it."""

    hand_id: int                           # stable id (seeds; which log line)
    stacks: Tuple[int, int]                # before blinds
    button: int                            # seat of the button (= small blind heads-up)
    sb: int
    bb: int
    known_seat: int                        # seat of x (the known strategy)
    holes: Tuple[Tuple[int, int], Tuple[int, int]]  # by seat
    board: Tuple[int, ...]                 # the board as dealt (0, 3, 4 or 5 cards)
    actions: Tuple[Tuple[int, int], ...]   # (ActionType, raise-to amount) in order
    net: Optional[int] = None              # logged net chips of x (cross-check)
    pair: Optional[int] = None             # duplicate deal id (the CI treats a deal as one sample)
    # per decision of x, in order: None (the blueprint model plays it) or a LoggedRows (x's strategy for every
    # hole as the agent logged it: docs/aivat.md section 7)
    x_rows: Optional[Tuple[Optional["LoggedRows"], ...]] = None

    @property
    def other_seat(self) -> int:
        return 1 - self.known_seat

    def deck_order(self) -> List[int]:
        """Seat 0's hole, seat 1's hole, the board, then every other card (the engine's order)."""
        order = list(self.holes[0]) + list(self.holes[1]) + list(self.board)
        used = set(order)
        return order + [c for c in range(52) if c not in used]

    def new_state(self) -> HandState:
        return HandState(list(self.stacks), self.button, self.sb, self.bb, 0,
                         deck=Deck.from_order(self.deck_order()), max_street=Street.RIVER)


def hand_from_slumbot(rec: dict, hand_id: Optional[int] = None) -> Optional[AivatHand]:
    """A Slumbot log record (negpluribus/slumbot/runner.py) -> AivatHand, or None when the hand
    cannot be scored (no result, Slumbot's cards unknown, a replay that does not reach the end)."""
    from ..slumbot.adapter import make_deck, replay
    from ..slumbot.protocol import BIG_BLIND, SMALL_BLIND, STACK_SIZE, BUTTON_SEAT, hero_seat, parse_cards

    if rec.get("type") != "hand" or rec.get("status") != "ok" or rec.get("winnings") is None:
        return None
    if not rec.get("bot_cards") or rec.get("check") not in ("match", "unverified"):
        return None
    hero = hero_seat(int(rec["client_pos"]))
    hole = parse_cards(rec["hero_cards"])
    bot = parse_cards(rec["bot_cards"])
    board = parse_cards(rec.get("board") or [])
    rp = replay(rec["action"], make_deck(hero, hole, board, bot))
    st = rp.state
    if not st.is_terminal:
        return None
    holes = [None, None]
    holes[hero] = tuple(hole)
    holes[1 - hero] = tuple(bot)
    acts = tuple((int(e.action.type), int(e.action.amount)) for e in st.events)
    return AivatHand(hand_id=int(rec["hand"] if hand_id is None else hand_id), stacks=(STACK_SIZE, STACK_SIZE),
                     button=BUTTON_SEAT, sb=SMALL_BLIND, bb=BIG_BLIND, known_seat=hero,
                     holes=(holes[0], holes[1]), board=tuple(st.board), actions=acts, net=int(rec["winnings"]))


def hand_from_duel(rec: dict, stack: int = 20000, sb: int = 50, bb: int = 100, hand_id: int = 0,
                   known: str = "hero") -> AivatHand:
    """A hand line of scripts/eval_archetypes.py --log-hands (heads-up): holes, board, events
    [street, seat, type, amount], hero_seat, button, net_bb, and the starting stacks in chips ("stacks";
    lines written before it had them: ``stack`` for both seats).  The board is cut to what was dealt.
    ``known``: "hero" (the logged hero is the known player) or "villain" (the other seat: in a duel
    search agent vs blueprint agent, the blueprint; its AIVAT value estimates minus the hero's result)."""
    hero = int(rec["hero_seat"])
    if known not in ("hero", "villain"):
        raise ValueError("known: 'hero' or 'villain'")
    seat = hero if known == "hero" else 1 - hero
    net_hero = int(round(float(rec["net_bb"]) * bb))
    holes = tuple(tuple(int(c) for c in h) for h in rec["holes"])
    acts = tuple((int(e[2]), int(e[3])) for e in rec["events"])
    stacks = tuple(int(s) for s in rec["stacks"]) if rec.get("stacks") else (stack, stack)
    h = AivatHand(hand_id=hand_id, stacks=stacks, button=int(rec["button"]), sb=sb, bb=bb,
                  known_seat=seat, holes=holes, board=tuple(int(c) for c in rec["board"]), actions=acts,
                  net=net_hero if seat == hero else -net_hero, pair=int(rec["deal"]))
    st = h.new_state()
    for t, a in acts:
        st.apply(_action(t, a))
    if not st.is_terminal:
        raise ValueError(f"hand {hand_id}: the events do not finish the hand")
    return AivatHand(**{**h.__dict__, "board": tuple(st.board)})


def _action(t: int, amount: int) -> Action:
    if t == int(ActionType.RAISE):
        return raise_to(amount)
    return CALL if t == int(ActionType.CALL) else FOLD


# ------------------------------------------------------------------ logged range strategies
Q_LOG = 65535  # a logged row sums to this (docs/aivat.md section 7)


@dataclass(frozen=True)
class LoggedRows:
    """x's strategy at one decision for every hole, as the agent logged it (docs/aivat.md section 7):
    ``actions`` = [(type, amount)] in column order, ``q`` = uint16 rows for the combos (index order) that share no
    card with the board, each summing to ``Q_LOG``."""

    actions: Tuple[Tuple[int, int], ...]
    q: bytes  # little-endian uint16, rows x len(actions)

    def matrix(self, board: Sequence[int]) -> np.ndarray:
        """(1326, A) probabilities q / Q_LOG; zero rows for combos on the board."""
        ok = disjoint(board)
        a = len(self.actions)
        rows = np.frombuffer(self.q, dtype="<u2").reshape(-1, a)
        if rows.shape[0] != int(ok.sum()):
            raise ValueError(f"logged rows: {rows.shape[0]} rows for {int(ok.sum())} combos off the board")
        if np.any(rows.astype(np.int64).sum(axis=1) != Q_LOG):
            raise ValueError("logged rows must sum to 65535")
        out = np.zeros((N_COMBOS, a))
        out[ok] = rows / float(Q_LOG)
        return out

    def to_json(self) -> dict:
        import base64
        import zlib

        return {"actions": [list(x) for x in self.actions], "q": base64.b64encode(zlib.compress(self.q, 9)).decode()}

    @classmethod
    def from_json(cls, d: dict) -> "LoggedRows":
        import base64
        import zlib

        return cls(tuple((int(t), int(a)) for t, a in d["actions"]), zlib.decompress(base64.b64decode(d["q"])))

    @classmethod
    def from_probs(cls, actions, probs: np.ndarray, board: Sequence[int]) -> "LoggedRows":
        """Quantize (1326, A) probabilities to rows summing to Q_LOG (largest remainders; what the agent must
        then sample from, docs/aivat.md section 7)."""
        ok = disjoint(board)
        p = np.asarray(probs, dtype=float)[ok]
        p = p / p.sum(axis=1, keepdims=True)
        raw = p * Q_LOG
        q = np.floor(raw).astype(np.int64)
        rest = Q_LOG - q.sum(axis=1)
        # the `rest` largest fractional parts of each row get one more (ties: the first column first)
        rank = np.argsort(np.argsort(-(raw - q), axis=1, kind="stable"), axis=1, kind="stable")
        q += rank < rest[:, None]
        return cls(tuple((int(t), int(a)) for t, a in actions), q.astype("<u2").tobytes())


# --------------------------------------------------------------- the agent's private randomness
class _StubRng:
    def __init__(self, u: float):
        self.u = u

    def random(self) -> float:
        return self.u


def coin_prob(p: float) -> float:
    """P(U < p) for U = random.random() = k / 2^53, k uniform on 0..2^53-1: ceil(p 2^53) / 2^53."""
    if not p > 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return math.ceil(p * TWO53) / TWO53


def translation_outcomes(grid: BetGrid, ev: Event) -> List[Tuple[str, float]]:
    """The tokens ``grid.from_concrete(ev, ev.all_in, rng)`` can give and their probabilities over
    the agent's coin (one ``rng.random()`` at most): mirrors BetGrid.from_concrete branch for branch
    (the tests check it against the method itself with stubbed coins)."""
    if ev.action.type != ActionType.RAISE:
        return [(grid.from_concrete(ev, ev.all_in, None), 1.0)]
    fracs = grid.fracs_for(ev.street)
    if ev.all_in and grid.allow_all_in:
        return [("a", 1.0)]
    if ev.raises_this_street >= grid.max_raises_per_street:  # a raise-capped node: f / c / a only (QA-1)
        return [("a" if grid.allow_all_in else "c", 1.0)]
    if not fracs:
        return [("a", 1.0)]
    x = grid.observed_frac(ev)
    x_allin = grid.all_in_frac(ev)
    if grid.allow_all_in and x_allin is not None:
        below = [f for f in sorted(fracs) if f < x_allin]
        if not below:
            return [("a", 1.0)]
        top = below[-1]
        if x > top:
            p_top = (x_allin - x) * (1 + top) / ((x_allin - top) * (1 + x))
            return _two(f"r{top:g}", "a", coin_prob(p_top))
        return _harmonic(x, below)
    return _harmonic(x, fracs)


def _harmonic(x: float, grid: Sequence[float]) -> List[Tuple[str, float]]:
    g = sorted(grid)
    if x <= g[0]:
        return [(f"r{g[0]:g}", 1.0)]
    if x >= g[-1]:
        return [(f"r{g[-1]:g}", 1.0)]
    for a, b in zip(g, g[1:]):
        if a <= x <= b:
            p_a = (b - x) * (1 + a) / ((b - a) * (1 + x))
            return _two(f"r{a:g}", f"r{b:g}", coin_prob(p_a))
    return [(f"r{g[-1]:g}", 1.0)]


def _two(low: str, high: str, p_low: float) -> List[Tuple[str, float]]:
    if p_low >= 1.0:
        return [(low, 1.0)]
    if p_low <= 0.0:
        return [(high, 1.0)]
    return [(low, p_low), (high, 1.0 - p_low)]


def sampling_law(probs: Sequence[float]) -> List[float]:
    """P(choice = i) of BlueprintAgent.act: r = random.random(); the first i with r < acc_i (acc = the
    running sum in order), else the last action.  P(r < a) = ceil(a 2^53) / 2^53 (capped at 1)."""
    n = len(probs)
    out = [0.0] * n
    acc = 0.0
    prev_f = 0.0
    for i in range(n - 1):
        acc += probs[i]
        f = min(coin_prob(acc), 1.0) if acc < 1.0 else 1.0
        out[i] = max(0.0, f - prev_f)
        prev_f = max(prev_f, f)
    out[n - 1] = max(0.0, 1.0 - prev_f)
    return out


@dataclass
class Branch:
    """One outcome combination t of x's translation coins so far: P(t), the tokens of the history
    and x's reach R_t(c) over the 1326 combos."""

    prob: float
    tokens: List[Tuple[int, str]]  # (street, token) per event
    reach: np.ndarray

    def history(self) -> str:
        parts: List[str] = []
        cur = None
        for street, tok in self.tokens:
            if street != cur:
                if cur is not None:
                    parts.append("/")
                cur = street
            parts.append(tok)
        return " ".join(parts).replace(" / ", "/")


def split_branches(branches: List[Branch], street: int, outcomes: List[Tuple[str, float]]) -> List[Branch]:
    if len(outcomes) == 1:
        tok = outcomes[0][0]
        for b in branches:
            b.tokens.append((street, tok))
        return branches
    out = []
    for b in branches:
        for tok, p in outcomes:
            out.append(Branch(b.prob * p, b.tokens + [(street, tok)], b.reach.copy()))
    return out


class BlueprintAgentModel:
    """sigma of ``negpluribus.agents.blueprint.BlueprintAgent`` (randomize_translation=True) for every
    hole: at x's decision, per coin outcome t and combo c, the probability of each concrete action.

    ``blueprint``: anything with ``policy(key, legal)`` (CppBlueprint / BlueprintStrategy);
    ``bucketer``: anything with ``bucket(hole, board)`` giving the agent's buckets (the C++ core
    bucketer: the same numbers as the Python one)."""

    def __init__(self, blueprint, bucketer, grid: BetGrid):
        self.blueprint = blueprint
        self.bucketer = bucketer
        self.grid = grid
        self._bucket_memo: Dict[Tuple[int, ...], np.ndarray] = {}

    def buckets(self, board: Sequence[int]) -> np.ndarray:
        """Bucket of every combo on ``board`` (-1 on the board)."""
        key = tuple(board)
        b = self._bucket_memo.get(key)
        if b is None:
            ok = disjoint(board)
            b = np.full(N_COMBOS, -1, dtype=np.int64)
            for i, (c0, c1) in enumerate(COMBOS):
                if ok[i]:
                    b[i] = self.bucketer.bucket([c0, c1], list(board))
            if len(self._bucket_memo) > 64:
                self._bucket_memo.clear()
            self._bucket_memo[key] = b
        return b

    def decision(self, st: HandState, seat: int, branches: List[Branch]) -> Tuple[List[Action], List[np.ndarray]]:
        """The concrete actions (distinct, in grid order) and per branch a (1326, A) array of the
        agent's probabilities (rows of combos on the board: 0)."""
        obs = st.observe(seat)
        legal = self.grid.abstract_actions(obs)
        conc = [self.grid.to_concrete(obs, n) for n in legal]
        acts: List[Action] = []
        col = []
        for a in conc:
            if a not in acts:
                acts.append(a)
            col.append(acts.index(a))
        call_col = acts.index(CALL) if CALL in acts else None
        bk = self.buckets(obs.board)
        present = sorted(set(int(b) for b in bk if b >= 0))
        prefix = f"{obs.street.name[0]}|{obs.position}|{obs.n_active}|b"
        out = []
        for br in branches:
            hist = br.history()
            rows = {}
            for b in present:
                probs = self.blueprint.policy(f"{prefix}{b}|{hist}", legal)
                row = np.zeros(len(acts))
                if probs is None:  # the agent's fallback: check / call
                    row[call_col] = 1.0
                else:
                    for j, p in enumerate(sampling_law(list(probs))):
                        row[col[j]] += p
                rows[b] = row
            sig = np.zeros((N_COMBOS, len(acts)))
            for b, row in rows.items():
                sig[bk == b] = row
            out.append(sig)
        return acts, out


# ------------------------------------------------------------------------------ the estimator
@dataclass
class Term:
    kind: str          # "root", "x", "flop", "turn", "river"
    value: float       # chips
    street: int = 0
    detail: dict = field(default_factory=dict)


@dataclass
class AivatResult:
    hand_id: int
    net: int                 # x's chips in the hand (engine)
    base: float              # imaginary-observation base value (chips)
    terms: List[Term]

    @property
    def value(self) -> float:
        return self.base + sum(t.value for t in self.terms)

    def term_sum(self, kind: str) -> float:
        return sum(t.value for t in self.terms if t.kind == kind)


def _street_groups(n_before: int, n_after: int) -> List[int]:
    """Board cards dealt between two decisions, as chance nodes: [3, 1, 1] from 0 to 5, etc."""
    out = []
    n = n_before
    while n < n_after:
        k = 3 if n == 0 else 1
        out.append(k)
        n += k
    return out


def _check_values(v: np.ndarray, mask: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(v[mask])):
        raise FloatingPointError(f"non-finite heuristic values at {what}")


def aivat_hand(hand: AivatHand, model: BlueprintAgentModel, values, *, seat_term: bool = True,
               trace: Optional[list] = None) -> AivatResult:
    """The AIVAT value of one hand for the known seat (chips) with its parts (see the module doc).
    ``trace``: a list that receives (name, value vector) per node, as the C++ evaluator's trace."""
    grid = model.grid
    x, y = hand.known_seat, hand.other_seat
    d = hand.holes[y]
    st = hand.new_state()
    ctx = values.start_hand(hand)
    branches = [Branch(1.0, [], disjoint(d).astype(float))]
    terms: List[Term] = []
    node = 0
    n_x = 0  # decisions of x so far

    # the root: y's hole deal (chance), and the seat as a 50/50 chance event (paper, HUNL setup)
    pos = 0 if hand.known_seat == hand.button else 1  # 0 = x is the small blind
    u0 = values.root(ctx, pos, d)
    if trace is not None:
        trace.append(("root", np.asarray(u0, dtype=float)))
    ok0 = disjoint(d)
    c_pos = values.root_mean(pos)
    terms.append(Term("root", c_pos - float(u0[ok0].mean()), detail={"pos": pos}))
    if seat_term:
        terms.append(Term("seat", 0.5 * (values.root_mean(0) + values.root_mean(1)) - c_pos, detail={"pos": pos}))
    node += 1

    for k, (t, amount) in enumerate(hand.actions):
        action = _action(t, amount)
        seat = st.current_player
        board0 = list(st.board)
        reused: Optional[np.ndarray] = None
        if seat == x:
            logged = hand.x_rows[n_x] if hand.x_rows is not None and n_x < len(hand.x_rows) else None
            n_x += 1
            if logged is None:
                acts, sig = model.decision(st, x, branches)
            else:  # the agent's own rows: the same for every coin outcome (a search needs no translation)
                acts = [_action(t_, a_) for t_, a_ in logged.actions]
                m = logged.matrix(st.board)
                sig = [m for _ in branches]
            if action not in acts:
                raise ValueError(f"hand {hand.hand_id}: x's action {action} is not one the agent can take {acts}")
            ia = acts.index(action)
            ok = disjoint(list(d) + board0).astype(float)
            w_c = sum(br.prob * br.reach for br in branches) * ok
            w_ca = sum(br.prob * br.reach[:, None] * s for br, s in zip(branches, sig)) * ok[:, None]
            vals = []
            for j, a in enumerate(acts):
                need = w_ca[:, j] > 0
                v = values.branch(ctx, st, a, [], node, need) if need.any() else np.zeros(N_COMBOS)
                _check_values(v, need, f"x decision {k}")
                vals.append(v)
                if trace is not None:
                    trace.append((f"x{k}:a{j}", np.asarray(v, dtype=float)))
            first = sum(float(np.dot(w_ca[:, j], vals[j])) for j in range(len(acts))) / float(w_c.sum())
            second = float(np.dot(w_ca[:, ia], vals[ia])) / float(w_ca[:, ia].sum())
            terms.append(Term("x", first - second, int(st.street),
                              {"k": k, "actions": [str(a) for a in acts], "taken": ia,
                               "sigma_bar": [float(w_ca[:, j].sum() / w_c.sum()) for j in range(len(acts))]}))
            for br, s in zip(branches, sig):
                br.reach = br.reach * s[:, ia]
            reused = vals[ia]
            decision_node = node
            node += 1
        before = st.clone()
        ev = st.apply(action)
        branches = split_branches(branches, int(ev.street), translation_outcomes(grid, ev))
        new_cards = list(st.board[len(board0):])
        if new_cards:
            fixed: List[int] = []
            for g in _street_groups(len(board0), len(st.board)):
                f_obs = new_cards[len(fixed):len(fixed) + g]
                w_c = sum(br.prob * br.reach for br in branches) * disjoint(list(d) + board0 + fixed)
                need = w_c > 0
                if reused is not None and not fixed:
                    v_before = reused  # x's own action closed the round: the same numbers
                else:
                    v_before = values.branch(ctx, before, action, list(fixed), node, need)
                _check_values(v_before, need, f"chance {k} before")
                ok2 = disjoint(f_obs)
                need2 = need & ok2
                v_after = values.branch(ctx, before, action, list(fixed) + f_obs, node, need2)
                _check_values(v_after, need2, f"chance {k} after")
                if trace is not None:
                    nbc = len(board0) + len(fixed)
                    trace.append((f"c{k}:{nbc}:before", np.asarray(v_before, dtype=float)))
                    trace.append((f"c{k}:{nbc}:after", np.asarray(v_after, dtype=float)))
                w2 = w_c * ok2
                first = float(np.dot(w_c, v_before)) / float(w_c.sum())
                second = float(np.dot(w2, v_after)) / float(w2.sum())
                street = {0: Street.FLOP, 3: Street.TURN, 4: Street.RIVER}[len(board0) + len(fixed)]
                terms.append(Term(street.name.lower(), first - second, int(street),
                                  {"k": k, "reused": reused is not None and not fixed}))
                fixed += f_obs
                node += 1
    if not st.is_terminal:
        raise ValueError(f"hand {hand.hand_id}: the actions do not finish the hand")
    rec = st.record()
    net = rec.net[x]
    # base value: every hole x could hold, weighted by its reach (imaginary observations)
    ok = disjoint(list(d) + list(st.board))
    reach = sum(br.prob * br.reach for br in branches) * ok
    v = values.terminal(ctx, st, ok)
    if trace is not None:
        trace.append(("base", np.asarray(v, dtype=float)))
    base = float(np.dot(reach, v)) / float(reach.sum())
    return AivatResult(hand.hand_id, int(net), base, terms)


# -------------------------------------------------------------------------------- statistics
def mean_ci(xs: Sequence[float], per_sample: float = 1.0) -> Tuple[float, float, float]:
    """(mean, 95% half-width, sd) of per-sample values divided by ``per_sample``."""
    a = np.asarray(xs, dtype=float) / per_sample
    n = len(a)
    if n < 2:
        return float(a.mean()) if n else 0.0, math.inf, math.inf
    sd = float(a.std(ddof=1))
    return float(a.mean()), 1.96 * sd / math.sqrt(n), sd
