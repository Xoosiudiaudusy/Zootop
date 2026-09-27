"""Value functions u_h(a) for AIVAT (negpluribus/eval/aivat.py).  Python references.

AIVAT is unbiased for ANY value function (Burch et al. 2018, Lemma 1 / Theorem 1); the value function
only decides how much variance is removed.  The one used for evaluations is fixed in docs/aivat.md
("Heuristic v1") before it touched any evaluation log (Kim & Sandholm 2026, arXiv 2605.14261).

Interface (what ``aivat_hand`` calls; all values are x's net chips, vectors over the 1326 combos c
of x with 0.0 wherever a combo is not asked for):

    start_hand(hand) -> ctx
    root(ctx, pos, d)               u at the start of the hand, x in seat ``pos`` (0 = small blind)
    root_mean(pos)                  its exact mean over every disjoint (c, d)
    branch(ctx, parent, action, fixed, node, need)
                                    u of the state reached by ``action`` from ``parent``, the next
                                    board cards being ``fixed`` and then unknown (so the value
                                    averages over them): the branch values of x's decisions and
                                    both halves of every chance term
    terminal(ctx, st, ok)           x's net in the finished hand ``st`` for every combo

Heuristic v1 (``SelfPlayValues``): the expected net of x when BOTH players play the blueprint from
the state (the key the blueprint agent builds, the pseudo-harmonic translation made deterministic,
check/call on unknown keys), x holding c, y holding its actual hole, unknown board cards uniform:
the paper's choice for HUNL was the self-play values of an abstraction equilibrium ("Experimental
Results"); ours are the self-play values of our blueprint on the full state.  Computed

  * exactly: at the end of the hand (fold, showdown), when nobody can act any more (all-in: equity
    over every run-out of at most 2 cards), on the river (the betting tree enumerated), and one
    card before river decisions (the tree for each of the 44-46 river cards);
  * by Monte-Carlo otherwise (a preflop all-in: ``eq_samples`` random boards; decisions left before
    the river: ``rollouts[street]`` blueprint self-play rollouts), each combo on its own random
    streams seeded from (seed, hand, node, combo, rollout) - independent of the cards dealt at the
    node, so the AIVAT terms keep mean zero (docs/aivat.md, "Unbiased with noisy values");
  * at the root: a table (``RootTable``) of rollout values per seat and suit-isomorphism class of
    (c, d), whose exact mean is the first half of the root term.
"""
from __future__ import annotations

import itertools
import math
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..abstraction import BetGrid
from ..abstraction.infoset import history_string
from ..cards import Deck
from ..engine import Action, HandState, Street
from .aivat import (COMBO_C0 as COMBO_C0_NP, COMBO_C1 as COMBO_C1_NP, COMBO_MASK as COMBO_MASK_NP, COMBOS, N_COMBOS,
                    AivatHand, CounterRng, combo_index, disjoint, stream_seed)

EQ_STREAM = 1_000_000  # rollout index of the Monte-Carlo equity stream


def _evaluate():
    from ..fast import core

    c = core()
    if c is not None:
        return c.evaluate
    from ..evaluator import evaluate

    return evaluate


# ------------------------------------------------------------------------------ test heuristics
class ZeroValues:
    """u = 0 everywhere: AIVAT reduces to the imaginary-observation base value (Bowling et al. 2008)."""

    def start_hand(self, hand: AivatHand):
        return {"hand": hand, "x": hand.known_seat, "y": hand.other_seat, "d": hand.holes[hand.other_seat]}

    def root(self, ctx, pos, d):
        return np.zeros(N_COMBOS)

    def root_mean(self, pos):
        return 0.0

    def branch(self, ctx, parent, action, fixed, node, need):
        return np.zeros(N_COMBOS)

    def terminal(self, ctx, st: HandState, ok: np.ndarray) -> np.ndarray:
        return terminal_net(st, ctx["x"], ctx["y"], ok)


def terminal_net(st: HandState, x: int, y: int, ok: np.ndarray) -> np.ndarray:
    """x's net in a finished hand for every combo (a fold: one number; a showdown: +-m or 0 with m =
    min(invested), the engine's side-pot result heads-up)."""
    out = np.zeros(N_COMBOS)
    if _folded(st):
        out[:] = st.players[x].stack - st.starting_stacks[x]
        out[~ok] = 0.0
        return out
    ev = _evaluate()
    board = list(st.board)
    sd = ev(list(st.players[y].hole) + board)
    m = min(st.players[x].invested, st.players[y].invested)
    for i in np.nonzero(ok)[0]:
        c0, c1 = COMBOS[i]
        sc = ev([c0, c1] + board)
        out[i] = m if sc > sd else (-m if sc < sd else 0.0)
    return out


class AdditiveCardValues:
    """A value function with no relation to real values but exact expectations over every chance
    node, to test that the terms have mean zero for ANY u (Lemma 1):

        u(c, state) = f(c, public actions) + sum over the board cards b of g(c, b, position of b)

    with f, g pseudo-random in [-scale, scale] (hashes of the arguments).  A chance node dealing k
    cards to positions p..p+k-1 out of the n cards valid for c (not in c, d or the board) has
    E[sum of g over the new cards] = sum over those positions of the mean of g over the valid cards."""

    def __init__(self, scale: float = 2000.0, seed: int = 7):
        self.scale = scale
        self.seed = seed
        i = np.arange(N_COMBOS, dtype=np.uint64)[:, None]
        b = np.arange(52, dtype=np.uint64)[None, :]
        # G[j][c, b]: g(c, b, j) for board positions j = 0..4
        self.G = np.stack([self._h(3, i, b, np.uint64(j)) for j in range(5)])
        self._rm = {}

    def _h(self, *parts):
        """scale * (2 u - 1), u the top 53 bits of stream_seed(seed, *parts), vectorised (uint64 wraps)."""
        with np.errstate(over="ignore"):
            h = np.uint64(0x6A09E667F3BCC908)
            for p in (np.uint64(self.seed),) + tuple(np.asarray(q, dtype=np.uint64) for q in parts):
                x = (h ^ p) + np.uint64(0x9E3779B97F4A7C15)
                x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                h = x ^ (x >> np.uint64(31))
            u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        return (u * 2.0 - 1.0) * self.scale

    def start_hand(self, hand):
        return {"hand": hand, "x": hand.known_seat, "y": hand.other_seat, "d": hand.holes[hand.other_seat]}

    def _root_all(self, pos):
        c = np.arange(N_COMBOS, dtype=np.uint64)[:, None]
        d = np.arange(N_COMBOS, dtype=np.uint64)[None, :]
        return self._h(1, np.uint64(pos), c, d)  # [c, d]

    def root(self, ctx, pos, d):
        di = combo_index(d)
        out = self._h(1, np.uint64(pos), np.arange(N_COMBOS, dtype=np.uint64), np.uint64(di))
        out[~disjoint(d)] = 0.0
        return out

    def root_mean(self, pos):
        if pos not in self._rm:
            u = self._root_all(pos)
            ok = (COMBO_MASK_NP[:, None] & COMBO_MASK_NP[None, :]) == 0
            self._rm[pos] = float(u[ok].mean())
        return self._rm[pos]

    def _public(self, st: HandState) -> int:
        h = 0
        for e in st.events:
            h = stream_seed(h, int(e.street), int(e.seat), int(e.action.type), int(e.action.amount))
        return h & 0xFFFFFFFF

    def _value(self, st: HandState, board: Sequence[int], d, ok: np.ndarray, n_random: int) -> np.ndarray:
        idx = np.arange(N_COMBOS, dtype=np.uint64)
        v = self._h(2, np.uint64(self._public(st)), idx)
        for j, b in enumerate(board):
            v = v + self.G[j][:, b]
        if n_random:
            known = set(d) | set(board)
            cand = np.array([b for b in range(52) if b not in known], dtype=np.int64)
            for jj in range(n_random):
                g = self.G[len(board) + jj]
                tot = g[:, cand].sum(axis=1)
                own0 = np.isin(COMBO_C0_NP, cand)
                own1 = np.isin(COMBO_C1_NP, cand)
                tot = tot - np.where(own0, g[np.arange(N_COMBOS), COMBO_C0_NP], 0.0) - np.where(own1, g[np.arange(N_COMBOS), COMBO_C1_NP], 0.0)
                n = len(cand) - own0.astype(int) - own1.astype(int)
                v = v + tot / n
        out = np.zeros(N_COMBOS)
        out[ok] = v[ok]
        return out

    def branch(self, ctx, parent: HandState, action: Action, fixed, node, need):
        x, y, d = ctx["x"], ctx["y"], ctx["d"]
        st, dealt = _apply_probe(parent, action, x, y, _any_hole(d, parent.board, fixed), d, fixed)
        if st.is_terminal and _folded(st):
            out = np.zeros(N_COMBOS)
            out[need] = st.players[x].stack - st.starting_stacks[x]
            return out
        board = list(parent.board) + list(fixed)
        n_random = max(0, dealt - len(fixed))
        if st.is_terminal:  # a showdown: the whole board, the rest random
            n_random = 5 - len(board)
        return self._value(st, board, d, need, n_random)

    def terminal(self, ctx, st, ok):
        return terminal_net(st, ctx["x"], ctx["y"], ok)


def _folded(st: HandState) -> bool:
    return sum(1 for p in st.players if not p.folded) == 1


def _apply_probe(parent: HandState, action: Optional[Action], x: int, y: int, c, d, runout: Sequence[int]) -> Tuple[HandState, int]:
    """``parent`` + ``action`` (None: no action) with x holding c, y holding d and the next board
    cards ``runout`` (then the other cards in increasing order); returns the state and how many
    board cards the action dealt."""
    st = parent.clone()
    holes = [None, None]
    holes[x], holes[y] = list(c), list(d)
    used = set(holes[0]) | set(holes[1]) | set(parent.board) | set(runout)
    fill = [k for k in range(52) if k not in used]
    order = holes[0] + holes[1] + list(parent.board) + list(runout) + fill
    deck = Deck.from_order(order)
    deck._pos = 4 + len(parent.board)
    st.deck = deck
    st.players[x].hole = list(c)
    st.players[y].hole = list(d)
    n0 = len(st.board)
    if action is not None:
        st.apply(action)
    return st, len(st.board) - n0


# ---------------------------------------------------------------------------- heuristic v1
class SelfPlayValues:
    """Heuristic v1 (docs/aivat.md): blueprint self-play values of the full state, Python reference.

    ``blueprint``: ``policy(key, legal)``; ``bucketer``: the C++ core bucketer (``bucket(hole, board)``
    and ``river_buckets_all(board)``); ``grid``: the game's BetGrid; ``root_table``: a RootTable (or
    None: root values 0); ``rollouts``: {street of the state: rollouts per combo}; ``seed``: global."""

    def __init__(self, blueprint, bucketer, grid: BetGrid, root_table=None, rollouts: Optional[Dict[int, int]] = None,
                 eq_samples: int = 2000, seed: int = 0):
        self.blueprint = blueprint
        self.bucketer = bucketer
        self.grid = grid
        self.root_table = root_table
        self.rollouts = dict(rollouts or {0: 4, 1: 8, 2: 8})
        self.eq_samples = eq_samples
        self.seed = seed
        self.evaluate = _evaluate()
        self.stats = {"rollouts": 0, "rollout_steps": 0, "river_trees": 0, "eq_mc": 0, "eq_exact": 0}

    # ----------------------------------------------------------------- interface
    def start_hand(self, hand: AivatHand):
        return {"hand": hand, "x": hand.known_seat, "y": hand.other_seat, "d": tuple(hand.holes[hand.other_seat]),
                "river_buckets": {}}

    def root(self, ctx, pos, d):
        if self.root_table is None:
            return np.zeros(N_COMBOS)
        return self.root_table.row(pos, d)

    def root_mean(self, pos):
        return 0.0 if self.root_table is None else self.root_table.mean(pos)

    def terminal(self, ctx, st, ok):
        return terminal_net(st, ctx["x"], ctx["y"], ok)

    def branch(self, ctx, parent: HandState, action: Action, fixed, node, need):
        x, y, d = ctx["x"], ctx["y"], ctx["d"]
        hand: AivatHand = ctx["hand"]
        probe, dealt = _apply_probe(parent, action, x, y, _any_hole(d, parent.board, fixed), d, fixed)
        out = np.zeros(N_COMBOS)
        idx = np.nonzero(need)[0]
        if len(idx) == 0:
            return out
        n_random = max(0, dealt - len(fixed))
        board = list(parent.board) + list(fixed)
        if probe.is_terminal:
            if _folded(probe):
                out[idx] = probe.players[x].stack - probe.starting_stacks[x]
                return out
            missing = 5 - len(board)
            m = min(probe.players[x].invested, probe.players[y].invested)
            if missing == 0:
                return self._showdown(board, d, m, idx, out)
            if missing <= 2:
                return self._equity_exact(board, d, m, idx, out)
            return self._equity_mc(ctx, board, d, m, node, idx, out)
        if n_random == 0 and probe.street == Street.RIVER:
            return self._river_tree(ctx, probe, idx, out)
        if n_random == 1 and probe.street == Street.RIVER:
            return self._river_card(ctx, parent, action, fixed, idx, out)
        return self._rollouts(ctx, parent, action, fixed, node, self.rollouts[int(probe.street)], idx, out)

    # ------------------------------------------------------------------ exact parts
    def _showdown(self, board, d, m, idx, out):
        ev = self.evaluate
        sd = ev(list(d) + list(board))
        for i in idx:
            c0, c1 = COMBOS[i]
            sc = ev([c0, c1] + list(board))
            out[i] = m if sc > sd else (-m if sc < sd else 0.0)
        return out

    def _equity_exact(self, board, d, m, idx, out):
        """m * (P(win) - P(lose)) over every run-out of the missing (1 or 2) cards, enumerated in
        increasing card order."""
        ev = self.evaluate
        known = set(d) | set(board)
        missing = 5 - len(board)
        self.stats["eq_exact"] += len(idx)
        for i in idx:
            c0, c1 = COMBOS[i]
            rest = [k for k in range(52) if k not in known and k != c0 and k != c1]
            tot = 0
            n = 0
            for extra in itertools.combinations(rest, missing):
                b = list(board) + list(extra)
                sc, sd = ev([c0, c1] + b), ev(list(d) + b)
                tot += 1 if sc > sd else (-1 if sc < sd else 0)
                n += 1
            out[i] = m * tot / n
        return out

    def _equity_mc(self, ctx, board, d, m, node, idx, out):
        """A preflop all-in: ``eq_samples`` random boards per combo, stream (seed, hand, node, combo, EQ_STREAM)."""
        ev = self.evaluate
        known = set(d) | set(board)
        missing = 5 - len(board)
        hid = ctx["hand"].hand_id
        self.stats["eq_mc"] += len(idx)
        for i in idx:
            c0, c1 = COMBOS[i]
            rng = CounterRng(stream_seed(self.seed, hid, node, int(i), EQ_STREAM))
            base_rest = [k for k in range(52) if k not in known and k != c0 and k != c1]
            tot = 0
            for _ in range(self.eq_samples):
                rest = list(base_rest)
                b = list(board)
                for _j in range(missing):
                    b.append(rest.pop(int(rng.uniform() * len(rest))))
                sc, sd = ev([c0, c1] + b), ev(list(d) + b)
                tot += 1 if sc > sd else (-1 if sc < sd else 0)
            out[i] = m * tot / self.eq_samples
        return out

    def _river_buckets(self, ctx, board) -> np.ndarray:
        key = tuple(sorted(board))
        memo = ctx["river_buckets"]
        b = memo.get(key)
        if b is None:
            r = self.bucketer.river_buckets_all(list(board))
            if r is None:  # no batch for this abstraction: one bucket() per combo, the same numbers
                ok = disjoint(board)
                b = np.full(N_COMBOS, 255, dtype=np.int64)
                for i in np.nonzero(ok)[0]:
                    b[i] = self.bucketer.bucket(list(COMBOS[i]), list(board))
            else:
                b = np.array(r, dtype=np.int64)
            memo[key] = b
        return b

    def _policy_row(self, st: HandState, seat: int, bucket: int) -> Tuple[List[str], List[Action], Optional[List[float]]]:
        obs = st.observe(seat)
        legal = self.grid.abstract_actions(obs)
        hist = history_string(st.events, self.grid, None)
        key = f"{obs.street.name[0]}|{obs.position}|{obs.n_active}|b{bucket}|{hist}"
        probs = self.blueprint.policy(key, legal)
        acts = [self.grid.to_concrete(obs, n) for n in legal]
        return legal, acts, (list(probs) if probs is not None else None)

    def _river_tree_ab(self, ctx, st: HandState, b_needed: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """(A, S) per x bucket of the river subtree at st: V(c) = A[b(c)] + S[b(c)] * s(c), s(c) = +1/0/-1
        as x's hand beats / ties / loses to d."""
        x, y, d = ctx["x"], ctx["y"], ctx["d"]
        board = list(st.board)
        by = int(self.bucketer.bucket(list(d), board))
        self.stats["river_trees"] += 1

        def walk(s: HandState) -> Tuple[np.ndarray, np.ndarray]:
            nb = len(b_needed)
            if s.is_terminal:
                if _folded(s):
                    return np.full(nb, float(s.players[x].stack - s.starting_stacks[x])), np.zeros(nb)
                m = float(min(s.players[x].invested, s.players[y].invested))
                return np.zeros(nb), np.full(nb, m)
            seat = s.current_player
            if seat == y:
                legal, acts, probs = self._policy_row(s, y, by)
                p = _probs_or_call(legal, probs)
                A = np.zeros(nb)
                S = np.zeros(nb)
                for j, a in enumerate(acts):
                    if p[j] == 0.0:
                        continue
                    ch = s.clone()
                    ch.apply(a)
                    a2, s2 = walk(ch)
                    A += p[j] * a2
                    S += p[j] * s2
                return A, S
            rows = []
            acts = None
            for b in b_needed:
                legal, acts, probs = self._policy_row(s, x, int(b))
                rows.append(_probs_or_call(legal, probs))
            P = np.array(rows)  # (nb, n_actions)
            A = np.zeros(nb)
            S = np.zeros(nb)
            for j, a in enumerate(acts):
                if not P[:, j].any():
                    continue
                ch = s.clone()
                ch.apply(a)
                a2, s2 = walk(ch)
                A += P[:, j] * a2
                S += P[:, j] * s2
            return A, S

        A, S = walk(st)
        return {int(b): float(A[i]) for i, b in enumerate(b_needed)}, {int(b): float(S[i]) for i, b in enumerate(b_needed)}

    def _river_tree(self, ctx, st: HandState, idx, out):
        board = list(st.board)
        bk = self._river_buckets(ctx, board)
        needed = sorted(set(int(bk[i]) for i in idx))
        A, S = self._river_tree_ab(ctx, st, needed)
        ev = self.evaluate
        sd = ev(list(ctx["d"]) + board)
        for i in idx:
            c0, c1 = COMBOS[i]
            sc = ev([c0, c1] + board)
            s = 1.0 if sc > sd else (-1.0 if sc < sd else 0.0)
            b = int(bk[i])
            out[i] = A[b] + S[b] * s
        return out

    def _river_card(self, ctx, parent, action, fixed, idx, out):
        """One card before river decisions: the exact river value averaged over the river card."""
        x, y, d = ctx["x"], ctx["y"], ctx["d"]
        board = list(parent.board) + list(fixed)
        known = set(d) | set(board)
        cards = [r for r in range(52) if r not in known]
        tot = np.zeros(N_COMBOS)
        cnt = np.zeros(N_COMBOS)
        for r in cards:
            sub = np.array([i for i in idx if r not in COMBOS[i]], dtype=np.int64)
            if len(sub) == 0:
                continue
            probe, _ = _apply_probe(parent, action, x, y, _any_hole(d, board + [r], []), d, list(fixed) + [r])
            vals = self._river_tree(ctx, probe, sub, np.zeros(N_COMBOS))
            tot[sub] += vals[sub]
            cnt[sub] += 1
        out[idx] = tot[idx] / cnt[idx]
        return out

    # ------------------------------------------------------------------ Monte-Carlo parts
    def _rollouts(self, ctx, parent, action, fixed, node, k, idx, out):
        hid = ctx["hand"].hand_id
        for i in idx:
            c = COMBOS[i]
            tot = 0.0
            for r in range(k):
                rng = CounterRng(stream_seed(self.seed, hid, node, int(i), r))
                tot += self._rollout(ctx, parent, action, fixed, c, rng)
            out[i] = tot / k
        return out

    def _rollout(self, ctx, parent: HandState, action: Action, fixed, c, rng: CounterRng) -> float:
        """One blueprint self-play rollout: first the unknown board cards (uniform, from the sorted
        remaining cards), then one uniform per decision (the agent's sampling rule)."""
        x, y, d = ctx["x"], ctx["y"], ctx["d"]
        known = set(c) | set(d) | set(parent.board) | set(fixed)
        rest = [k for k in range(52) if k not in known]
        runout = list(fixed)
        for _ in range(5 - len(parent.board) - len(fixed)):
            runout.append(rest.pop(int(rng.uniform() * len(rest))))
        st, _ = _apply_probe(parent, action, x, y, c, d, runout)
        self.stats["rollouts"] += 1
        memo: Dict[Tuple[int, int], int] = {}
        while not st.is_terminal:
            seat = st.current_player
            nb = len(st.board)
            b = memo.get((seat, nb))
            if b is None:
                b = int(self.bucketer.bucket(list(st.players[seat].hole), list(st.board)))
                memo[(seat, nb)] = b
            legal, acts, probs = self._policy_row(st, seat, b)
            u = rng.uniform()
            if probs is None:
                j = legal.index("c")
            else:
                j = len(probs) - 1
                acc = 0.0
                for t, p in enumerate(probs):
                    acc += p
                    if u < acc:
                        j = t
                        break
            st.apply(acts[j])
            self.stats["rollout_steps"] += 1
        return float(st.players[x].stack - st.starting_stacks[x])


def root_rollout_value(values: "SelfPlayValues", spec, pos: int, ci: int, di: int, cls: int, rollouts: int) -> float:
    """One root-table entry (csrc/aivat.h Evaluator::root_value): the mean of ``rollouts`` self-play
    rollouts from the start of a hand, x in seat ``pos`` (0: small blind, the button) holding combo
    ci, y holding di, streams (values.seed, pos, cls, k)."""
    x = 0 if pos == 0 else 1
    holes = [None, None]
    holes[x], holes[1 - x] = COMBOS[ci], COMBOS[di]
    hand = AivatHand(hand_id=0, stacks=tuple(spec.stacks), button=0, sb=spec.sb, bb=spec.bb, known_seat=x,
                     holes=(tuple(holes[0]), tuple(holes[1])), board=(), actions=())
    ctx = values.start_hand(hand)
    st = hand.new_state()
    tot = 0.0
    for r in range(rollouts):
        rng = CounterRng(stream_seed(values.seed, pos, cls, r))
        tot += values._rollout(ctx, st, None, [], COMBOS[ci], rng)
    return tot / rollouts


def _probs_or_call(legal: Sequence[str], probs: Optional[Sequence[float]]) -> List[float]:
    if probs is None:
        return [1.0 if n == "c" else 0.0 for n in legal]
    return list(probs)


def _disjoint2(hole, cards) -> bool:
    return not (set(hole) & set(cards))


def _any_hole(d, board, fixed) -> Tuple[int, int]:
    used = set(d) | set(board) | set(fixed)
    free = [k for k in range(52) if k not in used]
    return free[0], free[1]


# ---------------------------------------------------------------------------------- root table
class RootTable:
    """u_root(seat, c, d): one value per seat and suit-isomorphism class of the ordered pair of holes
    (x's c, y's d).  ``orbit`` (1326 x 1326, -1 on overlapping pairs) maps a pair to its class;
    ``values`` (2 x n_classes).  ``mean(seat)`` is the exact mean over all disjoint pairs."""

    MAGIC = b"NPAIVRT1"

    def __init__(self, orbit: np.ndarray, values: np.ndarray, meta: Optional[dict] = None):
        self.orbit = orbit
        self.values = values
        self.meta = dict(meta or {})
        ok = orbit >= 0
        counts = np.bincount(orbit[ok].ravel(), minlength=values.shape[1]).astype(float)
        self.counts = counts
        n = counts.sum()
        self._mean = [float(np.dot(counts, values[s]) / n) for s in range(values.shape[0])]

    def row(self, seat: int, d: Sequence[int]) -> np.ndarray:
        di = combo_index(d)
        o = self.orbit[:, di]
        out = np.zeros(N_COMBOS)
        ok = o >= 0
        out[ok] = self.values[seat][o[ok]]
        return out

    def mean(self, seat: int) -> float:
        return self._mean[seat]

    def save(self, path: str) -> None:
        import json

        meta = json.dumps(self.meta).encode()
        with open(path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<II", self.values.shape[0], self.values.shape[1]))
            f.write(struct.pack("<I", len(meta)))
            f.write(meta)
            f.write(np.ascontiguousarray(self.values, dtype="<f8").tobytes())

    @classmethod
    def load(cls, path: str, orbit: Optional[np.ndarray] = None) -> "RootTable":
        import json

        with open(path, "rb") as f:
            if f.read(8) != cls.MAGIC:
                raise ValueError(f"{path}: not a root table")
            ns, nc = struct.unpack("<II", f.read(8))
            (ml,) = struct.unpack("<I", f.read(4))
            meta = json.loads(f.read(ml).decode())
            values = np.frombuffer(f.read(8 * ns * nc), dtype="<f8").reshape(ns, nc).copy()
        if orbit is None:
            orbit = pair_orbits()
        return cls(orbit, values, meta)


def pair_orbits() -> np.ndarray:
    """Class id of every ordered pair (c, d) of disjoint holes under the 24 suit permutations
    (-1 for overlapping pairs); ids in order of first appearance scanning c then d.  The C++ core
    computes the same array (aivat_pair_orbits)."""
    from ..fast import core

    c = core()
    if c is not None and hasattr(c, "aivat_pair_orbits"):
        _, raw = c.aivat_pair_orbits()
        return np.frombuffer(raw, dtype=np.int32).reshape(N_COMBOS, N_COMBOS).copy()
    perms = list(itertools.permutations(range(4)))
    orbit = np.full((N_COMBOS, N_COMBOS), -1, dtype=np.int32)
    ids: Dict[Tuple[int, int, int, int], int] = {}
    for ci, (a, b) in enumerate(COMBOS):
        for di, (e, f) in enumerate(COMBOS):
            if len({a, b, e, f}) < 4:
                continue
            best = None
            for p in perms:
                m = [(k >> 2) * 4 + p[k & 3] for k in (a, b, e, f)]
                key = (min(m[0], m[1]), max(m[0], m[1]), min(m[2], m[3]), max(m[2], m[3]))
                if best is None or key < best:
                    best = key
            o = ids.get(best)
            if o is None:
                o = len(ids)
                ids[best] = o
            orbit[ci, di] = o
    return orbit
