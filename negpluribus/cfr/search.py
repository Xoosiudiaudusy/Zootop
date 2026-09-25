"""Depth-limited real-time search (the Pluribus trick).

The blueprint is a coarse, opponent-agnostic strategy.  When it is our turn
postflop we re-solve the *current* spot with more care:

1. **Ranges.**  For every player still in the hand we ask: which hole cards
   would the blueprint have played this way?  Each candidate hand gets weight
   = product of blueprint probabilities of the observed actions.  We keep a
   handful of weighted particles per player (``RangeSampler``).  This is
   "unsafe" subgame solving: it trusts that everyone followed the blueprint
   before the root.  (Hook #2 for opponent stats: replace blueprint reach with
   a stat-informed range.)

2. **A small game tree** rooted here.  Every iteration deals hands from the
   ranges (ours too: the opponent in the subgame must not "see" our hand) and
   runs external-sampling MCCFR for ``depth`` actions ahead.

3. **Leaves.**  At the depth limit the hand is not over, but instead of the
   whole remaining tree each live player picks ONE of four continuation
   strategies for the rest of the hand:

       "bp"    play the blueprint as is
       "fold"  blueprint with P(fold)  x bias_factor, renormalised
       "call"  blueprint with P(call)  x bias_factor
       "raise" blueprint with P(raise) x bias_factor   (all raise sizes)

   The pick is a regret-minimising decision like any other, so opponents end
   up choosing whichever continuation hurts us most: the solution is robust to
   the blueprint being off in the unexplored part of the tree.  The leaf value
   is one Monte-Carlo rollout with the chosen continuations.
   (Hook #1 for opponent stats: ``ContinuationPolicy.bias_factors(seat)`` -
   a station gets a stronger "call" continuation, a nit a stronger "fold" one.)

4. We read the average strategy at our actual information set and act.  Then
   the tree is thrown away; next decision, new search.

Only the reduced game is supported for now (postflop = flop), but nothing in
here depends on the number of streets.
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..abstraction import ALL_IN, CALL_NAME, FOLD_NAME, BetGrid, EquityBucketer, history_string, infoset_key, starting_stacks
from ..cards import ALL_HOLE_CLASSES, Deck, hole_class
from ..engine import BOARD_CARDS_BY_STREET, ActionType, HandState, Observation, Street, position_name
from .game import GameSpec
from .mccfr import Node
from .strategy import BlueprintStrategy

CONTINUATIONS = ("bp", "fold", "call", "raise")


@dataclass(frozen=True)
class SearchConfig:
    iterations: int = 800
    depth: Optional[int] = 2  # decisions ahead of the root before leaves; None = to the end of the hand
    n_particles: int = 200  # weighted candidate hands kept per player
    bias_factor: float = 5.0  # Pluribus: biased actions get x5 probability
    linear: bool = True
    min_reach: float = 1e-3  # floor for reach weights (unknown keys / zero-prob actions)
    focus: float = 0.7  # when WE traverse, P(our sampled hand is drawn from our actual bucket); see _deal
    rm_plus: bool = False  # regret matching+; with sampled values it inflates noisy actions, so off by default
    prior_iters: int = 300  # warm start: the blueprint counts as this many iterations of evidence (0 = off)
    common_random: bool = True  # same random draws for every branch of a traverser node (variance reduction)
    prior_regret_bb: float = 1.0  # typical regret size (bb) used to convert the prior into initial regrets


# ----------------------------------------------------------------- continuations
class ContinuationPolicy:
    """Blueprint plus three biased copies.

    ``bias_factors(seat)`` is the entry point for opponent modelling: it returns
    how strongly each continuation is biased *for that seat*.  The default is
    Pluribus' symmetric x5.  A stats-aware subclass can return, say,
    ``{"fold": 2, "call": 12, "raise": 3}`` for a calling station so that the
    search prepares for the continuation that player actually tends to take.
    """

    def __init__(self, blueprint: BlueprintStrategy, bias_factor: float = 5.0, models: Optional[Dict[int, object]] = None):
        self.blueprint = blueprint
        self.bias_factor = bias_factor
        # step 5b: seat -> opponent model (anything with .policy(key, legal)); the four continuation
        # strategies of a modelled seat are built on top of ITS model, not on the blueprint, so the
        # search prepares for what this opponent actually tends to do beyond the depth limit
        self.models: Dict[int, object] = dict(models or {})

    def base_strategy(self, seat: int):
        return self.models.get(seat, self.blueprint)

    def bias_factors(self, seat: int) -> Dict[str, float]:
        f = self.bias_factor
        return {"fold": f, "call": f, "raise": f}

    @staticmethod
    def _matches(kind: str, action: str) -> bool:
        if kind == "fold":
            return action == FOLD_NAME
        if kind == "call":
            return action == CALL_NAME
        if kind == "raise":
            return action == ALL_IN or action.startswith("r")
        return False

    def probs(self, kind: str, seat: int, key: str, legal: Sequence[str]) -> List[float]:
        base = self.base_strategy(seat).policy(key, legal)
        if base is None:
            base = [1.0 / len(legal)] * len(legal)
        if kind == "bp":
            return base
        factor = self.bias_factors(seat).get(kind, 1.0)
        boosted = [p * factor if self._matches(kind, a) else p for a, p in zip(legal, base)]
        s = sum(boosted)
        return [b / s for b in boosted] if s > 0 else base


# ------------------------------------------------------------------- ranges
class RangeSampler:
    """Weighted candidate hands per seat, from blueprint reach probabilities."""

    def __init__(self, blueprint: BlueprintStrategy, bucketer: EquityBucketer, grid: BetGrid, cfg: SearchConfig,
                 models: Optional[Dict[int, object]] = None):
        self.blueprint = blueprint
        self.bucketer = bucketer
        self.grid = grid
        self.cfg = cfg
        # step 5b: seat -> opponent model; that seat's range is its reach under the MODEL (what a
        # station / maniac would have played this way), not under the blueprint
        self.models: Dict[int, object] = dict(models or {})

    def _action_prob(self, seat: int, key: str, action_name: str) -> float:
        entry = self.blueprint.table.get(key)
        if entry is None:
            return self.cfg.min_reach
        names, probs = entry
        model = self.models.get(seat)
        if model is not None:
            mp = model.policy(key, names)
            if mp is not None:
                probs = mp
        p = dict(zip(names, probs)).get(action_name, 0.0)
        return max(p, self.cfg.min_reach)

    def _event_keys(self, obs: Observation, seat: int, event_rng) -> List[Tuple[Street, str, int, str, str]]:
        """For each event by ``seat``: (street, position, n_active, history-before, action-name)."""
        events = obs.events
        pos = position_name(seat, obs.button, obs.n_players)
        out = []
        folded = set()
        for i, ev in enumerate(events):
            if ev.seat == seat:
                hist = history_string(events[:i], self.grid, event_rng)
                name = self.grid.from_concrete(ev, ev.all_in, event_rng(i) if event_rng else None)
                out.append((ev.street, pos, obs.n_players - len(folded), hist, name))
            if ev.action.type == ActionType.FOLD:
                folded.add(ev.seat)
        return out

    def particles(
        self,
        obs: Observation,
        seat: int,
        rng: random.Random,
        event_rng=None,
        exclude: Sequence[int] = (),
    ) -> List[Tuple[Tuple[int, int], float]]:
        """``n_particles`` (hole, weight) pairs for ``seat``, consistent with its actions so far."""
        steps = self._event_keys(obs, seat, event_rng)
        blocked = set(obs.board) | set(exclude)
        cards = [c for c in range(52) if c not in blocked]
        combos = [(a, b) for i, a in enumerate(cards) for b in cards[i + 1:]]

        # preflop reach depends only on the 169-class -> compute per class
        pre_steps = [s for s in steps if s[0] == Street.PREFLOP]
        post_steps = [s for s in steps if s[0] != Street.PREFLOP]
        class_w: Dict[str, float] = {}
        for cls in ALL_HOLE_CLASSES:
            w = 1.0
            b = ALL_HOLE_CLASSES.index(cls)
            for street, pos, n_active, hist, name in pre_steps:
                w *= self._action_prob(seat, f"P|{pos}|{n_active}|b{b}|{hist}", name)
            class_w[cls] = w
        weights = [class_w[hole_class(a, b)] for a, b in combos]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(combos)
            total = float(len(combos))
        chosen = rng.choices(combos, weights=weights, k=self.cfg.n_particles)
        if not post_steps:
            return [(h, 1.0) for h in chosen]
        out = []
        for hole in chosen:
            w = 1.0
            for street, pos, n_active, hist, name in post_steps:
                board = obs.board[: BOARD_CARDS_BY_STREET[street]]
                b = self.bucketer.bucket(list(hole), board)
                w *= self._action_prob(seat, f"{street.name[0]}|{pos}|{n_active}|b{b}|{hist}", name)
            out.append((hole, w))
        return out


# ------------------------------------------------------------------- solver
class SubgameSolver:
    def __init__(
        self,
        spec: GameSpec,
        blueprint: BlueprintStrategy,
        bucketer: EquityBucketer,
        cfg: SearchConfig = SearchConfig(),
        continuations: Optional[ContinuationPolicy] = None,
        seed: int = 0,
        models: Optional[Dict[int, object]] = None,
    ):
        self.spec = spec
        self.blueprint = blueprint
        self.bucketer = bucketer
        self.grid = spec.grid
        self.cfg = cfg
        self.models: Dict[int, object] = dict(models or {})
        self.cont = continuations or ContinuationPolicy(blueprint, cfg.bias_factor, models=self.models)
        self.ranges = RangeSampler(blueprint, bucketer, self.grid, cfg, models=self.models)
        self.rng = random.Random(seed)
        self.nodes: Dict[str, Node] = {}
        self.last_root_key: Optional[str] = None
        self.last_nodes_touched = 0

    # ------------------------------------------------------ root rebuild
    def _rebuild(self, obs: Observation, holes: Dict[int, Tuple[int, int]], stacks_start: Sequence[int]) -> HandState:
        """A HandState identical to what ``obs`` describes, with hypothetical hole cards."""
        order: List[int] = []
        for s in range(obs.n_players):
            order.extend(holes[s])
        order.extend(obs.board)
        used = set(order)
        rest = [c for c in range(52) if c not in used]
        self.rng.shuffle(rest)
        order.extend(rest)
        state = HandState(list(stacks_start), obs.button, sb=self.spec.sb, bb=self.spec.bb, ante=self.spec.ante,
                          deck=Deck.from_order(order), max_street=self.spec.max_street)
        for ev in obs.events:
            state.apply(ev.action)
        assert state.current_player == obs.seat and state.street == obs.street
        return state

    def _deal(
        self,
        obs: Observation,
        particles: Dict[int, List[Tuple[Tuple[int, int], float]]],
        traverser: int,
        focused: List[Tuple[Tuple[int, int], float]],
    ) -> Dict[int, Tuple[int, int]]:
        """Sample one hand per seat from the particle ranges (no card conflicts).

        Targeted sampling: when the hero is the traverser, with probability
        ``cfg.focus`` the hero's hand is drawn only from particles that share the
        hero's *actual* bucket, so the regrets at the information set we will
        actually act on get updated several times more often.  This changes how
        often hero infosets are updated, not what is learned: opponents' regrets
        are updated only in their own traversals, where the hero's hand comes
        from the full range, and the hero's average strategy is accumulated there
        too.  (Same idea as Targeted CFR, Jackson 2017, without the exact weights.)
        """
        holes: Dict[int, Tuple[int, int]] = {}
        used = set(obs.board)
        for seat in sorted(particles):
            pts = particles[seat]
            if seat == obs.seat and seat == traverser and focused and self.rng.random() < self.cfg.focus:
                pts = focused
            ws = [w for _, w in pts]
            hole = pts[0][0]
            for _ in range(30):
                hole = self.rng.choices(pts, weights=ws)[0][0] if sum(ws) > 0 else self.rng.choice(pts)[0]
                if hole[0] not in used and hole[1] not in used:
                    break
            holes[seat] = hole
            used.update(hole)
        return holes

    # ------------------------------------------------------------ solve
    def solve(self, obs: Observation, event_rng: Optional[Callable[[int], random.Random]] = None) -> Optional[List[float]]:
        """Return search probabilities aligned with ``grid.abstract_actions(obs)`` or None."""
        self.nodes = {}
        self.last_nodes_touched = 0
        stacks_start = starting_stacks(obs)
        live = [s for s in range(obs.n_players) if not obs.folded[s]]
        particles: Dict[int, List[Tuple[Tuple[int, int], float]]] = {}
        for s in range(obs.n_players):
            if s in live:
                particles[s] = self.ranges.particles(obs, s, self.rng, event_rng)
            else:  # folded: cards are irrelevant, any two will do
                particles[s] = [(tuple(self.rng.sample([c for c in range(52) if c not in obs.board], 2)), 1.0)]  # type: ignore[arg-type]
        legal = self.grid.abstract_actions(obs)
        root_key = infoset_key(obs, self.bucketer, self.grid, event_rng)
        self.last_root_key = root_key
        my_bucket = self.bucketer.bucket(obs.hole, obs.board)
        focused = [(h, w) for h, w in particles[obs.seat] if self.bucketer.bucket(list(h), obs.board) == my_bucket]
        if not focused:
            focused = [(tuple(obs.hole), 1.0)]  # type: ignore[list-item]

        for it in range(1, self.cfg.iterations + 1):
            weight = float(it) if self.cfg.linear else 1.0
            for traverser in live:
                holes = self._deal(obs, particles, traverser, focused)
                state = self._rebuild(obs, holes, stacks_start)
                self._traverse(state, traverser, 0, weight, event_rng, 1.0)

        node = self.nodes.get(root_key)
        if node is None or node.actions != legal:
            return None
        return node.average_strategy()

    # --------------------------------------------------------- traversal
    def _key(self, state: HandState, obs: Observation, event_rng) -> str:
        return infoset_key(obs, self.bucketer, self.grid, event_rng)

    def _update(self, node: Node, sigma: List[float], utils: List[float], weight: float, own_reach: float) -> float:
        """Regret update at a traverser node; also accumulate its average strategy.

        External sampling normally accumulates a player's average strategy only in the
        *other* players' traversals.  Here we additionally accumulate it in the player's
        own traversals, weighted by the player's own reach inside the subgame (that is
        exactly CFR's average-strategy weight), because with targeted sampling that is
        where our actual information set is visited most.  Regret matching+ (optional)
        floors regrets at zero.
        """
        u = sum(p * v for p, v in zip(sigma, utils))
        for i, v in enumerate(utils):
            node.regret[i] += weight * (v - u)
            if self.cfg.rm_plus and node.regret[i] < 0:
                node.regret[i] = 0.0
        for i, p in enumerate(sigma):
            node.strategy_sum[i] += weight * own_reach * p
        return u

    def _traverse(self, state: HandState, traverser: int, depth: int, weight: float, event_rng, own_reach: float) -> float:
        if state.is_terminal:
            return state.record().net[traverser] / state.bb
        if self.cfg.depth is not None and depth >= self.cfg.depth:
            return self._leaf(state, traverser, {}, weight, event_rng, own_reach)
        seat = state.current_player
        assert seat is not None
        obs = state.observe(seat)
        actions = self.grid.abstract_actions(obs)
        key = self._key(state, obs, event_rng)
        node = self._node(key, actions)
        self.last_nodes_touched += 1
        sigma = node.current_strategy()
        if seat == traverser:
            utils = []
            branch_seed = self.rng.getrandbits(48)
            for a, p in zip(node.actions, sigma):
                child = state.clone()
                child.apply(self.grid.to_concrete(obs, a))
                with self._branch_rng(branch_seed):
                    utils.append(self._traverse(child, traverser, depth + 1, weight, event_rng, own_reach * p))
            return self._update(node, sigma, utils, weight, own_reach)
        for i, p in enumerate(sigma):
            node.strategy_sum[i] += weight * p
        a = self._sample(node.actions, sigma)
        state.apply(self.grid.to_concrete(obs, a))
        return self._traverse(state, traverser, depth + 1, weight, event_rng, own_reach)

    def _leaf(self, state: HandState, traverser: int, choices: Dict[int, str], weight: float, event_rng, own_reach: float) -> float:
        """Each live player picks a continuation strategy (a decision node with 4 options)."""
        pending = [p.seat for p in state.players if p.can_act and p.seat not in choices]
        if not pending:
            return self._rollout(state, traverser, choices, event_rng)
        seat = pending[0]
        obs = state.observe(seat)
        hist = history_string(obs.events, self.grid, event_rng)
        key = f"LEAF|{obs.position}|b{self.bucketer.bucket(obs.hole, obs.board)}|{hist}"
        node = self._node(key, list(CONTINUATIONS))
        self.last_nodes_touched += 1
        sigma = node.current_strategy()
        if seat == traverser:
            utils = []
            branch_seed = self.rng.getrandbits(48)
            for kind, p in zip(CONTINUATIONS, sigma):
                with self._branch_rng(branch_seed):
                    utils.append(self._leaf(state, traverser, {**choices, seat: kind}, weight, event_rng, own_reach * p))
            return self._update(node, sigma, utils, weight, own_reach)
        for i, p in enumerate(sigma):
            node.strategy_sum[i] += weight * p
        kind = self._sample(list(CONTINUATIONS), sigma)
        return self._leaf(state, traverser, {**choices, seat: kind}, weight, event_rng, own_reach)

    def _rollout(self, state: HandState, traverser: int, choices: Dict[int, str], event_rng) -> float:
        st = state.clone()
        while not st.is_terminal:
            seat = st.current_player
            assert seat is not None
            obs = st.observe(seat)
            legal = self.grid.abstract_actions(obs)
            key = self._key(st, obs, event_rng)
            probs = self.cont.probs(choices.get(seat, "bp"), seat, key, legal)
            a = self._sample(legal, probs)
            st.apply(self.grid.to_concrete(obs, a))
        return st.record().net[traverser] / st.bb

    @contextmanager
    def _branch_rng(self, seed: int):
        """Common random numbers: every sibling branch under a traverser node sees the same
        random stream, so opponents' sampled responses and rollouts are correlated across our
        candidate actions and the *differences* between action values are far less noisy.
        The main RNG is restored afterwards."""
        if not self.cfg.common_random:
            yield
            return
        saved = self.rng
        self.rng = random.Random(seed)
        try:
            yield
        finally:
            self.rng = saved

    def _node(self, key: str, actions: List[str]) -> Node:
        """Get/create a subgame node, warm-started from the blueprint.

        Without a prior the subgame starts uniform and, with a few hundred noisy
        iterations, stays diffuse (the log of decisions showed near-uniform mixes
        even where the blueprint is pure).  The warm start treats the blueprint as
        ``prior_iters`` iterations of evidence: initial regrets reproduce the
        blueprint under regret matching and the average strategy starts there.  The
        search then *moves away* from the blueprint only where the subgame finds a
        real reason to.  Leaf (continuation) nodes have no blueprint and start uniform.
        """
        node = self.nodes.get(key)
        if node is None:
            node = Node(list(actions))
            base = None if key.startswith("LEAF|") else self.blueprint.policy(key, actions)
            if base is not None and self.cfg.prior_iters > 0:
                n = self.cfg.prior_iters
                w_prior = n * (n + 1) / 2.0 if self.cfg.linear else float(n)
                for i, p in enumerate(base):
                    node.regret[i] = w_prior * self.cfg.prior_regret_bb * p
                    node.strategy_sum[i] = w_prior * p
            self.nodes[key] = node
        return node

    def _sample(self, actions: Sequence[str], probs: Sequence[float]) -> str:
        r = self.rng.random()
        acc = 0.0
        for a, p in zip(actions, probs):
            acc += p
            if r < acc:
                return a
        return actions[-1]
