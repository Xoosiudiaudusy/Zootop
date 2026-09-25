"""External-sampling Monte Carlo CFR on the abstracted game.

Same algorithm as ``kuhn.py``, with two changes that make it scale:

1. **Sampling instead of full traversal.**  Each iteration deals *one* random
   deck (chance is sampled) and, for one *traverser* at a time, explores all of
   the traverser's actions while opponents' actions are sampled from their
   current strategy.  Regrets are unbiased in expectation; the cost per
   iteration is a few hundred nodes instead of the whole tree.
2. **Abstraction.**  Regrets are keyed by ``infoset_key`` (bucket + abstract
   history), not by exact cards and chip amounts.

Optional **Linear CFR** (Pluribus): iteration ``t`` weights its regret and
strategy contributions by ``t``, so early, noisy iterations fade out faster.

Multiplayer caveat: with 3+ players CFR has no convergence guarantee to a Nash
equilibrium.  Pluribus showed it still produces very strong strategies in
practice; we simply measure the result in bb/100 against opponents.
"""
from __future__ import annotations

import json
import random
import time
from typing import Callable, Dict, List, Optional

from ..abstraction import EquityBucketer, infoset_key
from ..engine import HandState
from .game import GameSpec
from .strategy import BlueprintStrategy


class Node:
    __slots__ = ("actions", "regret", "strategy_sum", "visits")

    def __init__(self, actions: List[str]):
        self.actions = actions
        self.regret = [0.0] * len(actions)
        self.strategy_sum = [0.0] * len(actions)
        self.visits = 0

    def current_strategy(self) -> List[float]:
        pos = [r if r > 0 else 0.0 for r in self.regret]
        s = sum(pos)
        if s <= 0:
            n = len(self.actions)
            return [1.0 / n] * n
        return [p / s for p in pos]

    def average_strategy(self) -> List[float]:
        s = sum(self.strategy_sum)
        if s <= 0:
            n = len(self.actions)
            return [1.0 / n] * n
        return [x / s for x in self.strategy_sum]


class MCCFRTrainer:
    # ---- backend hook: ``backend="cpp"`` (or NEGPLURIBUS_BACKEND=cpp) builds the C++-backed
    # subclass from negpluribus/fast/trainer.py; same interface, same outputs.  Default: python.
    def __new__(cls, *args, backend: Optional[str] = None, **kwargs):
        if cls is MCCFRTrainer:
            from ..fast import resolve_backend

            if resolve_backend(backend) == "cpp":
                from ..fast.trainer import CppMCCFRTrainer

                return object.__new__(CppMCCFRTrainer)
        return object.__new__(cls)

    def __init__(
        self,
        spec: GameSpec,
        bucketer: Optional[EquityBucketer] = None,
        seed: int = 0,
        linear: bool = True,
        backend: Optional[str] = None,
        threads: Optional[int] = None,
        cache_caps=None,  # C++ backend: (flop, turn, river) bucket-cache capacities (docs/backends.md)
    ):
        self.backend = "python"
        self.spec = spec
        self.grid = spec.grid
        self.bucketer = bucketer if bucketer is not None else spec.make_bucketer()
        if spec.needs_buckets() and not self.bucketer.boundaries:
            raise ValueError("this spec bets postflop: pass a fitted bucketer (bucketer.fit())")
        self.rng = random.Random(seed)
        self.linear = linear
        self.nodes: Dict[str, Node] = {}
        self.iteration = 0
        self.nodes_touched = 0

    # ------------------------------------------------------------- traversal
    def _node(self, key: str, actions: List[str]) -> Node:
        node = self.nodes.get(key)
        if node is None:
            node = Node(list(actions))
            self.nodes[key] = node
        return node

    def _traverse(self, state: HandState, traverser: int, weight: float) -> float:
        """Expected utility (in bb) for ``traverser`` from this state; updates regrets on the way."""
        if state.is_terminal:
            return state.record().net[traverser] / self.spec.bb
        seat = state.current_player
        assert seat is not None
        obs = state.observe(seat)
        actions = self.grid.abstract_actions(obs)
        key = infoset_key(obs, self.bucketer, self.grid)
        node = self._node(key, actions)
        self.nodes_touched += 1
        if node.actions != actions:  # should not happen in self-play; keep going safely
            actions = node.actions
        sigma = node.current_strategy()

        if seat == traverser:
            utils = []
            for a in actions:
                child = state.clone()
                child.apply(self.grid.to_concrete(obs, a))
                utils.append(self._traverse(child, traverser, weight))
            u = sum(p * v for p, v in zip(sigma, utils))
            for i, v in enumerate(utils):
                node.regret[i] += weight * (v - u)
            return u

        # opponent node: sample one action, accumulate this player's average strategy
        for i, p in enumerate(sigma):
            node.strategy_sum[i] += weight * p
        node.visits += 1
        a = self._sample(actions, sigma)
        state.apply(self.grid.to_concrete(obs, a))
        return self._traverse(state, traverser, weight)

    def _sample(self, actions: List[str], probs: List[float]) -> str:
        r = self.rng.random()
        acc = 0.0
        for a, p in zip(actions, probs):
            acc += p
            if r < acc:
                return a
        return actions[-1]

    # ---------------------------------------------------------------- train
    def iterate(self) -> None:
        self.iteration += 1
        t = self.iteration
        weight = float(t) if self.linear else 1.0
        order = list(range(52))
        self.rng.shuffle(order)
        button = t % self.spec.n_players
        for traverser in range(self.spec.n_players):
            state = self.spec.new_hand(order, button)
            self._traverse(state, traverser, weight)

    def train(
        self,
        iterations: int,
        log_every: int = 0,
        callback: Optional[Callable[["MCCFRTrainer"], None]] = None,
    ) -> "MCCFRTrainer":
        t0 = time.perf_counter()
        for k in range(iterations):
            self.iterate()
            if log_every and (k + 1) % log_every == 0:
                dt = time.perf_counter() - t0
                print(
                    f"  iter {self.iteration:>7,}  infosets {len(self.nodes):>8,}  "
                    f"nodes/s {self.nodes_touched / max(dt, 1e-9):>8,.0f}  elapsed {dt:6.0f}s",
                    flush=True,
                )
                if callback:
                    callback(self)
        return self

    def train_parallel(
        self,
        iterations: int,
        workers: Optional[int] = None,
        sync_every: int = 1000,
        log_every: int = 0,
    ) -> "MCCFRTrainer":
        """Same result type as ``train`` but on ``workers`` processes (see cfr/parallel.py):
        each round every worker runs ``sync_every`` iterations from the merged tables, then
        regrets / strategy sums / visits are summed back into this trainer."""
        from .parallel import train_parallel

        return train_parallel(self, iterations, workers=workers, sync_every=sync_every, log_every=log_every)

    # ------------------------------------------------------------- outputs
    def strategy(self) -> BlueprintStrategy:
        return BlueprintStrategy({k: (list(n.actions), n.average_strategy()) for k, n in self.nodes.items()})

    def save_checkpoint(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "iteration": self.iteration,
                    "linear": self.linear,
                    "nodes": {k: [n.actions, n.regret, n.strategy_sum, n.visits] for k, n in self.nodes.items()},
                },
                f,
            )

    def load_checkpoint(self, path: str) -> "MCCFRTrainer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.iteration = d["iteration"]
        self.linear = d["linear"]
        self.nodes = {}
        for k, (actions, regret, ssum, visits) in d["nodes"].items():
            n = Node(actions)
            n.regret, n.strategy_sum, n.visits = regret, ssum, visits
            self.nodes[k] = n
        return self

    def summary(self, keys_prefix: str = "", limit: int = 20) -> str:
        rows = []
        for k, n in sorted(self.nodes.items()):
            if k.startswith(keys_prefix):
                probs = " ".join(f"{a}:{p:.2f}" for a, p in zip(n.actions, n.average_strategy()))
                rows.append(f"  {k:<40} {probs}")
                if len(rows) >= limit:
                    break
        return "\n".join(rows)
