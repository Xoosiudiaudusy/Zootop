"""``RNRTrainer`` on the C++ core (Restricted Nash Response, negpluribus/exploit/rnr.py).

    RNRTrainer(spec, bucketer, model, p, seed=0, warm_start=bp, backend="cpp", threads=8)

The opponent model has to be *tabular* for the compiled traversal: an ``OpponentModel`` (its
``policy`` is evaluated once per blueprint key in Python and handed over as a table) or a plain
``BlueprintStrategy``.  Anything else (an arbitrary object with ``policy(key, legal)``) keeps the
Python traversal; ``supports(model, warm_start)`` tells which.  ``threads=1`` reproduces the
Python trainer bit for bit for the same seed.

``hero_nodes`` / ``opp_nodes`` / ``nodes`` are live views of the C++ tables (see
``fast.trainer.NodeView``), not snapshots.  ``cache_caps`` bounds the bucket caches
(``NEGPLURIBUS_BUCKET_CACHE`` / ``fast.DEFAULT_CACHE_CAPS`` when not given); checkpoints carry
every thread's RNG state.  ``save_checkpoint`` writes JSON for ``*.json`` (the bytes json.dump
wrote before) and the binary checkpoint otherwise; ``load_checkpoint`` reads both, streamed in
C++ (docs/backends.md).
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Callable, Dict, Optional, Sequence

from ..abstraction import EquityBucketer
from ..cfr.game import GameSpec
from ..cfr.strategy import BlueprintStrategy
from ..exploit.rnr import RNRTrainer
from . import core, default_threads, resolve_cache_caps, verify_keys_enabled
from .blueprint import is_json_path
from .trainer import NodeView, core_bucketer, spec_to_dict


def _tabular_base(model) -> Optional[BlueprintStrategy]:
    if model is None:
        return None
    if type(model) is BlueprintStrategy:
        return model
    base = getattr(model, "base", None)
    if type(base) is BlueprintStrategy and hasattr(model, "policy"):
        return base
    return None


def supports(model, warm_start) -> bool:
    """Can this (model, warm_start) pair be materialised for the C++ traversal?"""
    if model is not None and _tabular_base(model) is None:
        return False
    if warm_start is not None and type(warm_start) is not BlueprintStrategy:
        return False
    return True


def tabulate_model(model) -> Dict[str, tuple]:
    """{key: (names, model.policy(key, names))} over the model's blueprint keys."""
    if model is None:
        return {}
    base = _tabular_base(model)
    if base is None:
        raise TypeError("the C++ RNR backend needs an OpponentModel or a BlueprintStrategy as opponent model")
    out = {}
    for key, (names, _) in base.table.items():
        probs = model.policy(key, list(names))
        if probs is not None:
            out[key] = (list(names), list(probs))
    return out


class CppRNRTrainer(RNRTrainer):
    """Constructed through ``RNRTrainer(..., backend="cpp")``."""

    def __init__(
        self,
        spec: GameSpec,
        bucketer: Optional[EquityBucketer],
        opponent_model,
        p_model: float,
        seed: int = 0,
        linear: bool = True,
        warm_start: Optional[BlueprintStrategy] = None,
        warm_visits: float = 30.0,
        regret_scale_bb: float = 1.0,
        backend: Optional[str] = None,
        threads: Optional[int] = None,
        cache_caps: Optional[Sequence[int]] = None,
    ):
        RNRTrainer.__init__(self, spec, bucketer, opponent_model, p_model, seed=seed, linear=linear,
                            warm_start=warm_start, warm_visits=warm_visits, regret_scale_bb=regret_scale_bb, backend="python")
        self.backend = "cpp"
        self.threads = int(threads) if threads else default_threads()
        self.seed = seed
        self.cache_caps = resolve_cache_caps(cache_caps)
        self._core_bucketer = core_bucketer(self.bucketer, self.cache_caps)
        warm_table = {k: (list(n), list(p)) for k, (n, p) in warm_start.table.items()} if warm_start is not None else {}
        self._core = core().RNRTrainer(
            spec_to_dict(spec), self._core_bucketer, int(seed) & 0xFFFFFFFFFFFFFFFF, bool(linear), self.threads,
            float(p_model), tabulate_model(opponent_model), warm_table, float(warm_visits), float(regret_scale_bb),
            verify_keys=verify_keys_enabled(),
        )
        self._hero_view = NodeView(lambda: self._core.n_hero, self._core.get_hero, self._core.hero_keys, self._core.export_hero)
        self._opp_view = NodeView(lambda: self._core.n_opp, self._core.get_opp, self._core.opp_keys, self._core.export_opp)

    # ------------------------------------------------------------ state mirrors
    def _ready(self) -> bool:
        return "_core" in self.__dict__

    @property
    def iteration(self) -> int:
        return self._core.iteration if self._ready() else self.__dict__.get("_iteration", 0)

    @iteration.setter
    def iteration(self, value: int) -> None:
        if self._ready():
            self._core.iteration = int(value)
        else:
            self.__dict__["_iteration"] = value

    @property
    def nodes_touched(self) -> int:
        return self._core.nodes_touched if self._ready() else self.__dict__.get("_nodes_touched", 0)

    @nodes_touched.setter
    def nodes_touched(self, value: int) -> None:
        if self._ready():
            self._core.nodes_touched = int(value)
        else:
            self.__dict__["_nodes_touched"] = value

    @property
    def linear(self) -> bool:
        """Linear CFR weighting, the core's own flag (a checkpoint sets it, as in the Python trainer)."""
        return self._core.linear if self._ready() else self.__dict__.get("_linear", True)

    @linear.setter
    def linear(self, value: bool) -> None:
        if self._ready():
            self._core.linear = bool(value)
        else:
            self.__dict__["_linear"] = value

    @property
    def planned_iters(self) -> int:
        return self._core.planned_iters if self._ready() else self.__dict__.get("_planned", 0)

    @planned_iters.setter
    def planned_iters(self, value: int) -> None:
        if self._ready():
            self._core.planned_iters = int(value)
        else:
            self.__dict__["_planned"] = value

    @property
    def hero_nodes(self) -> Mapping:  # type: ignore[override]
        if not self._ready():
            return self.__dict__.setdefault("_hero_py", {})
        return self._hero_view

    @hero_nodes.setter
    def hero_nodes(self, value: Mapping) -> None:
        if not self._ready():
            self.__dict__["_hero_py"] = value
            return
        self._core.import_hero({k: (n.actions, n.regret, n.strategy_sum, n.visits) for k, n in value.items()}, True)

    @property
    def opp_nodes(self) -> Mapping:  # type: ignore[override]
        if not self._ready():
            return self.__dict__.setdefault("_opp_py", {})
        return self._opp_view

    @opp_nodes.setter
    def opp_nodes(self, value: Mapping) -> None:
        if not self._ready():
            self.__dict__["_opp_py"] = value
            return
        self._core.import_opp({k: (n.actions, n.regret, n.strategy_sum, n.visits) for k, n in value.items()}, True)

    @property
    def nodes(self) -> Mapping:  # type: ignore[override]
        return self.hero_nodes

    @nodes.setter
    def nodes(self, value: Mapping) -> None:
        self.hero_nodes = value

    def cache_stats(self) -> Dict[str, Dict[str, int]]:
        return self._core.cache_stats()

    # ------------------------------------------------------------------ train
    def iterate(self) -> None:
        self._core.train(1)

    def train(self, iterations: int, log_every: int = 0, callback: Optional[Callable] = None) -> "CppRNRTrainer":
        self.planned_iters = self.iteration + iterations
        t0 = time.perf_counter()
        done = 0
        chunk = log_every if log_every else iterations
        while done < iterations:
            step = min(chunk, iterations - done)
            self._core.train(step)
            done += step
            if log_every:
                dt = time.perf_counter() - t0
                print(
                    f"  iter {self.iteration:>7,}  hero infosets {self._core.n_hero:>8,}  opp {self._core.n_opp:>8,}  "
                    f"nodes/s {self.nodes_touched / max(dt, 1e-9):>8,.0f}  elapsed {dt:6.0f}s  [cpp x{self.threads}]",
                    flush=True,
                )
                if callback:
                    callback(self)
        return self

    # ---------------------------------------------------------------- outputs
    def strategy(self) -> BlueprintStrategy:
        return BlueprintStrategy({k: (list(a), list(p)) for k, (a, p) in self._core.strategy().items()})

    def rational_opponent(self) -> BlueprintStrategy:
        return BlueprintStrategy({k: (list(a), list(p)) for k, (a, p) in self._core.rational_opponent().items()})

    def save_checkpoint(self, path: str) -> None:
        if is_json_path(path):
            self._core.save_checkpoint_json(str(path))
        else:
            self._core.save_checkpoint_bin(str(path))

    def load_checkpoint(self, path: str) -> "CppRNRTrainer":
        kind = core().file_kind(str(path))
        if kind == "checkpoint":
            self._core.load_checkpoint_bin(str(path))
        elif kind == "json":
            self._core.load_checkpoint_json(str(path))
        else:
            raise ValueError(f"{path}: not a checkpoint ({kind})")
        return self
