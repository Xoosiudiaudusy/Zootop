"""``MCCFRTrainer`` on the C++ core: same interface, same outputs, tens to hundreds of times
faster, all cores in one process.

    trainer = MCCFRTrainer(spec, bucketer, seed=0, backend="cpp", threads=8, cache_caps=(4_000_000, 32_000_000, 4_000_000))
    trainer.train(100_000)
    trainer.strategy()                # BlueprintStrategy with the same keys / action lists as the Python one
    trainer.blueprint()               # the same average strategy as a C++ lookup (no dict), for agents
    trainer.save_checkpoint("x.bin")  # binary checkpoint, written from C++ (docs/backends.md)
    trainer.save_checkpoint("x.json") # the JSON layout of the Python trainer (same bytes as json.dump)
    trainer.save_blueprint("b.bin")   # binary blueprint (probabilities rounded like the JSON); "b.json": JSON

With ``threads=1`` the run is bit-identical to the Python trainer for the same seed.  With
more threads it is statistically equivalent (see docs/backends.md for the race semantics).

Memory: the C++ table is the only copy of the regrets.  ``trainer.nodes`` is a *view* on it
(``len``, ``get``, ``[]``, iteration and ``items()`` go to the core each time), not a Python
snapshot.  Checkpoints and blueprints of either format stream between the table and the file in
C++ (no Python objects); only ``strategy()`` and ``nodes.items()`` build dicts.  The bucket caches
are bounded (``cache_caps``, default ``fast.DEFAULT_CACHE_CAPS`` or the ``NEGPLURIBUS_BUCKET_CACHE``
env var), so a run's memory is flat however long it goes.

Checkpoints written by this class carry every thread's RNG state, so a resumed run continues the
exact streams (single thread: bit-identical to an uninterrupted run); the Python trainer ignores
the JSON field and re-seeds, as it always did.  ``load_checkpoint`` reads either format (by the
file's first bytes); a binary checkpoint of another game (spec, grid or buckets) is refused.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Callable, Dict, Iterator, List, Optional, Sequence

from ..abstraction import EquityBucketer, PotentialAwareBucketer
from ..cfr.game import GameSpec
from ..cfr.mccfr import MCCFRTrainer, Node
from ..cfr.strategy import BlueprintStrategy
from . import core, default_threads, resolve_cache_caps, verify_keys_enabled
from .tables import tabulated
from .blueprint import CppBlueprint, is_json_path


def spec_to_dict(spec: GameSpec) -> dict:
    return {
        "n_players": spec.n_players,
        "stack_bb": spec.stack_bb,
        "sb": spec.sb,
        "bb": spec.bb,
        "ante": spec.ante,
        "max_street": int(spec.max_street),
        "preflop_fracs": [float(x) for x in spec.preflop_fracs],
        "postflop_fracs": [float(x) for x in spec.postflop_fracs],
        "max_raises_per_street": spec.max_raises_per_street,
        "n_buckets": spec.n_buckets,
        "forbid_open_limp": spec.forbid_open_limp,
        "allow_all_in": True,
    }


def core_bucketer(bucketer, cache_caps=None):
    """The C++ twin of a fitted Python bucketer (its cache starts empty): ``Bucketer`` for an
    ``EquityBucketer`` (same boundaries / sample count), ``PotentialBucketer`` for a
    ``PotentialAwareBucketer`` (same centroids, river cut points, samples, bins).
    ``cache_caps``: (flop, turn, river) cache capacities in entries (None: env / defaults).
    With ``NEGPLURIBUS_BUCKET_TABLES`` pointing at a directory that holds a precomputed table of
    this bucketer (fast/tables.py), the result answers from it: same buckets, no Monte-Carlo."""
    c = core()
    caps = list(resolve_cache_caps(cache_caps))
    boundaries = {int(k): list(v) for k, v in bucketer.boundaries.items()}
    if isinstance(bucketer, PotentialAwareBucketer):
        if not hasattr(c, "PotentialBucketer"):
            raise RuntimeError("the built C++ core predates potential-aware buckets; run `python scripts/build_fast.py`")
        centroids = {int(k): [list(cdf) for cdf in v] for k, v in bucketer.centroids.items()}
        return tabulated(c.PotentialBucketer(bucketer.n_buckets, bucketer.samples, bucketer.bins, centroids, boundaries, caps))
    return tabulated(c.Bucketer(bucketer.n_buckets, bucketer.samples, boundaries, caps))


def _to_node(row) -> Node:
    actions, regret, ssum, visits = row
    n = Node(list(actions))
    n.regret, n.strategy_sum, n.visits = list(regret), list(ssum), visits
    return n


class NodeView(Mapping):
    """Read-through view of a C++ node table as ``{key: Node}``: nothing is copied until asked,
    every access reflects the current regrets.  ``items()`` / ``values()`` export the table
    once (transient); mutating a returned ``Node`` does not write back (use the setter of the
    owning trainer's ``nodes`` for that)."""

    def __init__(self, n_keys: Callable[[], int], get_row: Callable[[str], object],
                 keys: Callable[[], List[str]], export: Callable[[], Dict[str, tuple]]):
        self._n_keys, self._get_row, self._keys, self._export = n_keys, get_row, keys, export

    def __len__(self) -> int:
        return self._n_keys()

    def __getitem__(self, key: str) -> Node:
        row = self._get_row(key)
        if row is None:
            raise KeyError(key)
        return _to_node(row)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._get_row(key) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def get(self, key: str, default=None):
        row = self._get_row(key)
        return default if row is None else _to_node(row)

    def items(self):
        return [(k, _to_node(row)) for k, row in self._export().items()]

    def values(self):
        return [_to_node(row) for row in self._export().values()]

    def keys(self):
        return self._keys()

    def snapshot(self) -> Dict[str, Node]:
        """A plain dict copy (what the Python trainer's ``nodes`` is)."""
        return {k: _to_node(row) for k, row in self._export().items()}


class CppMCCFRTrainer(MCCFRTrainer):
    """Constructed through ``MCCFRTrainer(..., backend="cpp")``; do not subclass RNRTrainer from it."""

    def __init__(
        self,
        spec: GameSpec,
        bucketer: Optional[EquityBucketer] = None,
        seed: int = 0,
        linear: bool = True,
        backend: Optional[str] = None,
        threads: Optional[int] = None,
        cache_caps: Optional[Sequence[int]] = None,
    ):
        super().__init__(spec, bucketer, seed=seed, linear=linear, backend="python")
        self.backend = "cpp"
        self.threads = int(threads) if threads else default_threads()
        self.seed = seed
        self.cache_caps = resolve_cache_caps(cache_caps)
        self._core_bucketer = core_bucketer(self.bucketer, self.cache_caps)
        self._core = core().Trainer(spec_to_dict(spec), self._core_bucketer, int(seed) & 0xFFFFFFFFFFFFFFFF, bool(linear), self.threads,
                                    verify_keys=verify_keys_enabled())
        self._view = NodeView(lambda: self._core.n_nodes, self._core.get_node, self._core.keys, self._core.export_nodes)

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
    def nodes(self) -> Mapping:  # type: ignore[override]
        """Live view of the C++ table (see ``NodeView``); ``nodes.snapshot()`` for a dict copy."""
        if not self._ready():
            return self.__dict__.setdefault("_nodes_py", {})
        return self._view

    @nodes.setter
    def nodes(self, value: Mapping) -> None:
        if not self._ready():
            self.__dict__["_nodes_py"] = value
            return
        self._core.import_nodes({k: (n.actions, n.regret, n.strategy_sum, n.visits) for k, n in value.items()}, True)

    @property
    def n_nodes(self) -> int:
        return self._core.n_nodes

    def cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Per street: bucket-cache capacity (slots), size, computes (= misses), evictions."""
        return self._core.cache_stats()

    # ------------------------------------------------------------------ train
    def iterate(self) -> None:
        self._core.train(1)

    def train(
        self,
        iterations: int,
        log_every: int = 0,
        callback: Optional[Callable[["MCCFRTrainer"], None]] = None,
    ) -> "CppMCCFRTrainer":
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
                    f"  iter {self.iteration:>7,}  infosets {self.n_nodes:>8,}  "
                    f"nodes/s {self.nodes_touched / max(dt, 1e-9):>8,.0f}  elapsed {dt:6.0f}s  [cpp x{self.threads}]",
                    flush=True,
                )
                if callback:
                    callback(self)
        return self

    # ---------------------------------------------------------------- outputs
    def strategy(self) -> BlueprintStrategy:
        return BlueprintStrategy({k: (list(a), list(p)) for k, (a, p) in self._core.strategy().items()})

    def blueprint(self, rounded: bool = False, keys: bool = False) -> CppBlueprint:
        """The average strategy as a C++ lookup (``fast.blueprint.CppBlueprint``): ``policy`` gives
        the floats ``strategy().policy`` gives (``rounded=True``: those of the saved file), without
        a Python dict.  ``keys``: keep the key strings too (``items``, ``save``)."""
        return CppBlueprint(self._core.blueprint_table(bool(rounded), bool(keys)))

    def save_blueprint(self, path: str, rounded: bool = True) -> None:
        """``*.json``: the bytes of ``strategy().save(path)``; any other name: the binary blueprint
        (probabilities rounded to 5 decimals like the JSON unless ``rounded=False``)."""
        if is_json_path(path):
            self._core.save_blueprint_json(str(path))
        else:
            self._core.save_blueprint_bin(str(path), bool(rounded))

    def strategy_change(self, prev: CppBlueprint) -> Optional[float]:
        """Mean L1 distance between the current average strategy and ``prev`` (a ``blueprint()``
        taken earlier) over the keys both have with the same actions, or None: the number
        ``scripts/train_blueprint.py``'s ``strategy_change(prev.to_strategy(), self.strategy())``
        computes, to the last bit, without either dict."""
        mean, _ = self._core.strategy_change(prev.lookup)
        return mean

    def count_keys(self, prefix: str, suffix: str = "") -> int:
        """Number of infoset keys starting with ``prefix`` and ending with ``suffix``."""
        return self._core.count_keys(prefix, suffix)

    def save_checkpoint(self, path: str) -> None:
        """``*.json``: the JSON checkpoint (the bytes json.dump wrote before, streamed from C++);
        any other name: the binary checkpoint (docs/backends.md).  Written to ``path + ".tmp"``
        first and renamed, so a crash never leaves a truncated checkpoint."""
        if is_json_path(path):
            self._core.save_checkpoint_json(str(path))
        else:
            self._core.save_checkpoint_bin(str(path))

    def load_checkpoint(self, path: str) -> "CppMCCFRTrainer":
        """Either format, told apart by the first bytes.  Binary: refused unless it was saved for this
        game (spec, grid, buckets).  JSON (either trainer's): streamed into the table by C++."""
        kind = core().file_kind(str(path))
        if kind == "checkpoint":
            self._core.load_checkpoint_bin(str(path))
        elif kind == "json":
            self._core.load_checkpoint_json(str(path))
        else:
            raise ValueError(f"{path}: not a checkpoint ({kind})")
        return self

    # ----------------------------------------------------- merge helpers (parallel driver)
    def export_tables(self) -> Dict[str, List]:
        return self._core.export_nodes()

    def add_tables(self, tables: Dict[str, List]) -> None:
        """Additive merge: regrets, strategy sums and visits from another trainer are summed in."""
        self._core.add_nodes(tables)
