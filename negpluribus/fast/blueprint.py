"""Blueprints on the C++ core: a lean lookup for the agents, binary files, JSON kept readable.

    bp = load_blueprint("data/blueprint_X.bin")      # or an old blueprint_X.json: same numbers
    agent = BlueprintAgent(bp, bucketer, spec.grid, seed=1)

``CppBlueprint.policy(key, legal)`` is ``BlueprintStrategy.policy`` computed in C++ on flat arrays
(numeric keys sorted, per infoset an offset into action indices and probabilities; no Python object
per infoset): the same floats for every key and legal list, so an agent samples the same actions
on the same seeds.  About 60 bytes per infoset instead of about 590 for the dict
(docs/backends.md, "Binary checkpoints and blueprints").

Files: ``blueprint_*.bin`` (binary, written by the C++ trainer) or the JSON of
``BlueprintStrategy.save``; ``load_blueprint`` tells them apart by their first bytes.  The binary
file stores the probabilities rounded to 5 decimals exactly as the JSON does (``round(p, 5)``), so
both formats of one training state give the same decisions.

``NEGPLURIBUS_BLUEPRINT`` = ``cpp`` (default when the core is built) / ``python``: what
``load_blueprint`` returns by default (``python``: a ``BlueprintStrategy`` dict, binary files read
by the pure-Python reader in ``fast.binfmt``).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

from ..cfr.strategy import BlueprintStrategy
from . import core


def is_json_path(path: str) -> bool:
    """Output format by name: ``*.json`` is JSON, anything else binary."""
    return str(path).lower().endswith(".json")


def newest_file(paths) -> Optional[str]:
    """The most recently written of the existing ``paths`` (the earlier one on a tie), or None."""
    found = [(os.path.getmtime(p), -i, p) for i, p in enumerate(paths) if os.path.exists(p)]
    return max(found)[2] if found else None


def tagged_path(data_dir: str, kind: str, tag: str) -> str:
    """``<data_dir>/<kind>_<tag>.bin`` or ``.json``, whichever was written last (the C++ trainer
    writes binary by default, the Python one JSON); the ``.json`` name when neither exists."""
    stem = os.path.join(data_dir, f"{kind}_{tag}")
    return newest_file([stem + ".bin", stem + ".json"]) or stem + ".json"


def file_kind(path: str) -> str:
    """``checkpoint`` / ``blueprint`` (binary, by their magic bytes), ``json``, ``unknown`` or ``unreadable``."""
    c = core()
    if c is not None:
        return c.file_kind(str(path))
    from .binfmt import file_kind as py_file_kind

    return py_file_kind(str(path))


def default_blueprint_backend() -> str:
    v = os.environ.get("NEGPLURIBUS_BLUEPRINT", "").strip().lower()
    if v in ("python", "dict"):
        return "python"
    if v in ("", "cpp"):
        return "cpp" if core() is not None else "python"
    raise ValueError(f"NEGPLURIBUS_BLUEPRINT must be 'cpp' or 'python', got {v!r}")


class CppBlueprint:
    """Average strategy held by the C++ core (``_fastcore.BlueprintTable``); duck-types
    ``BlueprintStrategy`` for the agents: ``policy``, ``len``, ``in``, ``get``.

    ``policy`` is the C++ method itself (no Python frame per decision).  The key strings are
    loaded only with ``keys=True`` (``items``, ``to_strategy``, ``save``); ``table`` builds the
    old dict on first use for code that still reads it (costs the dict's memory)."""

    backend = "cpp"

    def __init__(self, lookup, path: Optional[str] = None):
        self.lookup = lookup
        self.path = path
        self.policy = lookup.policy
        self._dict: Optional[BlueprintStrategy] = None

    def __len__(self) -> int:
        return len(self.lookup)

    def __contains__(self, key: object) -> bool:
        return key in self.lookup

    def get(self, key: str) -> Optional[Tuple[List[str], List[float]]]:
        return self.lookup.get(key)

    @property
    def iteration(self) -> int:
        return self.lookup.iteration

    @property
    def rounded(self) -> bool:
        return self.lookup.rounded

    @property
    def identity(self) -> Optional[dict]:
        return self.lookup.identity

    def stats(self) -> Dict[str, int]:
        """Bytes of the lookup by part (keys, offsets, action indices, probabilities, directory, key strings)."""
        return self.lookup.stats()

    def _with_keys(self):
        if self.lookup.has_keys:
            return self.lookup
        if self.path is None:
            raise RuntimeError("this blueprint has no key strings (build it with keys=True)")
        return core().BlueprintTable.load(self.path, keys=True, n_players=self.lookup.n_players)

    def items(self) -> List[Tuple[str, List[str], List[float]]]:
        return self._with_keys().items()

    def to_strategy(self) -> BlueprintStrategy:
        """The same contents as a ``BlueprintStrategy`` dict."""
        return BlueprintStrategy({k: (names, probs) for k, names, probs in self.items()})

    @property
    def table(self) -> Dict[str, Tuple[List[str], List[float]]]:
        if self._dict is None:
            self._dict = self.to_strategy()
        return self._dict.table

    def save(self, path: str) -> None:
        """``*.json``: the JSON of ``BlueprintStrategy.save``; else a binary blueprint."""
        lk = self._with_keys()
        if is_json_path(path):
            lk.save_json(str(path))
        else:
            lk.save(str(path))


def load_blueprint(path: str, backend: Optional[str] = None, keys: bool = False,
                   n_players: int = 0) -> Union[CppBlueprint, BlueprintStrategy]:
    """A blueprint file of either format.  ``backend="cpp"`` (default when the core is built):
    a ``CppBlueprint``; ``"python"``: a ``BlueprintStrategy`` dict.  ``keys``: keep the key
    strings in the C++ lookup too.  ``n_players``: player count of the numeric keys of a JSON file
    (0 = from its keys)."""
    b = (backend or default_blueprint_backend()).lower()
    kind = file_kind(path)
    if kind == "checkpoint":
        raise ValueError(f"{path} is a checkpoint, not a blueprint")
    if kind not in ("blueprint", "json"):
        raise ValueError(f"{path}: not a blueprint ({kind})")
    if b == "cpp":
        if core() is None:
            raise RuntimeError("backend='cpp' needs the C++ core (python scripts/build_fast.py)")
        return CppBlueprint(core().BlueprintTable.load(str(path), keys=keys, n_players=n_players), path=str(path))
    if b != "python":
        raise ValueError(f"backend must be 'cpp' or 'python', got {backend!r}")
    if kind == "json":
        return BlueprintStrategy.load(str(path))
    if core() is not None:  # a binary file as the dict, read by C++
        return CppBlueprint(core().BlueprintTable.load(str(path), keys=True), path=str(path)).to_strategy()
    from .binfmt import read_blueprint

    return read_blueprint(str(path))
