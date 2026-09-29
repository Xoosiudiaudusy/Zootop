"""AIVAT on the C++ core (csrc/aivat.h): the production path.  negpluribus/eval/aivat.py and
aivat_values.py are the reference that defines the numbers (tests/test_aivat.py compares them).

    game = make_game(spec, core_bucketer, cpp_blueprint)          # AivatGame
    root = root_table(game, rollouts=64, seed=0, threads=4)        # AivatRootTable (cached on disk)
    ev = FastAivat(game, rollouts=(4, 8, 8), seed=0, root=root)
    out = ev.evaluate_many([hand_to_dict(h) for h in hands], threads=4)

Terms come back as (kind, street, action index, value) with kind 0 root, 1 seat, 2 x decision,
3 flop, 4 turn, 5 river (``TERM_NAMES``).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .aivat import AivatHand

TERM_NAMES = {0: "root", 1: "seat", 2: "x", 3: "flop", 4: "turn", 5: "river"}


def _core():
    from ..fast import core

    c = core()
    if c is None or not hasattr(c, "AivatEvaluator"):
        raise RuntimeError("the C++ core with AIVAT is not built (python scripts/build_fast.py)")
    return c


def hand_to_dict(h: AivatHand) -> dict:
    d = {"hand_id": int(h.hand_id), "stacks": list(h.stacks), "button": int(h.button), "sb": int(h.sb), "bb": int(h.bb),
         "known_seat": int(h.known_seat), "holes": [list(h.holes[0]), list(h.holes[1])], "board": list(h.board),
         "actions": [list(a) for a in h.actions]}
    if h.x_rows is not None:  # logged range strategies of x (docs/aivat.md section 7)
        d["x_rows"] = [None if r is None else {"actions": [list(a) for a in r.actions], "q": r.q} for r in h.x_rows]
    return d


def make_game(spec, core_bucketer, blueprint):
    """``blueprint``: a CppBlueprint (negpluribus.fast.blueprint.load_blueprint) or a core BlueprintTable."""
    from ..fast.trainer import spec_to_dict

    lookup = getattr(blueprint, "lookup", blueprint)
    return _core().AivatGame(spec_to_dict(spec), core_bucketer, lookup)


class FastAivat:
    def __init__(self, game, rollouts: Sequence[int] = (4, 8, 8), eq_samples: int = 2000, seed: int = 0, root=None,
                 alloc: int = 0, turn_exact: bool = False):
        """``alloc``: 0 heuristic v1 (``rollouts`` per combo and branch); 1 v2, rollouts by the combo's weight
        in the terms (at least 1).  ``turn_exact``: v2, turn states with decisions ahead valued exactly
        instead of by rollouts.  Independent options (docs/aivat.md)."""
        self.game = game
        self.rollouts = list(rollouts)
        self.eq_samples = eq_samples
        self.seed = seed
        self.root = root
        self.alloc = alloc
        self.turn_exact = turn_exact
        self.ev = _core().AivatEvaluator(game, self.rollouts, eq_samples, seed, root, alloc, turn_exact)

    def evaluate(self, hand, trace: bool = False) -> dict:
        return self.ev.evaluate(hand if isinstance(hand, dict) else hand_to_dict(hand), trace)

    def evaluate_many(self, hands: Iterable, threads: int = 1) -> List[dict]:
        return self.ev.evaluate_many([h if isinstance(h, dict) else hand_to_dict(h) for h in hands], threads)

    def branch_values(self, hand, k: int, action, fixed: Sequence[int], node: int, combos: Sequence[int]) -> np.ndarray:
        d = hand if isinstance(hand, dict) else hand_to_dict(hand)
        return np.array(self.ev.branch_values(d, k, action, list(fixed), node, list(combos)))


def root_table(game, rollouts: int, seed: int = 0, threads: int = 1, cache_path: Optional[str] = None,
               identity: Optional[dict] = None):
    """The root table (u_root per seat and suit class of the hole pair), built by rollouts or read
    from ``cache_path`` (.npz with the parameters it was built with; rebuilt when they differ)."""
    c = _core()
    meta = {"rollouts": int(rollouts), "seed": int(seed), **(identity or {})}
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if json.loads(str(z["meta"])) == meta:
            return c.AivatRootTable.from_values(list(z["v0"]), list(z["v1"]))
    rt = c.AivatRootTable.build(game, int(rollouts), int(seed), int(threads))
    if cache_path:
        v0, v1 = rt.values
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez(cache_path, v0=np.array(v0), v1=np.array(v1), meta=json.dumps(meta))
    return rt


def summarize(results: Sequence[dict], nets: Optional[Sequence[float]] = None, per: float = 100.0,
              pairs: Optional[Sequence[int]] = None) -> Dict[str, dict]:
    """bb/100 with 95% CIs for the raw nets and the AIVAT values (``per`` chips = 1 bb).  With
    ``pairs`` (a duplicate deal id per hand) a deal is one sample, as in negpluribus/eval/duel.py."""
    import math

    raw = np.array([r["net"] for r in results] if nets is None else nets, dtype=float) / per
    av = np.array([r["value"] for r in results], dtype=float) / per

    def stat(x):
        if pairs is not None:
            groups: Dict[int, float] = {}
            count: Dict[int, int] = {}
            for p, v in zip(pairs, x):
                groups[p] = groups.get(p, 0.0) + v
                count[p] = count.get(p, 0) + 1
            k = max(count.values())
            ys = np.array([groups[p] for p in groups]) / k
            m = float(np.mean(ys))
            sd = float(np.std(ys, ddof=1))
            n = len(ys)
            return {"bb100": m * 100, "ci95": 1.96 * sd / math.sqrt(n) * 100, "sd_per_sample": sd, "n": n,
                    "sd_per_hand": float(np.std(x, ddof=1))}
        m = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        n = len(x)
        return {"bb100": m * 100, "ci95": 1.96 * sd / math.sqrt(n) * 100, "sd_per_hand": sd, "n": n}

    return {"raw": stat(raw), "aivat": stat(av)}


def hands_needed(sd_per_hand_bb: float, half_width_bb100: float) -> float:
    """Hands for a 95% CI of +-half_width bb/100 at a per-hand sd (bb): (1.96 sd 100 / w)^2."""
    return (1.96 * sd_per_hand_bb * 100.0 / half_width_bb100) ** 2
