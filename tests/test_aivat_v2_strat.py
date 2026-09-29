"""AIVAT heuristic v2 "stratified" (FastAivat(strat=True), csrc/aivat.h; type (b), docs/aivat.md): the k rollouts of
a combo take the first missing board card from one random permutation of the cards left (rollout r: entry r mod n).

  * off: v1, number for number;
  * every rollout's run-out stays uniform on its own, so the estimate is unbiased: self-play mean 0, E[AIVAT - net]
    = 0 against a random bettor, every term kind mean 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from negpluribus import fast
from negpluribus.eval.aivat_fast import TERM_NAMES, FastAivat, hand_to_dict

from test_aivat import THREADS, _z, game, play  # noqa: F401  (module fixture)

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "AivatEvaluator"), reason="C++ core with AIVAT not built")


def _numbers(r):
    return (r["net"], float(r["value"]).hex(), [(t[0], t[1], t[2], float(t[3]).hex()) for t in r["terms"]], r["rollouts"])


def test_off_is_v1(game):  # noqa: F811
    hands = [hand_to_dict(h) for h in play(game, 12, "random", 81)]
    a = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"])
    b = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"], strat=False)
    c = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"], strat=True)
    ra, rb, rc = (e.evaluate_many(hands, THREADS) for e in (a, b, c))
    assert [_numbers(r) for r in ra] == [_numbers(r) for r in rb]
    assert [_numbers(r) for r in ra] != [_numbers(r) for r in rc]


def test_stratified_is_unbiased(game):  # noqa: F811
    ev = FastAivat(game["game"], (2, 4, 4), 50, 9, game["root"], strat=True)
    hands = play(game, 1500, "self", 31)
    res = ev.evaluate_many(hands, THREADS)
    av = np.array([r["value"] for r in res]) / 100
    assert abs(_z(av)) < 4.0, (av.mean(), av.std())
    hands = play(game, 1000, "random", 32)
    res = ev.evaluate_many(hands, THREADS)
    diff = np.array([r["value"] - h.net for r, h in zip(res, hands)])
    assert abs(_z(diff)) < 4.0, (diff.mean(), diff.std())
    by_kind = {}
    for r in res:
        for k, _, _, v in r["terms"]:
            by_kind.setdefault(TERM_NAMES[k], []).append(v)
    for k, v in by_kind.items():
        if k != "seat" and len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))
