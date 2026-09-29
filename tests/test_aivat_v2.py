"""AIVAT heuristic v2 "rollouts by weight" (FastAivat(alloc=1), csrc/aivat.h; type (b), docs/aivat.md):
combo c of a branch gets k_c = max(1, ceil(k n_eff w_c / sum w)) rollouts instead of k.

  * alloc=0 is heuristic v1, number for number;
  * v2 stays unbiased (the rollout counts depend only on weights known before the node's outcome, and every
    combo keeps at least one rollout): in blueprint self-play the AIVAT mean is within its CI of 0, against a
    random bettor AIVAT - net has mean 0, every kind of term has mean 0 (the tests of v1, test_aivat.py);
  * it spends fewer rollouts than v1 where x's range is uneven.
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


def test_alloc_0_is_v1(game):  # noqa: F811
    hands = [hand_to_dict(h) for h in play(game, 12, "random", 81)]
    v1 = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"])
    v1b = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"], alloc=0)
    assert [_numbers(r) for r in v1.evaluate_many(hands, THREADS)] == [_numbers(r) for r in v1b.evaluate_many(hands, THREADS)]
    with pytest.raises(ValueError):
        FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"], alloc=2)


def _kinds(res):
    by_kind = {}
    for r in res:
        for k, _, _, v in r["terms"]:
            by_kind.setdefault(TERM_NAMES[k], []).append(v)
    return by_kind


def test_v2_is_unbiased(game):  # noqa: F811
    v2 = FastAivat(game["game"], (1, 4, 4), 50, 9, game["root"], alloc=1)
    v1 = FastAivat(game["game"], (1, 4, 4), 50, 9, game["root"])
    # self-play, alternating seats: the exact value is 0
    hands = play(game, 1500, "self", 31)
    res = v2.evaluate_many(hands, THREADS)
    av = np.array([r["value"] for r in res]) / 100
    raw = np.array([h.net for h in hands]) / 100
    assert abs(_z(av)) < 4.0, (av.mean(), av.std())
    assert av.std() < 0.35 * raw.std(), (av.std(), raw.std())
    for k, v in _kinds(res).items():
        if k != "seat" and len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))
    # fewer rollouts than v1 on the same hands
    assert sum(r["rollouts"] for r in res) < sum(r["rollouts"] for r in v1.evaluate_many(hands, THREADS))
    # against a random bettor (off-grid sizes, the translation coins): E[AIVAT - net] = 0
    hands = play(game, 1000, "random", 32)
    res = v2.evaluate_many(hands, THREADS)
    diff = np.array([r["value"] - h.net for r, h in zip(res, hands)])
    assert abs(_z(diff)) < 4.0, (diff.mean(), diff.std())
    for k, v in _kinds(res).items():
        if k != "seat" and len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))
