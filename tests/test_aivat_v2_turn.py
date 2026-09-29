"""AIVAT heuristic v2 "turn exact" (FastAivat(turn_exact=True), csrc/aivat.h; type (b), docs/aivat.md): a turn state
with decisions ahead and its turn card known is valued exactly (both betting trees walked with the blueprint's
policies, the river card enumerated) instead of by rollouts.

  * the exact values are the rollouts' expectation: on every turn-exact value vector of a set of hands, the
    difference "mean of K rollouts - exact" has mean 0 and the spread of the rollout noise alone (two seeds);
  * v2 stays unbiased: self-play mean 0, E[AIVAT - net] = 0 against a random bettor, every term kind mean 0;
  * without the flag: v1, number for number.
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
    b = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"], turn_exact=False)
    assert [_numbers(r) for r in a.evaluate_many(hands, THREADS)] == [_numbers(r) for r in b.evaluate_many(hands, THREADS)]


def test_exact_is_the_rollouts_expectation(game):  # noqa: F811
    K = 400
    hands = [hand_to_dict(h) for h in play(game, 200, "self", 91)]
    ex = FastAivat(game["game"], (1, 2, 2), 100, 0, game["root"], turn_exact=True)
    v1 = FastAivat(game["game"], (1, 2, 2), 100, 0, game["root"])
    mc = [FastAivat(game["game"], (1, 2, K), 100, s, game["root"]) for s in (0, 1)]
    de, dm, n_vec = [], [], 0
    for h in hands:
        te = ex.evaluate(h, True)["trace"]
        base = v1.evaluate(h, True)["trace"]
        m0, m1 = (m.evaluate(h, True)["trace"] for m in mc)
        for (name, va), (_, vb), (_, w0), (_, w1) in zip(te, base, m0, m1):
            va, vb, w0, w1 = map(np.array, (va, vb, w0, w1))
            if name.endswith(":before") or np.array_equal(va, vb):
                continue  # not valued by the turn walk (a flop-closing branch keeps its rollouts)
            used = va != 0
            n_vec += 1
            de.append((w0 - va)[used])
            dm.append((w1 - w0)[used] / np.sqrt(2))
    assert n_vec >= 5, n_vec
    de, dm = np.concatenate(de), np.concatenate(dm)
    assert len(de) >= 300, len(de)
    assert abs(de.mean()) < 4 * de.std() / np.sqrt(len(de)), (de.mean(), de.std(), len(de))
    assert 0.85 < de.std() / dm.std() < 1.15, (de.std(), dm.std())


def test_turn_exact_is_unbiased(game):  # noqa: F811
    te = FastAivat(game["game"], (1, 2, 2), 50, 9, game["root"], turn_exact=True)
    hands = play(game, 1500, "self", 31)
    res = te.evaluate_many(hands, THREADS)
    av = np.array([r["value"] for r in res]) / 100
    raw = np.array([h.net for h in hands]) / 100
    assert abs(_z(av)) < 4.0, (av.mean(), av.std())
    assert av.std() < 0.35 * raw.std(), (av.std(), raw.std())
    hands = play(game, 1000, "random", 32)
    res = te.evaluate_many(hands, THREADS)
    diff = np.array([r["value"] - h.net for r, h in zip(res, hands)])
    assert abs(_z(diff)) < 4.0, (diff.mean(), diff.std())
    by_kind = {}
    for r in res:
        for k, _, _, v in r["terms"]:
            by_kind.setdefault(TERM_NAMES[k], []).append(v)
    for k, v in by_kind.items():
        if k != "seat" and len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))
