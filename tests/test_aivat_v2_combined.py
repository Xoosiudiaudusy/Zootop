"""AIVAT v2 with both options (alloc=1: rollouts by weight; turn_exact=True: turn states valued exactly).

The options act on different branches and neither changes the weights the other reads, so with both on every node
value vector is, number for number, the one of the option that values it: a turn-exact vector equals the one of
turn_exact alone, every other vector the one of alloc alone.  Hence the combination is unbiased when each is (their
tests); checked here as well on the small game.
"""
from __future__ import annotations

import numpy as np
import pytest

from negpluribus import fast
from negpluribus.eval.aivat_fast import TERM_NAMES, FastAivat, hand_to_dict

from test_aivat import THREADS, _z, game, play  # noqa: F401  (module fixture)

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "AivatEvaluator"), reason="C++ core with AIVAT not built")


def test_each_vector_is_the_one_of_its_option(game):  # noqa: F811
    hands = [hand_to_dict(h) for h in play(game, 120, "self", 93) + play(game, 60, "random", 94)]
    kw = dict(rollouts=(1, 3, 3), eq_samples=100, seed=2, root=game["root"])
    both = FastAivat(game["game"], alloc=1, turn_exact=True, **kw)
    alloc = FastAivat(game["game"], alloc=1, **kw)
    turn = FastAivat(game["game"], turn_exact=True, **kw)
    v1 = FastAivat(game["game"], **kw)
    n_turn = n_alloc = 0
    for h in hands:
        tb, ta, tt, t1 = (e.evaluate(h, True)["trace"] for e in (both, alloc, turn, v1))
        for (name, vb), (_, va), (_, vt), (_, v) in zip(tb, ta, tt, t1):
            if vt != v and vt == vb and not name.endswith(":before"):
                n_turn += 1  # valued by the turn walk
            else:
                assert vb == va, (h["hand_id"], name)
                n_alloc += vb != v
    assert n_turn >= 5 and n_alloc >= 20, (n_turn, n_alloc)


def test_both_options_are_unbiased(game):  # noqa: F811
    ev = FastAivat(game["game"], (1, 4, 4), 50, 9, game["root"], alloc=1, turn_exact=True)
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
