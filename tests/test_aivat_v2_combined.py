"""AIVAT v2 with its options together: alloc=1 (rollouts by weight), turn_exact (turn states valued exactly),
preflop (a preflop all-in by the exact table), strat (a combo's rollouts take distinct first cards).

turn_exact and preflop value their own branches; alloc and strat act on the branches left to rollouts; none changes
the weights another reads.  So with every option on, each node value vector is, number for number, the one of the
configuration that values it: a turn-exact vector the one of turn_exact alone, a preflop all-in vector the one of
preflop alone, every other vector the one of alloc + strat.  Hence the combination is unbiased when each is (their
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
    hands = [h for h in hands if not _preflop_all_in(h)]  # (their vectors: test_the_preflop_vector_is_the_tables)
    kw = dict(rollouts=(1, 3, 3), eq_samples=100, seed=2, root=game["root"])
    both = FastAivat(game["game"], alloc=1, turn_exact=True, strat=True, **kw)
    alloc = FastAivat(game["game"], alloc=1, strat=True, **kw)
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


def _preflop_all_in(h):
    """True when some branch of the hand can be a preflop all-in (a raise to a whole stack before the flop)."""
    return any(t == 2 and a >= min(h["stacks"]) for t, a in h["actions"])


def test_the_preflop_vector_is_the_tables(game):  # noqa: F811
    spec = game["spec"]
    stack = spec.stacks[0]
    y_hole = (12, 17)
    x_combos = [(48, 49), (5, 9), (26, 33)]
    hand = {"hand_id": 7, "stacks": [stack, stack], "button": 0, "sb": spec.sb, "bb": spec.bb, "known_seat": 1,
            "holes": [list(y_hole), list(x_combos[0])], "board": [1, 22, 30, 40, 51], "actions": [[2, stack], [1, 0]]}
    idx = {c: i for i, c in enumerate((a, b) for a in range(52) for b in range(a + 1, 52))}
    P = core.AivatPreflopEquity
    probe = P.build(1, [])
    table = P.build(THREADS, [probe.class_of(idx[c], idx[y_hole]) for c in x_combos])
    kw = dict(rollouts=(1, 2, 2), eq_samples=100, seed=0, root=game["root"])
    every = FastAivat(game["game"], alloc=1, turn_exact=True, strat=True, preflop=table, **kw)
    alone = FastAivat(game["game"], preflop=table, **kw)
    combos = [idx[c] for c in x_combos]
    assert np.array_equal(every.branch_values(hand, 1, None, [], 3, combos), alone.branch_values(hand, 1, None, [], 3, combos))


def test_both_options_are_unbiased(game):  # noqa: F811
    ev = FastAivat(game["game"], (1, 4, 4), 50, 9, game["root"], alloc=1, turn_exact=True, strat=True)
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
