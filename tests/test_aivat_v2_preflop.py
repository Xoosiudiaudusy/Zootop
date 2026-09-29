"""AIVAT heuristic v2 "preflop exact" (FastAivat(preflop=table), csrc/aivat.h PreflopEquity; type (b), docs/aivat.md):
a preflop all-in is valued by the exact equity over every board instead of 2000 random boards.

  * the table: wins - losses over the 1,712,304 boards per suit class of the hole pair, antisymmetric (the mirror
    class has minus the value), and it agrees with an independent Monte-Carlo count;
  * the flop term stays exactly consistent: the preflop all-in value of a combo equals the mean, over every flop,
    of the values after the flop (v1's exact enumeration of the turn and river) - to rounding;
  * without a table: v1, number for number.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from negpluribus import fast
from negpluribus.eval.aivat_fast import FastAivat, hand_to_dict

from test_aivat import THREADS, game, play  # noqa: F401  (module fixture)

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "AivatPreflopEquity"), reason="C++ core with AIVAT v2 not built")
IDX = {c: i for i, c in enumerate((a, b) for a in range(52) for b in range(a + 1, 52))}


def _numbers(r):
    return (r["net"], float(r["value"]).hex(), [(t[0], t[1], t[2], float(t[3]).hex()) for t in r["terms"]], r["rollouts"])


def test_no_table_is_v1(game):  # noqa: F811
    hands = [hand_to_dict(h) for h in play(game, 12, "random", 81)]
    a = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"])
    b = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"], preflop=None)
    assert [_numbers(r) for r in a.evaluate_many(hands, THREADS)] == [_numbers(r) for r in b.evaluate_many(hands, THREADS)]


def test_table_entries():
    P = core.AivatPreflopEquity
    probe = P.build(1, [])
    pairs = [((48, 49), (46, 47)), ((47, 51), (1, 2)), ((0, 4), (49, 50)), ((20, 21), (22, 23)), ((8, 33), (9, 32))]
    cls = [probe.class_of(IDX[x], IDX[y]) for x, y in pairs]
    t = P.build(THREADS, cls)
    rng = random.Random(5)
    n = 60_000
    for (x, y), o in zip(pairs, cls):
        assert t.net[t.class_of(IDX[y], IDX[x])] == -t.net[o]  # antisymmetric
        rest = [k for k in range(52) if k not in x + y]
        hands = []
        for _ in range(n):
            b = rng.sample(rest, 5)
            hands += [list(x) + b, list(y) + b]
        r = core.evaluate_many(hands)
        mc = sum((r[2 * i] > r[2 * i + 1]) - (r[2 * i] < r[2 * i + 1]) for i in range(n)) / n
        assert abs(mc - t.net[o] / P.boards) < 4.5 / np.sqrt(n), (x, y, mc, t.net[o] / P.boards)
    assert abs(t.net[cls[0]] / P.boards - 0.625) < 0.01  # AA vs KK, no shared suit: about 81% to win


def test_the_flop_term_stays_exactly_consistent(game):  # noqa: F811
    spec = game["spec"]
    stack = spec.stacks[0]
    y_hole, board = (12, 17), (1, 22, 30, 40, 51)
    x_combos = [(48, 49), (5, 9), (26, 33)]
    hand = {"hand_id": 7, "stacks": [stack, stack], "button": 0, "sb": spec.sb, "bb": spec.bb, "known_seat": 1,
            "holes": [list(y_hole), list(x_combos[0])], "board": list(board),
            "actions": [[2, stack], [1, 0]]}  # the small blind shoves, x (the big blind) calls
    P = core.AivatPreflopEquity
    probe = P.build(1, [])
    di = IDX[y_hole]
    table = P.build(THREADS, [probe.class_of(IDX[c], di) for c in x_combos])
    ev = FastAivat(game["game"], (1, 2, 2), 100, 0, game["root"], preflop=table)
    combos = [IDX[c] for c in x_combos]
    before = ev.branch_values(hand, 1, None, [], 3, combos)[combos]
    known = set(y_hole)
    tot = np.zeros(len(combos))
    cnt = np.zeros(len(combos))
    for flop in itertools.combinations([k for k in range(52) if k not in known], 3):
        after = ev.branch_values(hand, 1, None, list(flop), 3, combos)[combos]
        for i, c in enumerate(x_combos):
            if not set(flop) & set(c):
                tot[i] += after[i]
                cnt[i] += 1
    assert np.allclose(tot / cnt, before, rtol=1e-12, atol=1e-9), (tot / cnt, before)
    v1 = FastAivat(game["game"], (1, 2, 2), 2000, 0, game["root"])
    assert not np.array_equal(v1.branch_values(hand, 1, None, [], 3, combos)[combos], before)  # v1: 2000 random boards
