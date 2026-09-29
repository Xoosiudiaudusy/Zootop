"""AIVAT speed work (csrc/aivat.h, type (a): the same numbers):

  * the rollouts of a branch give the same values whatever the number of them in flight (aivat_set_lanes);
  * the per-thread memo of policy and bucket lookups belongs to one game: evaluating hands of two games in
    turn on one thread gives each game's own numbers.

The tables path (a TabulatedBucketer: bucket lookups of later streets started ahead) is checked against the
core on the HUNL blueprint (docs/aivat.md section 4): here the small test game's bucketer has no tables.
"""
from __future__ import annotations

import pytest

from negpluribus import fast
from negpluribus.agents.base import RandomAgent  # noqa: F401  (the hands come from test_aivat.play)
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.eval.aivat_fast import FastAivat, hand_to_dict, make_game

from test_aivat import THREADS, game, play  # noqa: F401  (module fixture)

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "aivat_set_lanes"), reason="C++ core with AIVAT lanes not built")


def _numbers(r):
    return (r["net"], float(r["value"]).hex(), float(r["base"]).hex(), [(t[0], t[1], t[2], float(t[3]).hex()) for t in r["terms"]],
            r["rollouts"], r["rollout_steps"])


@pytest.fixture(scope="module")
def hands(game):  # noqa: F811
    return [hand_to_dict(h) for h in play(game, 10, "self", 71) + play(game, 10, "random", 72)]


def test_lanes_do_not_change_the_numbers(game, hands):  # noqa: F811
    ev = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"])
    try:
        ref = [_numbers(r) for r in ev.evaluate_many(hands, 1)]
        for lanes in (2, 4, 8):
            assert core.aivat_set_lanes(lanes) == lanes
            assert [_numbers(r) for r in ev.evaluate_many(hands, 1)] == ref, lanes
            assert [_numbers(r) for r in ev.evaluate_many(hands, THREADS)] == ref, lanes
    finally:
        core.aivat_set_lanes(1)
    assert core.aivat_set_lanes(0) == 1 and core.aivat_set_lanes(99) == 8
    core.aivat_set_lanes(1)


def test_the_lookup_memo_belongs_to_one_game(game, hands):  # noqa: F811
    t2 = MCCFRTrainer(game["spec"], game["bk"], seed=4, backend="cpp", threads=1).train(3000)
    g2 = make_game(game["spec"], game["cbk"], t2.blueprint())
    a = FastAivat(game["game"], (1, 2, 2), 200, 5, game["root"])
    b = FastAivat(g2, (1, 2, 2), 200, 5, game["root"])
    first_a = [_numbers(a.evaluate(h)) for h in hands[:8]]  # evaluate(): this thread, its memo kept between calls
    first_b = [_numbers(b.evaluate(h)) for h in hands[:8]]
    assert first_a != first_b  # the two blueprints differ where it shows
    for _ in range(2):
        assert [_numbers(a.evaluate(h)) for h in hands[:8]] == first_a
        assert [_numbers(b.evaluate(h)) for h in hands[:8]] == first_b
        mixed = [(_numbers(a.evaluate(h)), _numbers(b.evaluate(h))) for h in hands[:8]]
        assert [m[0] for m in mixed] == first_a and [m[1] for m in mixed] == first_b
