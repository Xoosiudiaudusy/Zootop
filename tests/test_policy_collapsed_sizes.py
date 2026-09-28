"""A blueprint played at a stack depth other than its own (the 200bb blueprint at a 50bb table, a depth-grid point
between its neighbours, a carried stack): grid sizes that are the all-in at THIS depth must keep their probability.

BetGrid.abstract_actions drops a pot fraction whose chips reach the stack ("a fraction that hits the whole stack *is*
the all-in"), and BlueprintStrategy.policy / the C++ BlueprintTable renormalise the stored row over the legal names.
So the mass the row put on those sizes disappears instead of going to "a"; a row whose mass sat on them only gives
None and the agent checks / calls ("off the map").

H4 (defect hunter; confirmed here): negpluribus/abstraction/actions.py:73-88 with negpluribus/cfr/strategy.py:65-75
and csrc/persist.h policy_at.  HU 50bb, SB opens pot (r1), BB 3-bets 3 x pot (r3): the SB's legal names are
[f, c, r0.5, a] (r1 and r3 would be all-in); a 200bb row [f, c, r0.5, r1, r3, a] = [0, .2, 0, 0, .8, 0] plays
call 100 %, and [0, 0, 0, 0, 1, 0] gives None.
"""
from __future__ import annotations

import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import BetGrid, infoset_key
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cfr.strategy import BlueprintStrategy
from negpluribus.engine import ActionType, HandState

core = fast.core()
GRID = BetGrid(preflop_fracs=(0.5, 1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3)
ROW_NAMES = ["f", "c", "r0.5", "r1", "r3", "a"]  # the SB's row after "r1 r3" in the 200bb game
H4 = ("H4: BlueprintStrategy.policy / BlueprintTable.policy_at renormalise over the legal names, so the mass of grid "
      "sizes that are the all-in at this depth is dropped instead of moved to 'a' (None when it was all there)")


class _Buckets:
    def bucket(self, hole, board):
        return 0


def _spot(stack_bb=50):
    h = HandState([stack_bb * 100] * 2, button=0)
    h.apply(GRID.to_concrete(h.observe(), "r1"))   # SB opens pot, to 300
    h.apply(GRID.to_concrete(h.observe(), "r3"))   # BB 3-bets 3 x pot, to 2100
    obs = h.observe()
    return obs, infoset_key(obs, _Buckets(), GRID), GRID.abstract_actions(obs)


def test_the_spot_collapses_r1_and_r3_into_the_all_in_at_50bb_only():
    obs, key, legal = _spot(50)
    assert key == "P|BTN/SB|2|b0|r1 r3" and legal == ["f", "c", "r0.5", "a"]
    _, key200, legal200 = _spot(200)
    assert key200 == key and legal200 == ROW_NAMES  # the same key: the 200bb row is what a 50bb table looks up


def _as_dict(legal, probs):
    return dict(zip(legal, probs))


@pytest.mark.xfail(strict=True, reason=H4)
def test_policy_moves_the_mass_of_collapsed_sizes_to_the_all_in():
    obs, key, legal = _spot(50)
    bp = BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.2, 0.0, 0.0, 0.8, 0.0])})
    p = _as_dict(legal, bp.policy(key, legal))
    assert p["a"] == pytest.approx(0.8) and p["c"] == pytest.approx(0.2)


@pytest.mark.xfail(strict=True, reason=H4)
def test_a_row_entirely_on_collapsed_sizes_is_not_off_the_map():
    obs, key, legal = _spot(50)
    bp = BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.0, 0.0, 0.3, 0.7, 0.0])})
    p = bp.policy(key, legal)
    assert p is not None and _as_dict(legal, p)["a"] == pytest.approx(1.0)
    agent = BlueprintAgent(bp, _Buckets(), GRID, seed=1)
    a = agent.act(obs)
    assert agent.n_fallback == 0 and a.type == ActionType.RAISE and a.amount == obs.max_raise_to


@pytest.mark.xfail(strict=True, reason=H4)
def test_the_blueprint_agent_shoves_as_often_as_the_row_raises_big():
    obs, key, legal = _spot(50)
    bp = BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.2, 0.0, 0.0, 0.8, 0.0])})
    rng = random.Random(3)
    shoves = 0
    for i in range(400):
        agent = BlueprintAgent(bp, _Buckets(), GRID, seed=rng.getrandbits(32))
        a = agent.act(obs)
        shoves += a.type == ActionType.RAISE and a.amount == obs.max_raise_to
    assert 0.7 < shoves / 400 < 0.9, shoves


@pytest.mark.skipif(core is None or not hasattr(core, "BlueprintTable"), reason="C++ core not built")
@pytest.mark.xfail(strict=True, reason=H4 + " (C++ lookup, csrc/persist.h policy_at)")
def test_the_cpp_lookup_moves_the_mass_of_collapsed_sizes_to_the_all_in(tmp_path):
    from negpluribus.fast.blueprint import load_blueprint

    obs, key, legal = _spot(50)
    path = tmp_path / "bp.json"
    BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.2, 0.0, 0.0, 0.8, 0.0])}).save(str(path))
    lk = load_blueprint(str(path), backend="cpp", n_players=2)
    assert key in lk
    p = lk.policy(key, legal)
    assert p is not None and _as_dict(legal, p)["a"] == pytest.approx(0.8)


def test_at_the_blueprints_own_depth_nothing_changes():
    """Control: at 200bb every stored name is legal, so the row is played as stored (passes today, must stay)."""
    obs, key, legal = _spot(200)
    probs = [0.0, 0.2, 0.0, 0.0, 0.8, 0.0]
    bp = BlueprintStrategy({key: (ROW_NAMES, probs)})
    assert bp.policy(key, legal) == pytest.approx(probs)
