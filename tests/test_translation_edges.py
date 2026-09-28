"""Bet translation at the edges of the abstraction (negpluribus/abstraction/actions.py, BetGrid.from_concrete).

The blueprint's keys only exist for the abstract actions ``abstract_actions`` lists at a node, so the translation
of an observed raise must be one of them: otherwise every later key of the hand is unknown and the blueprint agent
plays check/call ("off the map").  Checked on the production grid (HU 200bb: preflop 0.5/1/3 pot, postflop
0.5/1/2/4 pot, 3 raises per street) and a 3-max game, with and without the per-event coins.

QA-1 (found while writing these tests, production code unchanged): at a raise-capped node (``raises_this_street >=
max_raises_per_street``) the abstract actions are f / c / a, but a non-all-in raise is still translated between the
grid's pot fractions ("r1", "r3"), which do not exist there.  At 200bb a preflop 5-bet short of all-in (open, 3-bet,
4-bet, 5-bet) reaches it; the blueprint agent then misses every key of the hand and calls.
"""
from __future__ import annotations

import random

import pytest

from negpluribus.abstraction import BetGrid, infoset_key
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cfr.strategy import BlueprintStrategy
from negpluribus.engine import ActionType, HandState, Street, raise_to

GRID_HU200 = BetGrid(preflop_fracs=(0.5, 1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3)
GRID_SMALL = BetGrid(preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0), max_raises_per_street=2)


def _random_raises(grid, n_players, stacks_bb, n_hands, seed):
    """(obs before the raise, the raise event) for every raise of random hands: legal actions, raise sizes drawn
    uniformly between the min-raise and the all-in half the time (off-grid sizes), grid sizes otherwise."""
    rng = random.Random(seed)
    out = []
    for i in range(n_hands):
        stacks = [rng.choice(stacks_bb) * 100 for _ in range(n_players)]
        h = HandState(stacks, button=i % n_players, seed=seed * 100_003 + i)
        while not h.is_terminal:
            obs = h.observe()
            names = grid.abstract_actions(obs)
            if obs.can_raise and rng.random() < 0.35:
                a = raise_to(rng.randint(obs.min_raise_to, obs.max_raise_to))
            else:
                name = rng.choice(names)
                if name == "f" and rng.random() < 0.6:
                    name = "c"
                a = grid.to_concrete(obs, name)
            ev = h.apply(a)
            if ev.action.type == ActionType.RAISE:
                out.append((obs, ev))
    return out


@pytest.mark.parametrize("grid,n_players,stacks_bb", [
    (GRID_HU200, 2, [200]),
    (GRID_HU200, 2, [20, 50, 100, 150, 200]),  # the depths of the stack grid
    (GRID_SMALL, 3, [15, 30, 60]),             # 3-max, unequal stacks
])
def test_a_raise_below_the_raise_cap_translates_to_an_action_of_its_node(grid, n_players, stacks_bb):
    raises = _random_raises(grid, n_players, stacks_bb, n_hands=700, seed=n_players * 7 + len(stacks_bb))
    uncapped = [(o, e) for o, e in raises if o.raises_this_street < grid.max_raises_per_street]
    assert len(uncapped) > 1000
    for obs, ev in uncapped:
        legal = grid.abstract_actions(obs)
        assert grid.from_concrete(ev, ev.all_in) in legal, (obs.street, legal, ev)
        for k in range(3):  # whatever the translation coin says
            assert grid.from_concrete(ev, ev.all_in, random.Random(k)) in legal, (obs.street, legal, ev, k)


def test_an_all_in_raise_is_the_all_in_even_when_short_of_a_min_raise():
    h = HandState([20_000, 20_000, 1000], button=0)  # seat 2 is the big blind with 10bb
    h.apply(raise_to(300))   # BTN opens
    h.apply(raise_to(900))   # SB 3-bets (raise size 600)
    obs = h.observe()        # BB: its all-in to 1000 is short of the min-raise to 1500
    assert obs.can_raise and obs.min_raise_to == obs.max_raise_to == 1000
    assert "a" in GRID_HU200.abstract_actions(obs)
    ev = h.apply(raise_to(1000))
    assert ev.all_in and GRID_HU200.from_concrete(ev, ev.all_in) == "a"
    assert {GRID_HU200.from_concrete(ev, ev.all_in, random.Random(k)) for k in range(20)} == {"a"}


@pytest.mark.parametrize("line", [[], ["r1"], ["r1", "r3"], ["c", "c"], ["c", "c", "r1"]])  # opens, 3-/4-bets, flop
def test_every_grid_size_translates_to_itself_with_any_coin(line):
    checked = 0
    for size in ("r0.5", "r1", "r3", "r2", "r4"):
        h = HandState([20_000, 20_000], button=0)
        for name in line:
            h.apply(GRID_HU200.to_concrete(h.observe(), name))
        obs = h.observe()
        if size not in GRID_HU200.abstract_actions(obs):
            continue
        ev = h.apply(GRID_HU200.to_concrete(obs, size))
        assert GRID_HU200.from_concrete(ev, ev.all_in) == size
        assert {GRID_HU200.from_concrete(ev, ev.all_in, random.Random(k)) for k in range(40)} == {size}
        checked += 1
    assert checked >= 2


def _five_bet_short_of_all_in():
    """HU 200bb, SB opens pot, BB 3-bets pot, SB 4-bets pot (all on the grid), BB 5-bets to 60bb (not all-in):
    the fourth raise of the preflop, at a node where only f / c / a exist."""
    h = HandState([20_000, 20_000], button=0)
    for _ in range(3):
        h.apply(GRID_HU200.to_concrete(h.observe(), "r1"))
    obs = h.observe()
    assert obs.raises_this_street == GRID_HU200.max_raises_per_street
    assert GRID_HU200.abstract_actions(obs) == ["f", "c", "a"]
    ev = h.apply(raise_to(6000))
    assert not ev.all_in
    return h, obs, ev


@pytest.mark.xfail(strict=True, reason="QA-1: BetGrid.from_concrete (abstraction/actions.py) ignores the raise cap: a "
                                        "non-all-in raise at a capped node becomes 'r1'/'r3', which that node does not have")
def test_a_raise_at_a_raise_capped_node_translates_to_the_all_in():
    _, obs, ev = _five_bet_short_of_all_in()
    assert GRID_HU200.from_concrete(ev, ev.all_in) == "a"
    assert {GRID_HU200.from_concrete(ev, ev.all_in, random.Random(k)) for k in range(40)} == {"a"}


class _Buckets:
    """Every hand in bucket 0 (preflop too): the key depends on the history only."""

    def bucket(self, hole, board):
        return 0


@pytest.mark.xfail(strict=True, reason="QA-1: after a capped-node raise the blueprint agent looks up a history the "
                                        "abstract game never has, falls back to check/call and counts it off the map")
def test_the_blueprint_agent_finds_its_key_after_a_raise_at_a_capped_node():
    h, capped_obs, ev = _five_bet_short_of_all_in()
    obs = h.observe()  # the SB answers the 5-bet
    legal = GRID_HU200.abstract_actions(obs)
    hist = " ".join(["r1", "r1", "r1", "a"])  # the abstract game's only raise at the capped node is the all-in
    key = f"P|{obs.position}|2|b0|{hist}"
    strategy = BlueprintStrategy({key: (legal, [1.0] + [0.0] * (len(legal) - 1))})  # fold to the 5-bet
    agent = BlueprintAgent(strategy, _Buckets(), GRID_HU200, seed=1)
    a = agent.act(obs)
    assert agent.n_fallback == 0 and a.type == ActionType.FOLD, "off the map after a capped-node raise"
    assert infoset_key(obs, _Buckets(), GRID_HU200) == key
