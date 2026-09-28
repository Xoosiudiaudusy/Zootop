"""A blueprint played at a stack depth other than its own (the 200bb blueprint at a 50bb table, a depth-grid point
between its neighbours, a carried stack): grid sizes that are the all-in at THIS depth must keep their probability.

BetGrid.abstract_actions drops a pot fraction whose chips reach the stack ("a fraction that hits the whole stack *is*
the all-in"), and BlueprintStrategy.policy / the C++ BlueprintTable renormalised the stored row over the legal names.
So the mass the row put on those sizes disappeared instead of going to "a"; a row whose mass sat on them only gave
None and the agent checked / called ("off the map").

H4 (defect hunter; fixed 2026-09-29): negpluribus/cfr/strategy.py policy_of_row and csrc/persist.h
BlueprintTable::policy_at now give the probability of a raise size the row has, the legal list lacks and that is
larger than every legal size of the row, to "a" when "a" is legal.  HU 50bb, SB opens pot (r1), BB 3-bets 3 x pot
(r3): the SB's legal names are [f, c, r0.5, a] (r1 and r3 would be all-in); a 200bb row [f, c, r0.5, r1, r3, a] =
[0, .2, 0, 0, .8, 0] used to play call 100 %, and [0, 0, 0, 0, 1, 0] gave None.
"""
from __future__ import annotations

import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import BetGrid, infoset_key
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cfr.strategy import BlueprintStrategy, policy_of_row, size_fraction
from negpluribus.engine import ActionType, HandState, raise_to

core = fast.core()
GRID = BetGrid(preflop_fracs=(0.5, 1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3)
ROW_NAMES = ["f", "c", "r0.5", "r1", "r3", "a"]  # the SB's row after "r1 r3" in the 200bb game


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


def test_policy_moves_the_mass_of_collapsed_sizes_to_the_all_in():
    obs, key, legal = _spot(50)
    bp = BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.2, 0.0, 0.0, 0.8, 0.0])})
    p = _as_dict(legal, bp.policy(key, legal))
    assert p["a"] == pytest.approx(0.8) and p["c"] == pytest.approx(0.2)


def test_a_row_entirely_on_collapsed_sizes_is_not_off_the_map():
    obs, key, legal = _spot(50)
    bp = BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.0, 0.0, 0.3, 0.7, 0.0])})
    p = bp.policy(key, legal)
    assert p is not None and _as_dict(legal, p)["a"] == pytest.approx(1.0)
    agent = BlueprintAgent(bp, _Buckets(), GRID, seed=1)
    a = agent.act(obs)
    assert agent.n_fallback == 0 and a.type == ActionType.RAISE and a.amount == obs.max_raise_to


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
    """Control: at 200bb every stored name is legal, so the row is played as stored."""
    obs, key, legal = _spot(200)
    probs = [0.0, 0.2, 0.0, 0.0, 0.8, 0.0]
    bp = BlueprintStrategy({key: (ROW_NAMES, probs)})
    assert bp.policy(key, legal) == pytest.approx(probs)


# ------------------------------------------------------------------------------------------ the rule, case by case
def _old_policy(names, probs, legal):
    """The lookup before H4: the stored probability of each legal name, renormalised (None: nothing left)."""
    lookup = dict(zip(names, probs))
    out = [lookup.get(a, 0.0) for a in legal]
    s = sum(out)
    return None if s <= 0 else [x / s for x in out]


def test_the_example_row_moves_exactly_the_collapsed_mass():
    """The 200bb row after "r1 r3" at 50bb: before, r1 and r3 went to the other names in proportion (fold 25 %,
    call 50 %, r0.5 25 %); now the all-in gets their 80 %, the rest keeps its stored probabilities."""
    obs, key, legal = _spot(50)
    probs = [0.05, 0.1, 0.05, 0.02, 0.78, 0.0]
    assert _old_policy(ROW_NAMES, probs, legal) == pytest.approx([0.25, 0.5, 0.25, 0.0])
    got, moved = policy_of_row(ROW_NAMES, probs, legal)
    assert moved == 0.02 + 0.78
    out = [0.05, 0.1, 0.05, 0.0 + 0.02 + 0.78]
    assert got == [x / sum(out) for x in out]
    assert BlueprintStrategy({key: (ROW_NAMES, probs)}).policy(key, legal) == got


def test_rows_whose_names_are_all_legal_are_unchanged():
    """At the row's own node every stored name is legal: the old floats, bit for bit (legal in any order, with legal
    names the row lacks too)."""
    rng = random.Random(5)
    grid_names = ["f", "c", "r0.5", "r1", "r2", "r3", "r4", "a"]
    for _ in range(3000):
        names = [n for n in grid_names if rng.random() < 0.6] or ["c"]
        probs = [round(rng.random(), 5) if rng.random() < 0.8 else 0.0 for _ in names]
        legal = list(names) + [n for n in grid_names if n not in names and rng.random() < 0.3]
        rng.shuffle(legal)
        got, moved = policy_of_row(names, probs, legal)
        assert moved == 0.0 and got == _old_policy(names, probs, legal)


def test_without_a_legal_all_in_the_missing_sizes_are_renormalised_away():
    """Facing an all-in nobody can raise: the legal names are f / c, and the row's sizes and all-in are renormalised
    away as before."""
    h = HandState([5000, 5000], button=0)
    h.apply(GRID.to_concrete(h.observe(), "r1"))
    h.apply(raise_to(5000))  # the BB shoves
    obs = h.observe()
    legal = GRID.abstract_actions(obs)
    assert not obs.can_raise and legal == ["f", "c"]
    names, probs = ["f", "c", "r0.5", "r1", "r3", "a"], [0.1, 0.1, 0.1, 0.2, 0.3, 0.2]
    got, moved = policy_of_row(names, probs, legal)
    assert moved == 0.0 and got == _old_policy(names, probs, legal) == [0.5, 0.5]


def test_at_a_raise_capped_node_the_row_is_played_as_stored():
    """The raise cap removes the sizes, not the all-in (f / c / a), and the key's history carries the raise count,
    so a row stored at a capped node has no sizes: nothing moves, the stored row as before."""
    h = HandState([20_000, 20_000], button=0)
    for _ in range(3):
        h.apply(GRID.to_concrete(h.observe(), "r1"))
    obs = h.observe()
    legal = GRID.abstract_actions(obs)
    assert obs.raises_this_street == GRID.max_raises_per_street and legal == ["f", "c", "a"]
    probs = [0.3, 0.5, 0.2]
    got, moved = policy_of_row(legal, probs, legal)
    assert moved == 0.0 and got == _old_policy(legal, probs, legal)


def test_a_size_below_a_legal_size_is_renormalised_away():
    """Only sizes above every legal size of the row are the all-in (chips grow with the fraction): a missing size
    below a legal one (a grid with small fractions, two of them clamped onto the same min-raise) keeps the old
    renormalisation."""
    names = ["f", "c", "r0.25", "r0.5", "r1", "a"]
    probs = [0.1, 0.2, 0.1, 0.3, 0.2, 0.1]
    legal = ["f", "c", "r0.25", "r1", "a"]
    got, moved = policy_of_row(names, probs, legal)
    assert moved == 0.0 and got == _old_policy(names, probs, legal)
    got, moved = policy_of_row(names, probs, ["f", "c", "r0.25", "a"])  # r0.5 and r1 above the largest legal size
    assert moved == 0.3 + 0.2


def test_size_fraction_reads_grid_names_only():
    assert [size_fraction(n) for n in ("r0.5", "r1", "r3", "r4", "r1e+06", "r.5")] == [0.5, 1.0, 3.0, 4.0, 1e6, 0.5]
    assert [size_fraction(n) for n in ("f", "c", "a", "r", "rx", "r+1", "r 1", "r1_0", "rinf", "rnan", 3)] == [None] * 11


def test_the_agent_counts_its_decisions_with_collapsed_sizes():
    obs, key, legal = _spot(50)
    bp = BlueprintStrategy({key: (ROW_NAMES, [0.0, 0.2, 0.0, 0.0, 0.8, 0.0])})
    agent = BlueprintAgent(bp, _Buckets(), GRID, seed=1, count_all_in=True)
    agent.act(obs)
    assert agent.n_all_in == 1 and agent.n_decisions == 1
    obs200, _, _ = _spot(200)
    agent.act(obs200)  # its own depth: nothing collapsed
    assert agent.n_all_in == 1 and agent.n_decisions == 2
    calm = BlueprintAgent(BlueprintStrategy({key: (ROW_NAMES, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])}), _Buckets(), GRID, seed=1,
                          count_all_in=True)
    calm.act(obs)  # no probability on the collapsed sizes: not counted
    assert calm.n_all_in == 0


# ------------------------------------------------------------------------------------------ Python == C++
def _random_rows(rng, n):
    grid_names = ["f", "c", "r0.5", "r0.75", "r1", "r1.5", "r2", "r3", "r4", "r10", "a"]
    rows = {}
    for i in range(n):
        names = [x for x in grid_names if rng.random() < 0.55] or ["c", "a"]
        probs = [rng.random() if rng.random() < 0.75 else 0.0 for _ in names]
        s = sum(probs) or 1.0
        rows[f"k{i}"] = (names, [p / s for p in probs])
    return rows, grid_names


@pytest.mark.skipif(core is None or not hasattr(core, "BlueprintTable"), reason="C++ core not built")
def test_python_and_cpp_give_the_same_floats(tmp_path):
    """Random rows (rounded to 5 decimals by the JSON; both sides read the same file) and random legal lists: the
    largest sizes dropped (collapsed), the all-in dropped or added, names the row lacks added, the order kept or
    shuffled.  BlueprintStrategy.policy (Python) and the C++ lookup agree exactly, None included."""
    from negpluribus.fast.blueprint import load_blueprint

    rng = random.Random(11)
    rows, grid_names = _random_rows(rng, 3000)
    path = tmp_path / "bp.json"
    BlueprintStrategy(rows).save(str(path))
    py = BlueprintStrategy.load(str(path))
    cpp = load_blueprint(str(path), backend="cpp", n_players=2)
    assert "a" in cpp.lookup.action_names
    n_moved = n_none = n = 0
    for key in rows:
        names, probs = py.table[key]
        for _ in range(6):
            sizes = [x for x in names if size_fraction(x) is not None]
            keep = rng.randrange(len(sizes) + 1)  # the smallest `keep` sizes stay legal, the others collapsed
            legal = [x for x in names if size_fraction(x) is None or x in sizes[:keep]]
            if rng.random() < 0.25 and "a" in legal:
                legal.remove("a")
            elif rng.random() < 0.3 and "a" not in legal:
                legal.append("a")
            legal += [x for x in grid_names if x not in names and rng.random() < 0.1]
            if rng.random() < 0.3:
                rng.shuffle(legal)
            if not legal:
                continue
            want, moved = policy_of_row(names, probs, legal)
            assert py.policy(key, legal) == want
            assert cpp.policy(key, legal) == want, (key, names, probs, legal)
            n += 1
            n_moved += moved > 0
            n_none += want is None
    assert n > 15000 and n_moved > 3000 and n_none > 50, (n, n_moved, n_none)
