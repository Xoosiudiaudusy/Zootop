import random

import pytest

from negpluribus.abstraction import BetGrid, EquityBucketer, canonical_form, history_string, infoset_key
from negpluribus.abstraction.actions import pseudo_harmonic
from negpluribus.cards import cards_from_str
from negpluribus.engine import CALL, FOLD, HandState, Street, raise_to


def cs(s):
    return cards_from_str(s)


# ---------------------------------------------------------------- canonical
def test_canonical_merges_suit_relabellings():
    a = canonical_form(cs("Ah Kh"), cs("2h 7c 9s"))
    b = canonical_form(cs("As Ks"), cs("2s 7c 9h"))
    c = canonical_form(cs("Ad Kd"), cs("2d 7h 9c"))
    assert a == b == c


def test_canonical_keeps_real_differences():
    suited = canonical_form(cs("Ah Kh"), cs("2h 7c 9s"))
    offsuit = canonical_form(cs("Ah Kc"), cs("2h 7c 9s"))
    assert suited != offsuit
    # paired board with matching hole suit vs not
    x = canonical_form(cs("Ah 3c"), cs("7h 7s Kd"))
    y = canonical_form(cs("As 3c"), cs("7h 7s Kd"))
    assert x == y  # symmetric: ace matches one of the two sevens either way
    z = canonical_form(cs("Ad 3c"), cs("7h 7s Kd"))
    assert z != x  # ace shares suit with the king instead


# ------------------------------------------------------------------ buckets
@pytest.fixture(scope="module")
def bucketer():
    return EquityBucketer(n_buckets=5, samples=60).fit(n_situations=300, seed=1)


def test_preflop_is_lossless_169(bucketer):
    assert bucketer.bucket(cs("Ah Ad"), []) == 0  # AA is class index 0
    assert bucketer.bucket(cs("Ah Kh"), []) == bucketer.bucket(cs("As Ks"), [])
    assert bucketer.bucket(cs("Ah Kh"), []) != bucketer.bucket(cs("Ah Kc"), [])
    assert bucketer.n_buckets_for(Street.PREFLOP) == 169


def test_postflop_buckets_monotone_in_strength(bucketer):
    board = cs("Kh 9h 2c")
    assert bucketer.bucket(cs("Kc Kd"), board) >= bucketer.bucket(cs("Kd 3d"), board) >= bucketer.bucket(cs("6d 5d"), board)
    assert bucketer.bucket(cs("Kc Kd"), board) == 4  # top set is top bucket
    assert bucketer.bucket(cs("6d 5d"), board) == 0
    for b in (cs("Kh 9h 2c"), cs("Kh 9h 2c 5d"), cs("Kh 9h 2c 5d Js")):
        assert 0 <= bucketer.bucket(cs("Ah Qh"), b) < 5


def test_bucketer_roundtrip(tmp_path, bucketer):
    p = tmp_path / "b.json"
    bucketer.save(str(p))
    b2 = EquityBucketer.load(str(p))
    assert b2.boundaries == bucketer.boundaries
    assert b2.bucket(cs("Ah Qh"), cs("Kh 9h 2c")) == bucketer.bucket(cs("Ah Qh"), cs("Kh 9h 2c"))


# ------------------------------------------------------------------ actions
def test_pseudo_harmonic_endpoints_and_midpoint():
    assert pseudo_harmonic(0.3, (0.5, 1.0)) == 0.5
    assert pseudo_harmonic(1.5, (0.5, 1.0)) == 1.0
    # exact formula at x=0.75: p_A = (1-0.75)(1.5)/((0.5)(1.75)) = 0.4286 -> deterministic picks B
    assert pseudo_harmonic(0.75, (0.5, 1.0)) == 1.0
    rng = random.Random(0)
    n = sum(1 for _ in range(4000) if pseudo_harmonic(0.75, (0.5, 1.0), rng) == 0.5)
    assert abs(n / 4000 - 0.4286) < 0.03


def test_grid_lists_legal_distinct_actions():
    grid = BetGrid()
    h = HandState([10_000] * 6, button=0)
    obs = h.observe()  # UTG preflop, pot 150, to_call 100
    names = grid.abstract_actions(obs)
    assert names == ["f", "c", "r0.5", "r1", "a"]
    assert grid.to_concrete(obs, "r0.5").amount == 225  # 100 + 0.5*(150+100)
    assert grid.to_concrete(obs, "r1").amount == 350
    assert grid.to_concrete(obs, "a").amount == 10_000
    # short stack: fractions collapse into all-in
    h2 = HandState([10_000, 10_000, 10_000, 250, 10_000, 10_000], button=0)
    names2 = grid.abstract_actions(h2.observe())
    assert names2 == ["f", "c", "r0.5", "a"]  # r1 would be the whole 250 stack -> it is the all-in


def test_grid_bb_option_has_no_fold():
    grid = BetGrid()
    h = HandState([10_000] * 6, button=0)
    for _ in range(3):
        h.apply(FOLD)
    h.apply(CALL)
    h.apply(CALL)
    obs = h.observe()
    assert grid.abstract_actions(obs)[0] == "c"
    assert grid.to_concrete(obs, "f") == CALL  # fold impossible -> check


def test_history_string_translates_sizes_and_all_in():
    grid = BetGrid()
    h = HandState([10_000] * 6, button=0)
    h.apply(raise_to(300))  # UTG 3bb: increment 200 / pot-after-call 250 = 0.8 -> deterministic r1
    h.apply(FOLD)
    h.apply(raise_to(10_000))  # CO shoves
    hist = history_string(h.events, grid)
    assert hist == "r1 f a"
    h.apply(FOLD)
    h.apply(FOLD)
    h.apply(FOLD)
    h.apply(CALL)  # UTG calls all-in -> hand over
    assert h.is_terminal


def test_infoset_key_shape(bucketer):
    grid = BetGrid()
    h = HandState([10_000] * 6, button=0, seed=7)
    h.apply(raise_to(250))
    h.apply(FOLD)
    h.apply(CALL)
    h.apply(FOLD)
    h.apply(FOLD)
    h.apply(CALL)
    h.apply(CALL)
    obs = h.observe()
    key = infoset_key(obs, bucketer, grid)
    street, pos, n_active, bucket, hist = key.split("|")
    assert street == "F" and pos == "UTG" and n_active == "3"
    assert bucket.startswith("b") and 0 <= int(bucket[1:]) < 5
    assert hist == "r0.5 f c f f c/c"


def _raise_event(street, pot, call, inc, stack_before):
    """An Event for a non-all-in raise: ``inc`` chips over the call, actor had ``stack_before``."""
    from negpluribus.engine import Action, ActionType, Event

    paid = call + inc
    return Event(street, 0, Action(ActionType.RAISE, 0), call, pot, call > 0, 0, paid,
                 all_in=False, stack_after=stack_before - paid)


def test_translation_uses_all_in_as_upper_neighbour():
    """Ganzfried & Sandholm (IJCAI 2013): a size above the largest grid size is translated between
    that size and the actor's all-in, with the pseudo-harmonic probability; numbers from the paper's
    formula f(x) = (B - x)(1 + A) / ((B - A)(1 + x))."""
    grid = BetGrid(preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    # 60bb into a 2bb flop pot, 99bb behind: x = 30, all-in = 49.5 pot -> P(pot) = 0.026
    ev = _raise_event(Street.FLOP, 200, 0, 6000, 9900)
    assert grid.from_concrete(ev, False) == "a"
    rng = random.Random(3)
    share = sum(grid.from_concrete(ev, False, rng) == "a" for _ in range(20000)) / 20000
    assert abs(share - (1 - (49.5 - 30) * 2 / ((49.5 - 1) * 31))) < 0.01
    # 0.9 pot into a big pot with deep stacks: between the grid sizes 0.5 and 1, never the all-in
    ev = _raise_event(Street.TURN, 4000, 0, 3600, 8000)
    assert grid.from_concrete(ev, False) == "r1"
    assert {grid.from_concrete(ev, False, random.Random(i)) for i in range(300)} == {"r0.5", "r1"}
    # preflop open to 3.5bb: between pot (1.0) and all-in (49.5): P(pot) = 0.884
    ev = _raise_event(Street.PREFLOP, 150, 50, 250, 9950)
    rng = random.Random(4)
    share = sum(grid.from_concrete(ev, False, rng) == "r1" for _ in range(20000)) / 20000
    assert abs(share - (49.5 - 1.25) * 2 / ((49.5 - 1) * 2.25)) < 0.01


def test_translation_drops_grid_sizes_at_or_above_the_all_in():
    """Short stack: the all-in is 0.8 pot, so 'r1' does not exist in this spot; a 0.7-pot raise sits
    between 0.5 and the all-in (P(0.5) = 0.29 -> deterministic 'a')."""
    grid = BetGrid(preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    ev = _raise_event(Street.FLOP, 1000, 0, 700, 800)
    assert grid.from_concrete(ev, False) == "a"
    ev = _raise_event(Street.FLOP, 1000, 0, 400, 800)  # 0.4 pot: below the smallest size
    assert grid.from_concrete(ev, False) == "r0.5"


def test_translation_keeps_on_grid_sizes_and_old_records():
    grid = BetGrid()
    h = HandState([10_000] * 2, button=0)
    h.apply(CALL)
    h.apply(CALL)
    obs = h.observe()
    for name in ("r0.5", "r1"):
        h2 = HandState([10_000] * 2, button=0)
        h2.apply(CALL)
        h2.apply(CALL)
        h2.apply(grid.to_concrete(h2.observe(), name))
        ev = h2.events[-1]
        assert grid.from_concrete(ev, ev.all_in) == name
        assert all(grid.from_concrete(ev, ev.all_in, random.Random(i)) == name for i in range(50))
    # records without stack information keep the old nearest-size behaviour
    old = _raise_event(Street.FLOP, 200, 0, 6000, 9900)
    from dataclasses import replace

    old = replace(old, stack_after=-1)
    assert grid.from_concrete(old, False) == "r1"


def test_grid_random_agent_plays_only_grid_sizes():
    from negpluribus.agents import GridRandomAgent

    grid = BetGrid(preflop_fracs=(1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3)
    agents = [GridRandomAgent(grid, seed=1), GridRandomAgent(grid, seed=2)]
    from negpluribus.table import play_hand

    raises = same = 0
    for i in range(150):
        rec = play_hand(agents, [10_000, 10_000], i % 2, seed=i)
        for ev in rec.events:
            if ev.action.type.name == "RAISE" and not ev.all_in:
                raises += 1
                x = grid.observed_frac(ev)
                same += any(abs(x - f) < 0.05 for f in grid.fracs_for(ev.street)) or x < min(grid.fracs_for(ev.street))
    assert raises > 50 and same / raises > 0.9
