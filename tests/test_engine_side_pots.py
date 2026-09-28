"""Side pots with three unequal stacks and real odd chips, against hand-computed nets (negpluribus/engine.py
HandState._finish), and the C++ engine (csrc/engine.h, core.Hand) on the same hands.

test_engine.py covers two stack sizes (a short stack against two equal ones) and a conservation fuzz; its odd-chip
test never produces an odd chip (heads-up it cannot: two eligible players always split an even layer).  Here: three
all-in levels (30bb / 15bb / 5bb) with every order of hand strength, an uncalled top layer, a chopped side pot, a
folded raiser's dead money, and odd chips of a split going to the first winners left of the button.
"""
from __future__ import annotations

import itertools

import pytest

from negpluribus import fast
from negpluribus.cards import Deck, cards_from_str
from negpluribus.engine import CALL, FOLD, HandState, Street, raise_to

core = fast.core()

HANDS = {"AA": "Ah Ad", "KK": "Kh Kd", "QQ": "Qh Qd"}
DRY_BOARD = "2c 7d 9h Js 3s"  # no straight, no flush for any of the three pairs


def _order(holes, board):
    """A deck order dealing ``holes`` in seat order, then the board (the engine burns no cards)."""
    order = [c for h in holes for c in cards_from_str(h)] + cards_from_str(board)
    assert len(set(order)) == len(order)
    return order + [c for c in range(52) if c not in set(order)]


def _play(stacks, button, holes, board, actions, sb=50, bb=100, ante=0):
    """The hand on the Python engine and, when the core is built, on the C++ engine: the same nets, winners and
    showdown seats."""
    order = _order(holes, board)
    h = HandState(list(stacks), button, sb, bb, ante, deck=Deck.from_order(order))
    c = core.Hand(list(stacks), button, sb, bb, ante, order, int(Street.RIVER)) if core is not None else None
    for a in actions:
        h.apply(a)
        if c is not None:
            c.apply(int(a.type), a.amount)
    assert h.is_terminal
    rec = h.record()
    if c is not None:
        assert c.is_terminal
        cr = c.record()
        assert (cr["net"], cr["winners"], cr["showdown_seats"]) == (rec.net, rec.winners, rec.showdown_seats)
    assert sum(rec.net) == 0
    return rec


# 3-handed, button 0: seat 0 BTN (30bb) acts first and shoves, seat 1 SB (15bb) and seat 2 BB (5bb) call all-in.
# Invested 3000 / 1500 / 500: main pot 3 x 500 = 1500, side pot 2 x 1000 = 2000 (seats 0, 1), the BTN's last 1500
# uncalled.
STACKS = [3000, 1500, 500]
THREE_LEVELS = [raise_to(3000), CALL, CALL]


def _expected_nets(strength):
    """``strength[seat]``: higher wins."""
    won = [0, 0, 0]
    won[max(range(3), key=lambda s: strength[s])] += 1500      # main pot: everybody
    won[max((0, 1), key=lambda s: strength[s])] += 2000        # side pot: the two deeper stacks
    won[0] += 1500                                              # uncalled
    return [w - i for w, i in zip(won, STACKS)]


@pytest.mark.parametrize("perm", list(itertools.permutations(["AA", "KK", "QQ"])))
def test_three_unequal_all_ins_every_order_of_strength(perm):
    rank = {"AA": 3, "KK": 2, "QQ": 1}
    rec = _play(STACKS, 0, [HANDS[p] for p in perm], DRY_BOARD, THREE_LEVELS)
    assert rec.net == _expected_nets([rank[p] for p in perm]), perm
    assert sorted(rec.showdown_seats) == [0, 1, 2]
    assert rec.net[0] >= -1500  # the deep stack always gets its uncalled chips back


def test_three_unequal_all_ins_chopped_side_pot():
    # seats 0 and 1 hold A-K (A K J 9 7 each), the short seat 2 QQ: main pot 1500 to QQ, side pot split 1000 / 1000
    rec = _play(STACKS, 0, ["Ac Kd", "As Kc", "Qh Qd"], DRY_BOARD, THREE_LEVELS)
    assert rec.net == [1000 + 1500 - 3000, 1000 - 1500, 1500 - 500]
    assert rec.winners == [0, 1, 2]


def test_a_folded_raisers_dead_money_goes_to_the_only_live_player_of_its_layer():
    """Stacks 5bb / 30bb / 30bb, button 0: the BTN limps, the SB raises to 1000, the BB shoves 3000, the BTN calls
    all-in (500), the SB folds.  Invested 500 / 1000 / 3000: layer 0..500 = 1500 (BTN and BB eligible), layer
    500..1000 = 1000 (the SB's dead 500 plus the BB's: only the BB is live there), layer 1000..3000 back to the BB."""
    actions = [CALL, raise_to(1000), raise_to(3000), CALL, FOLD]
    rec = _play([500, 3000, 3000], 0, ["Ah Ad", "Kh Kd", "Qh Qd"], DRY_BOARD, actions)
    assert rec.net == [1500 - 500, -1000, 1000 + 2000 - 3000]
    assert rec.showdown_seats == [0, 2]
    # the same hand, the BB's QQ now best: it takes every layer
    rec = _play([500, 3000, 3000], 0, ["Qh Qd", "Kh Kd", "Ah Ad"], DRY_BOARD, actions)
    assert rec.net == [-500, -1000, 1500]


def test_odd_chip_of_a_split_goes_to_the_first_winner_left_of_the_button():
    """sb 51 / bb 101, 3-handed, button 0: the BTN limps 101, the SB folds its 51, the BB checks; BTN and BB split
    (A-K each).  Layer 0..51: 3 x 51 = 153 -> 77 to the BB (first winner left of the button) and 76 to the BTN;
    layer 51..101: 2 x 50 -> 50 / 50."""
    actions = [CALL, FOLD, CALL] + [CALL] * 6  # limp, fold, check; then checked down (BB first, then BTN)
    rec = _play([10_000] * 3, 0, ["Ah Kd", "2c 3d", "As Kc"], "7s 8h 9c Tc 4d", actions, sb=51, bb=101)
    assert rec.showdown_seats == [0, 2] and rec.winners == [0, 2]
    assert rec.net == [76 + 50 - 101, -51, 77 + 50 - 101]


def test_two_odd_chips_of_a_three_way_split_follow_the_seat_order_left_of_the_button():
    """4-handed, button 0: UTG (seat 3) and the BTN limp, the SB (seat 1) folds its 50, the BB checks; the three live
    players play the board's straight.  Layer 0..50: 4 x 50 = 200 = 3 x 66 + 2 -> one extra chip each to the BB
    (seat 2) and UTG (seat 3), the first winners left of the button; layer 50..100: 3 x 50 -> 50 each."""
    actions = [CALL, CALL, FOLD, CALL] + [CALL] * 9  # UTG limp, BTN limp, SB fold, BB check; 3 checks x 3 streets
    rec = _play([10_000] * 4, 0, ["2h 3d", "4c 4d", "2c 3h", "2s 3s"], "7s 8h 9c Tc Jd", actions)
    assert rec.winners == [0, 2, 3]
    assert rec.net == [66 + 50 - 100, -50, 67 + 50 - 100, 67 + 50 - 100]


def test_everybody_plays_the_board_three_ways_without_a_remainder():
    actions = [CALL, CALL, CALL] + [CALL] * 9  # limp, complete, check; checked down (3 players x 3 streets)
    rec = _play([10_000] * 3, 0, ["2h 3d", "2c 3c", "2s 3h"], "7s 8h 9c Tc Jd", actions, sb=51, bb=101)
    assert rec.net == [0, 0, 0] and rec.winners == [0, 1, 2]
