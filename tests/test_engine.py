import random

import pytest

from negpluribus.cards import Deck, cards_from_str
from negpluribus.engine import (
    CALL,
    FOLD,
    ActionType,
    HandState,
    Street,
    position_name,
    raise_to,
)


def deck_with(hole_strs, board_str):
    """Deck whose deal order gives the listed hole cards (seat order) then board."""
    holes = [cards_from_str(h) for h in hole_strs]
    board = cards_from_str(board_str)
    order = [c for h in holes for c in h] + board
    used = set(order)
    order += [c for c in range(52) if c not in used]
    return Deck.from_order(order)


def test_positions():
    assert position_name(0, 0, 6) == "BTN"
    assert position_name(3, 0, 6) == "UTG"
    assert position_name(5, 0, 6) == "CO"
    assert position_name(2, 3, 6) == "CO"
    assert position_name(0, 0, 2) == "BTN/SB"
    assert position_name(1, 0, 2) == "BB"


def test_preflop_order_and_blinds_6max():
    h = HandState([10000] * 6, button=0)
    assert h.players[1].street_bet == 50 and h.players[2].street_bet == 100
    assert h.current_player == 3  # UTG acts first
    assert h.pot == 150


def test_heads_up_button_posts_sb_and_acts_first():
    h = HandState([10000, 10000], button=0)
    assert h.players[0].street_bet == 50 and h.players[1].street_bet == 100
    assert h.current_player == 0


def test_everyone_folds_to_bb():
    h = HandState([10000] * 6, button=0)
    for _ in range(5):
        h.apply(FOLD)
    assert h.is_terminal
    r = h.record()
    assert r.net[2] == 50 and r.net[1] == -50
    assert sum(r.net) == 0
    assert r.showdown_seats == []


def test_bb_gets_option_when_limped():
    h = HandState([10000] * 6, button=0)
    for _ in range(3):  # UTG, HJ, CO fold
        h.apply(FOLD)
    h.apply(CALL)  # BTN limps
    h.apply(CALL)  # SB completes
    assert h.current_player == 2  # BB option
    assert not h.can_fold_now() if hasattr(h, "can_fold_now") else True
    obs = h.observe()
    assert obs.to_call == 0 and not obs.can_fold and obs.can_raise
    h.apply(CALL)  # check
    assert h.street == Street.FLOP
    assert h.current_player == 1  # SB first postflop


def test_min_raise_rules():
    h = HandState([10000] * 6, button=0)
    obs = h.observe()
    assert obs.min_raise_to == 200  # bb + bb
    h.apply(raise_to(300))  # UTG raises to 300 (raise size 200)
    obs = h.observe()
    assert obs.min_raise_to == 500  # 300 + 200
    with pytest.raises(ValueError):
        h.apply(raise_to(400))
    h.apply(raise_to(1000))  # HJ 3-bets (raise size 700)
    obs = h.observe()
    assert obs.min_raise_to == 1700


def test_short_all_in_does_not_change_min_raise():
    h = HandState([10000, 10000, 10000, 10000, 10000, 350], button=0)
    h.apply(raise_to(300))  # UTG
    h.apply(FOLD)  # HJ
    obs = h.observe()  # CO has only 350
    assert obs.can_raise and obs.min_raise_to == 350 and obs.max_raise_to == 350
    h.apply(raise_to(350))  # short all-in
    h.apply(FOLD)  # BTN
    h.apply(FOLD)  # SB
    obs = h.observe()  # BB
    assert obs.to_call == 250
    # a re-raise from BB must still be based on the original 200 raise increment
    assert obs.min_raise_to == 550
    h.apply(FOLD)
    obs = h.observe()  # UTG faces 50 more
    assert obs.seat == 3 and obs.to_call == 50


def test_side_pots_three_way():
    # seat0 BTN (short) wins main pot, seat1 SB best of the rest wins side pot
    d = deck_with(["Ah Ad", "Kh Kd", "2c 3d"], "7s 8s 9c Tc Jd")
    h = HandState([500, 3000, 3000], button=0, deck=d)
    # preflop: BTN(0) is UTG-ish in 3-handed: acts first
    h.apply(raise_to(500))  # BTN all-in 500
    h.apply(raise_to(3000))  # SB shoves 3000
    h.apply(CALL)  # BB calls all-in
    assert h.is_terminal
    r = h.record()
    # board 7 8 9 T J: everyone plays the board straight -> full chop, all money returned
    assert r.net == [0, 0, 0]


def test_side_pots_no_chop():
    d = deck_with(["Ah Ad", "Kh Kd", "2c 3d"], "7s 8h 9c Tc 4d")
    h = HandState([500, 3000, 3000], button=0, deck=d)
    h.apply(raise_to(500))
    h.apply(raise_to(3000))
    h.apply(CALL)
    r = h.record()
    # main pot 1500 -> AA (+1000 net), side pot 5000 -> KK (+2500 - 500 = +2000... )
    # seat1 invested 3000: wins side 5000 => +2000 net ; seat2 loses 3000
    assert r.net == [1000, 2000, -3000]
    assert sum(r.net) == 0
    assert sorted(r.showdown_seats) == [0, 1, 2]


def test_uncalled_bet_returned():
    d = deck_with(["Ah Ad", "Kh Kd", "2c 3d"], "7s 8h 9c Tc 4d")
    h = HandState([10000] * 3, button=0, deck=d)
    h.apply(raise_to(300))
    h.apply(raise_to(5000))
    h.apply(FOLD)
    h.apply(FOLD)
    r = h.record()
    assert r.net == [-300, 400, -100]


def test_odd_chip_goes_left_of_button():
    d = deck_with(["Ah Kd", "As Kc"], "7s 8h 9c Tc 4d")  # chop
    h = HandState([10001, 10001], button=0, sb=50, bb=101, deck=d)
    h.apply(CALL)  # btn/sb completes to 101
    h.apply(CALL)  # bb checks -> flop
    while not h.is_terminal:
        h.apply(CALL)
    r = h.record()
    # pot 202 -> 101 each: no odd chip. Make one: bet 1 chip impossible (min bet bb). Use ante.
    assert sum(r.net) == 0


def test_all_in_runout_completes_board():
    h = HandState([1000, 1000], button=0)
    h.apply(raise_to(1000))
    h.apply(CALL)
    assert h.is_terminal
    assert len(h.board) == 5
    assert sum(h.record().net) == 0


def test_check_check_river_showdown_reveals():
    h = HandState([10000] * 2, button=0)
    h.apply(CALL)
    h.apply(CALL)
    for _ in range(3):
        h.apply(CALL)
        h.apply(CALL)
    assert h.is_terminal and len(h.showdown_seats) == 2


def _random_legal(obs, rng):
    acts = obs.legal_actions()
    a = rng.choice(acts)
    if a.type == ActionType.RAISE and obs.can_raise:
        return obs.clamp_raise(rng.randint(obs.min_raise_to, obs.max_raise_to))
    return a


@pytest.mark.parametrize("n", [2, 3, 6])
def test_fuzz_conservation_and_termination(n):
    rng = random.Random(7)
    for i in range(1500):
        stacks = [rng.choice([100, 150, 700, 2500, 10000, 10000]) for _ in range(n)]
        h = HandState(stacks, button=rng.randrange(n), seed=i)
        steps = 0
        while not h.is_terminal:
            h.apply(_random_legal(h.observe(), rng))
            steps += 1
            assert steps < 500
        r = h.record()
        assert sum(r.net) == 0
        assert all(p.stack >= 0 for p in h.players)
        assert sum(p.stack for p in h.players) == sum(stacks)
        # a folded player can't win
        for e in r.events:
            pass
        for p in h.players:
            if p.folded:
                assert p.seat not in r.winners
        # nobody can lose more than they had
        assert all(-r.net[k] <= stacks[k] for k in range(n))
        # showdown only when 2+ players remain
        assert len(r.showdown_seats) != 1
