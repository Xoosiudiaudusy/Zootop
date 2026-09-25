import itertools
import random

import pytest

from negpluribus.cards import cards_from_str
from negpluribus.evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    category,
    evaluate,
)


def ev(s: str) -> int:
    return evaluate(cards_from_str(s))


@pytest.mark.parametrize(
    "hand,cat",
    [
        ("As Ks Qs Js Ts 2d 3c", STRAIGHT_FLUSH),
        ("5s 4s 3s 2s As Kd Qc", STRAIGHT_FLUSH),  # steel wheel
        ("Ah Ad Ac As 2d 3c 4h", QUADS),
        ("Kh Kd Kc 2s 2d 3c 4h", FULL_HOUSE),
        ("Kh Kd Kc 2s 2d 2c 4h", FULL_HOUSE),  # two trips -> FH
        ("Ah 9h 7h 4h 2h Kd Qc", FLUSH),
        ("9h 8d 7c 6s 5h Kd Qc", STRAIGHT),
        ("Ah 2d 3c 4s 5h Kd Qc", STRAIGHT),  # wheel
        ("Ah Ad Ac 4s 5h Kd Qc", TRIPS),
        ("Ah Ad 4c 4s 5h Kd Qc", TWO_PAIR),
        ("Ah Ad 4c 6s 5h Kd Qc", PAIR),
        ("Ah 3d 4c 6s 8h Kd Qc", HIGH_CARD),
    ],
)
def test_categories(hand, cat):
    assert category(ev(hand)) == cat


def test_ordering_basics():
    assert ev("Ah Ad Ac As 2d 3c 4h") > ev("Kh Kd Kc 2s 2d 3c 4h")
    assert ev("9h 8d 7c 6s 5h Kd Qc") > ev("Ah 2d 3c 4s 5h Kd Qc")  # 9-high > wheel
    assert ev("Ah Ad 4c 4s 5h Kd Qc") > ev("Ah Ad 3c 3s 5h Kd Qc")  # two pair kicker on pair
    assert ev("Ah Ad 4c 4s Kh 2d 3c") > ev("Ah Ad 4c 4s Qh 2d 3c")  # kicker
    assert ev("Ah Kd 4c 6s 8h Jd Qc") > ev("Ah Kd 4c 6s 8h Td Qc")  # high card kickers
    assert ev("Ah 9h 7h 4h 2h Kd Qc") > ev("Kh 9h 7h 4h 2h Ad Qc")  # flush by top card
    assert ev("Ah Ad Ac Ks Kh 2d 3c") > ev("Ah Ad Ac Qs Qh Kd 3c")  # FH pair rank
    # flush uses best 5 of 6 suited cards
    assert ev("Ah Kh 9h 7h 4h 2h 3c") == ev("Ah Kh 9h 7h 4h 2d 3c")


def test_seven_card_equals_best_five_combo():
    """Cross-check the 7-card shortcut logic against brute force over 21 five-card subsets."""
    rng = random.Random(1234)
    for _ in range(3000):
        cards = rng.sample(range(52), 7)
        best5 = max(evaluate(list(c)) for c in itertools.combinations(cards, 5))
        assert evaluate(cards) == best5, cards


def test_bad_lengths():
    with pytest.raises(ValueError):
        evaluate([0, 1, 2, 3])
