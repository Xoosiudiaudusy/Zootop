"""The subgame search's classes on the root's round (SubgameSearch.class_of: "the combo's lossless class on the root's
round (its infoset there)", csrc/search.h compute_classes).

Two holes may share a class only if a suit relabelling maps one onto the other AND keeps the public history: every
card must stay on its street.  On a turn root the flop was bet on before the turn card came, so the ranges the
players bring are not symmetric between a flop card's suit and the turn card's.

L4 (defect hunter, low; confirmed): compute_classes packs canonical_form(hole, board) of the whole sorted 4-card board,
which also merges holes under relabellings that swap a flop card with the turn card: on Ah 2c 3s | Ad, Kh Jh (a flush
draw on the flop) and Kd Jd (none on the flop) get one class, so the search gives them one strategy although their
reaches differ.  Controls: suits the board treats alike merge (legitimate), and a flop root keeps the two apart.
"""
from __future__ import annotations

import pytest

from negpluribus import fast
from negpluribus.cards import cards_from_str
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "SubgameSearch"), reason="C++ core with the search not built")


@pytest.fixture(scope="module")
def game():
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=4, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    bp = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=1).train(1500).blueprint()
    return spec, core.SearchGame(spec_to_dict(spec), core_bucketer(bk, (1 << 16,) * 3), bp.lookup)


def _search(game, board: str, street: Street):
    """Heads-up, SB opens pot, BB calls, checked to ``street``; holes 9c 8c / 7s 6s; ``board`` = flop turn river."""
    spec, g = game
    order = cards_from_str("9c 8c 7s 6s") + cards_from_str(board)
    order += [c for c in range(52) if c not in set(order)]
    st = spec.new_hand(order, button=0)
    acts = []
    for name in ["r1", "c"] + ["c", "c"] * (int(street) - 1):
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        acts.append((int(a.type), int(a.amount)))
        st.apply(a)
    assert st.street == street
    obs = st.observe(st.current_player)
    return core.SubgameSearch(g, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole),
                              iterations=1, time_budget=0.0, threads=1, seed=1, depth="end")


def _cls(s, hole: str) -> int:
    a, b = sorted(cards_from_str(hole))
    return s.class_of(core.combo_index(a, b))


def test_a_monotone_board_merges_the_other_three_suits(game):
    s = _search(game, "Ah 2h 3h 4h 5s", Street.TURN)  # clubs, diamonds and spades are alike on this board
    assert _cls(s, "Kc Qc") == _cls(s, "Kd Qd") == _cls(s, "Ks Qs")
    assert _cls(s, "Kh Qh") != _cls(s, "Kc Qc")
    assert _cls(s, "Ah Kc") == -1  # meets the board


def test_a_flop_root_keeps_a_flush_draw_apart_from_a_backdoor(game):
    s = _search(game, "Ah 2c 3s Ad 5c", Street.FLOP)
    assert _cls(s, "Kh Jh") != _cls(s, "Kd Jd")


@pytest.mark.xfail(strict=True, reason="L4: csrc/search.h compute_classes canonicalises hole + the whole sorted board, so a "
                                        "suit swap of a flop card with the turn card merges holes with different flop histories")
def test_a_turn_root_does_not_merge_across_a_flop_turn_suit_swap(game):
    s = _search(game, "Ah 2c 3s Ad 5c", Street.TURN)
    assert _cls(s, "Kh Jh") != _cls(s, "Kd Jd")
