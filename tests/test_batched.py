"""Batched synchronous MCCFR (Trainer::batch_size, the CPU reference of the GPU trainer) and Philox4x32-10."""
from __future__ import annotations

import pytest

from negpluribus import fast
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "philox4x32_10"), reason="C++ core without batched mode (rebuild)")


def _spec():
    return GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))


def _table(tr):
    return {k: (n.regret, n.strategy_sum, n.visits) for k, n in tr.nodes.items()}


def test_philox_known_answers():
    # Random123 known-answer vectors for philox4x32_10
    assert core.philox4x32_10([0, 0, 0, 0], [0, 0]) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    assert core.philox4x32_10([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2) == [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]
    assert core.philox4x32_10([0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344], [0xA4093822, 0x299F31D0]) == \
        [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]


def test_deal_is_a_permutation_and_a_function_of_seed_and_iteration():
    d = core.philox_deal(7, 12)
    assert sorted(d) == list(range(52)) and d == core.philox_deal(7, 12) and d != core.philox_deal(7, 13)


def test_same_result_for_any_thread_count_and_aligned_splits():
    a = MCCFRTrainer(_spec(), seed=4, backend="cpp", threads=1).set_batch(256).train(3000)
    b = MCCFRTrainer(_spec(), seed=4, backend="cpp", threads=4).set_batch(256).train(3000)
    c = MCCFRTrainer(_spec(), seed=4, backend="cpp", threads=3).set_batch(256).train(1024).train(1976)
    assert _table(a) == _table(b) == _table(c)


def test_off_is_the_sequential_trainer():
    a = MCCFRTrainer(_spec(), seed=4, backend="cpp", threads=1).train(2000)
    b = MCCFRTrainer(_spec(), seed=4, backend="cpp", threads=1).set_batch(0).train(2000)
    assert _table(a) == _table(b)
