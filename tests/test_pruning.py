"""Regret-based pruning of the C++ MCCFR trainer (csrc/mccfr.h, Trainer::prune_below).

Off by default and then invisible (bit-identical to a trainer that never heard of it); on, it
skips actions and still trains a sane strategy; never on the last betting street unless asked.
Its effect on quality is measured by scripts/prune_experiment.py, not here.
"""
from __future__ import annotations

import pytest

from negpluribus import fast
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core.Trainer, "set_pruning"), reason="C++ core without pruning (rebuild)")


def _spec():
    return GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))


def _table(tr):
    return {k: (n.regret, n.strategy_sum, n.visits) for k, n in tr.nodes.items()}


def test_off_changes_nothing():
    a = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=1).train(3000)
    b = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=1).set_pruning(0.0).train(3000)
    assert _table(a) == _table(b) and b.pruned_actions == 0


def test_last_street_is_never_pruned_unless_asked():
    tr = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=1).set_pruning(1.0, prob=1.0, after=100).train(5000)
    assert tr.pruned_actions == 0  # preflop is the last street of this game


def test_pruning_skips_actions_and_trains():
    tr = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=2).set_pruning(1e5, prob=0.95, after=500, last_street=True).train(20000)
    assert tr.pruned_actions > 0
    s = tr.strategy()
    assert len(s) == tr.n_nodes and all(abs(sum(p) - 1) < 1e-9 for _, p in s.table.values())


def test_linear_until_off_changes_nothing_and_on_changes_weights():
    a = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=1).train(3000)
    b = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=1).set_linear_until(0).train(3000)
    c = MCCFRTrainer(_spec(), seed=3, backend="cpp", threads=1).set_linear_until(1000).train(3000)
    assert _table(a) == _table(b)
    assert _table(a) != _table(c)
