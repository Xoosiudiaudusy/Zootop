"""Exact potential-aware features (csrc/exactfeat.h): the per-board batch gives the definition's numbers."""
from __future__ import annotations

import random

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "exact_feature"), reason="C++ core without exact features (rebuild)")


@pytest.mark.parametrize("n_board,n_holes", [(4, 30), (3, 4)])
def test_batch_equals_definition(n_board, n_holes):
    rng = random.Random(n_board)
    board = rng.sample(range(52), n_board)
    batch = core.exact_feature_batch(board, 10)
    assert len(batch) == (52 - n_board) * (51 - n_board) // 2
    for a, b in rng.sample(sorted(batch), n_holes):
        counts, mean = core.exact_feature([a, b], board, 10)
        got_counts, got_mean = batch[(a, b)]
        assert list(got_counts) == list(counts) and got_mean == mean
        assert sum(counts) == (46 if n_board == 4 else 47)  # one value per next card


def test_river_values_are_the_exact_equity():
    # a turn feature is the histogram of river_equity_exact over the 46 rivers
    hole, board = [0, 5], [10, 20, 30, 40]
    counts, mean = core.exact_feature(hole, board, 10)
    es = [core.river_equity_exact(hole, board + [c]) for c in range(52) if c not in hole + board]
    want = [0] * 10
    for e in es:
        want[min(int(e * 10), 9)] += 1
    total = 0.0
    for e in es:  # plain left-to-right sum, as the C++ does (Python's sum() is compensated since 3.12)
        total += e
    assert counts == want and mean == total / len(es)
