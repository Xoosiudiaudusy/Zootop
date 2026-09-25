"""``equity_vs_random`` on the C++ core, bit-identical to the reference.

The reference draws its runouts with ``rng.sample`` from a ``random.Random``.  The compiled
version takes that generator's state (``getstate()``), runs the identical Mersenne-Twister
draws in C++, and writes the advanced state back (``setstate()``), so the Python object
continues exactly where the pure-Python loop would have left it.  Every caller that passes a
seeded ``random.Random`` therefore gets the same equity to the last bit, and the bucketer's
per-canonical-form cache (seeded from ``hash(key)``) stays reproducible across backends.
"""
from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

from . import core, fast_eval_enabled


def make_fast_equity(equity_py: Callable) -> Optional[Callable]:
    if not fast_eval_enabled():
        return None
    c = core()
    if c is None:
        return None
    _eq_state = c.equity_vs_random
    _eq_seed = c.equity_vs_random_seeded

    def equity_vs_random(
        hole: Sequence[int],
        board: Sequence[int],
        n_opponents: int = 1,
        samples: int = 200,
        rng: Optional[random.Random] = None,
    ) -> float:
        if rng is None:
            return _eq_seed(hole, board, n_opponents, samples, random.getrandbits(64))
        if type(rng) is not random.Random:
            return equity_py(hole, board, n_opponents, samples, rng)
        version, state, gauss = rng.getstate()
        eq, new_state = _eq_state(hole, board, n_opponents, samples, state)
        rng.setstate((version, new_state, gauss))
        return eq

    return equity_vs_random
