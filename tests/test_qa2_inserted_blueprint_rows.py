"""QA-2: the blueprint's rows at a path node with an inserted (off-grid) action (csrc/search.h rollout_probs).

The inserted action has no name in the blueprint: its column maps to -1 (probability 0 from the blueprint, the
row renormalised over the grid's actions), as profile_rows does.  Before the fix the id INSERTED_ID (255)
indexed grid_to_bp_ (one entry per grid name) past its end.  The read itself shows only under a checking
build (-D_GLIBCXX_ASSERTIONS aborts on it); here: the paths that read these rows, the blueprint's
exploitability of turn roots with an inserted raise of either seat in the round, give finite, consistent
and thread-independent numbers.
"""
from __future__ import annotations

import math

import pytest

from negpluribus.engine import Street, raise_to

from test_search_core import _act, bucketer, line_hand, make_search, trained  # noqa: F401  (module fixtures)


def _turn_with_inserted_raise(spec, first_inserted: bool):
    """A turn root: the first actor raises 0.7 pot (between the grid's 0.5 and 1) or bets the grid's 0.5 and the
    other raises off the grid; the next player to act decides facing it."""
    st, acts = line_hand(spec, ["r1", "c", "c", "c"])
    assert st.street == Street.TURN
    grid = spec.grid
    if first_inserted:
        obs = st.observe(st.current_player)
        off = raise_to(int(obs.pot * 0.7))
    else:
        _act(st, acts, grid.to_concrete(st.observe(st.current_player), "r0.5"))
        obs = st.observe(st.current_player)
        off = raise_to(min(obs.max_raise_to - 1, obs.min_raise_to + 37))
    assert off.amount not in {grid.to_concrete(obs, n).amount for n in grid.abstract_actions(obs)}
    _act(st, acts, off)
    assert st.street == Street.TURN and not st.is_terminal
    return st, acts


@pytest.mark.parametrize("first_inserted", [True, False])
def test_blueprint_rows_at_an_inserted_node(trained, first_inserted):  # noqa: F811
    spec, _, game, _ = trained[2]
    st, acts = _turn_with_inserted_raise(spec, first_inserted)
    runs = []
    for threads in (1, 4):
        s = make_search(game, st, acts, iterations=2_000, time_budget=0.0, threads=threads)
        assert any(p["inserted"] for p in s.path())
        full = list(s.subgame_exploitability(2, 2))
        river = list(s.subgame_exploitability(2, 3))
        assert all(math.isfinite(v) for v in full + river), (full, river)
        assert abs(full[3] + full[4]) < 1e-9 and full[3] == river[3], full  # the profile's values sum to zero
        assert min(full[1], full[2], river[1], river[2]) > -1e-9, (full, river)  # best responses gain >= 0
        assert river[1] <= full[1] + 1e-9 and river[2] <= full[2] + 1e-9, (full, river)
        runs.append((full, river))
    assert runs[0] == runs[1]  # summed in card order, whatever the thread count
