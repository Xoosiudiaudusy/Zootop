"""The subgame search when memory runs out: a node table that cannot grow (std::bad_alloc) must end solve() with a
Python exception (the search agent then plays the blueprint), never abort the process or hang.

Before the fix a failed growth could escape from TableGroup::leave() / end() outside any handler (std::terminate,
Windows 0xC0000409) or leave the other workers parked for ever.  _debug_fail_table_growth(n) makes the n-th growth
from now fail (n < 0: every one), as an exhausted commit would.  Each scenario runs in a child process, so an abort
fails the test instead of killing pytest.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "_debug_fail_table_growth"),
                                reason="C++ core not built or without the test hook")
ROOT = os.path.join(os.path.dirname(__file__), "..")

SCRIPT = textwrap.dedent('''
    import os, random, sys
    sys.path.insert(0, {root!r})
    from negpluribus import fast
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict
    core = fast.core()
    p = os.path.join({root!r}, "data", "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    t = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=2).train(2000)
    game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk), t.blueprint().lookup)
    rng = random.Random(3); order = list(range(52)); rng.shuffle(order)
    st = spec.new_hand(order, button=0); acts = []
    for name in ["r1", "c"]:
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        acts.append((int(a.type), int(a.amount))); st.apply(a)
    obs = st.observe(st.current_player)
    def search(th):
        return core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                  iterations=100000, time_budget=0.0, threads=th, seed=1, depth="end")
    for th in (1, 2, 4, 8):
        core._debug_fail_table_growth({fail})
        try:
            search(th).solve()
            print("solved", th, flush=True)
        except Exception as e:
            print("raised", th, type(e).__name__, str(e), flush=True)
    core._debug_fail_table_growth(0)
    r = search(4).solve()
    print("after", r["table_size"] > 0, flush=True)
''')


@pytest.mark.parametrize("fail", [1, -1])
def test_a_table_that_cannot_grow_raises_instead_of_aborting(fail):
    """fail = 1: the first growth fails once (a retry may succeed); -1: every growth fails (memory exhausted)."""
    code = SCRIPT.format(root=os.path.abspath(ROOT), fail=fail)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    lines = p.stdout.splitlines()
    assert lines[-1] == "after True", p.stdout
    outcomes = [ln for ln in lines if ln.startswith(("solved", "raised"))]
    assert len(outcomes) == 4, p.stdout
    if fail < 0:  # nothing can grow: every thread count raises a Python error that names the cause
        for ln in outcomes:
            assert ln.startswith("raised") and "RuntimeError" in ln and "memory" in ln, p.stdout
