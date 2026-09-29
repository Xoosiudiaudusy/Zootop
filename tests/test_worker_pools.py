"""Thread pools outside the trainers (workers.h run_pool): a worker that fails, or a thread that cannot start, ends the
call with a Python exception once every started thread has joined -- never an abort (std::terminate; 0xC0000409 on
Windows).  Hooks: _debug_fail_worker(n) (the n-th worker from now throws std::bad_alloc before its work; < 0 every one)
and _debug_fail_thread_start(n) (the n-th thread start fails).  Each scenario runs in a child process, so an abort
fails the test instead of killing pytest; after the hooks are cleared the same call works in the same process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "_debug_fail_worker"), reason="C++ core without the pool test hooks")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HEAD = textwrap.dedent('''
    import os, sys, tempfile
    sys.path.insert(0, {root!r})
    from negpluribus import fast
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.fast.trainer import core_bucketer
    core = fast.core()
    def bucketer():
        p = os.path.join({root!r}, "data", "buckets_3p_15bb_flop.json")
        bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
        return core_bucketer(bk, cache_caps=(0, 0, 0))
    def check(name, call, start=False):
        if start:
            core._debug_fail_thread_start(1)  # the first thread start fails: the call raises that error
        else:
            core._debug_fail_worker(-1)  # every worker fails at once (the expensive calls end quickly)
        try:
            call()
            print(name, "no error")
        except RuntimeError as e:
            print(name, "raised", ("starting the worker threads" if start else "out of memory") in str(e))
        core._debug_fail_worker(0)
        core._debug_fail_thread_start(0)
''')

CALLS = {
    "equity_vs_hand": "lambda: core.equity_vs_hand([0, 13], [26, 39], [], 4)",
    "exact_feature_many": "lambda: core.exact_feature_many([[0, 13, 2, 7, 30], [5, 9, 2, 7, 30]] * 8, 3, 8, 4)",
    "table_stress": "lambda: core._table_stress(4, 2000)",
    "precompute": "lambda: bucketer().precompute(1, 4)",
    "tables_flop": "lambda: core.BucketTables().build(bucketer(), 1, 4)",
    "tables_river": "lambda: core.BucketTables().build(bucketer(), 3, 4)",
    "aivat_river": "lambda: core.aivat_build_tables(bucketer(), 4, [3])",
    "aivat_flop": "lambda: core.aivat_build_tables(bucketer(), 4, [1])",
    "exact_features": "lambda: core.build_exact_features(3, 8, 4, os.path.join(tempfile.mkdtemp(), 'f.npxf'))",
}
CHEAP = {  # run again without the hooks: the process goes on and the call works
    "equity_vs_hand": "core.equity_vs_hand([0, 13], [26, 39], [], 4)",
    "exact_feature_many": "core.exact_feature_many([[0, 13, 2, 7, 30]] * 8, 3, 8, 4)",
    "table_stress": "core._table_stress(4, 2000)",
}


def run(body: str) -> str:
    code = HEAD.format(root=ROOT) + textwrap.dedent(body)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.returncode, p.stdout[-3000:], p.stderr[-3000:])
    return p.stdout


@pytest.mark.parametrize("name", list(CALLS))
def test_failing_worker_raises(name):
    out = run(f"check({name!r}, {CALLS[name]})\n" + (f"{CHEAP[name]}\nprint('after ok')\n" if name in CHEAP else ""))
    assert f"{name} raised True" in out, out
    if name in CHEAP:
        assert "after ok" in out, out


@pytest.mark.parametrize("name", list(CHEAP))
def test_thread_that_cannot_start_raises(name):
    out = run(f"check({name!r}, {CALLS[name]}, start=True)\n{CHEAP[name]}\nprint('after ok')\n")
    assert f"{name} raised True" in out and "after ok" in out, out


def test_trainer_thread_that_cannot_start_raises():
    out = run('''
        from negpluribus.cfr.game import GameSpec
        from negpluribus.cfr.mccfr import MCCFRTrainer
        from negpluribus.engine import Street
        bk = EquityBucketer(n_buckets=8, samples=40).fit(n_situations=60, seed=0)
        spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, n_buckets=8)
        t = MCCFRTrainer(spec, bk, seed=1, backend="cpp", threads=4)
        core._debug_fail_thread_start(1)
        try:
            t.train(2000)
            print("no error")
        except RuntimeError as e:
            print("raised", "starting the worker threads" in str(e))
        core._debug_fail_thread_start(0)
        MCCFRTrainer(spec, bk, seed=1, backend="cpp", threads=4).train(500)
        print("after ok")
    ''')
    assert "raised True" in out and "after ok" in out, out


def test_search_pools_raise():
    """The subgame search's own pools: the ranges and the river table (built when the search is made, on a flop
    root), the exploitability's river cards; each raises and the process goes on."""
    out = run('''
        import random
        from negpluribus.cfr.game import GameSpec
        from negpluribus.cfr.mccfr import MCCFRTrainer
        from negpluribus.engine import Street
        from negpluribus.fast.trainer import spec_to_dict
        p = os.path.join(sys.path[0], "data", "buckets_3p_15bb_flop.json")
        bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
        spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                        preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
        t = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=1).train(2000)
        game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk), t.blueprint().lookup)
        rng = random.Random(3); order = list(range(52)); rng.shuffle(order)
        def root(line):
            st = spec.new_hand(order, button=0); acts = []
            for name in line:
                a = spec.grid.to_concrete(st.observe(st.current_player), name)
                acts.append((int(a.type), int(a.amount))); st.apply(a)
            return st, acts
        def search(line):
            st, acts = root(line)
            obs = st.observe(st.current_player)
            return core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                      iterations=300, time_budget=0.0, threads=4, seed=1, depth="end")
        check("search_setup", lambda: search(["r1", "c"]))  # every worker fails: the ranges' pool, the first one
        core._debug_fail_worker(5)  # the ranges take 4 workers (4 threads): the 5th is the river table's first
        try:
            search(["r1", "c"])
            print("river_table no error")
        except RuntimeError as e:
            print("river_table raised", "river table" in str(e), "out of memory" in str(e))
        core._debug_fail_worker(0)
        s = search(["r1", "c", "c", "c"])  # a turn root
        s.solve()
        check("exploitability", lambda: s.subgame_exploitability(0, 2))
        s = search(["r1", "c", "c", "c"])
        s.solve()
        search(["r1", "c"])  # the flop root's setup works again
        print("after ok", s.subgame_exploitability(0, 2)[0] >= 0)
    ''')
    assert "search_setup raised True" in out and "river_table raised True True" in out, out
    assert "exploitability raised True" in out and "after ok True" in out, out


def test_the_other_workers_stop_at_an_error():
    """One worker fails: the others stop at their next chunk (run_pool's stop flag) instead of doing the whole work
    that is then thrown away."""
    out = run('''
        import time
        bk = bucketer()
        t = time.perf_counter(); core.BucketTables().build(bk, 1, 4); normal = time.perf_counter() - t
        core._debug_fail_worker(2)
        t = time.perf_counter()
        try:
            core.BucketTables().build(bk, 1, 4)
            print("no error")
        except RuntimeError as e:
            print("raised", "out of memory" in str(e))
        failed = time.perf_counter() - t
        core._debug_fail_worker(0)
        print("fast", failed < 0.5 * normal, round(normal, 2), round(failed, 2))
    ''')
    assert "raised True" in out and "fast True" in out, out


def test_progress_callback_that_raises():
    """BucketTables.build's progress callback raising: the build is waited for, then the callback's error is raised
    (it used to leave with the build thread joinable: std::terminate)."""
    out = run('''
        def progress(done, total):
            raise ValueError("stop from the callback")
        try:
            core.BucketTables().build(bucketer(), 1, 2, progress, 0.05)
            print("no error")
        except ValueError as e:
            print("raised", "stop from the callback" in str(e))
        print("after ok")
    ''')
    assert "raised True" in out and "after ok" in out, out
