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
