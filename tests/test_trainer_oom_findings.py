"""Regression tests of the bug-hunter findings M2 / M5 (trainer OOM); the defects were fixed by
opt/trainer-oom 8cd9af6 (merged 29.09), so these run as ordinary tests now.

The MCCFR trainer when its node table cannot grow (std::bad_alloc: memory exhausted; the user's rule of 28.09: no
page file, running out of memory must become a Python exception, never an abort).  tests/test_search_oom.py covers
the search; nothing covered the trainer.  Each scenario runs in a child process with a timeout, so an abort or a hang
fails the test instead of pytest; _debug_fail_table_growth(n) makes the n-th growth from now fail (n < 0: every one).

  * M2 (defect hunter; confirmed): one thread.  A failed growth leaves grow_requested_ set (csrc/nodetable.h grow()
    throws before the rehash resets it), so the next train() after the failure spins for ever in get_or_create.
  * M5 (defect hunter; confirmed): two threads.  The worker lambdas of Trainer::train / train_batched (csrc/mccfr.h)
    have no handler: an exception in a pool thread is std::terminate (Windows 0xC0000409), and one on the calling
    thread leaves joinable std::threads behind (terminate again).

Both are being fixed on the unmerged opt/trainer-oom branch (handoff 28.09 23:50); these tests should turn green with
it.  The hanging child of M2 spins one core until its 20 s timeout.
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
                                reason="C++ core without the growth-failure hook")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMEOUT = 20

SCRIPT = textwrap.dedent('''
    import sys
    sys.path.insert(0, {root!r})
    from negpluribus import fast
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street
    core = fast.core()
    spec = GameSpec(n_players=2, stack_bb=50, max_street=Street.PREFLOP, preflop_fracs=(0.5, 1.0, 3.0), max_raises_per_street=3)
    t = MCCFRTrainer(spec, seed=1, backend="cpp", threads={threads})
    if {batch}:
        t.set_batch({batch})
    core._debug_fail_table_growth({fail})
    try:
        t.train(20000)
        print("first finished", flush=True)
    except Exception as e:
        print("first raised", type(e).__name__, str(e)[:80], flush=True)
    core._debug_fail_table_growth(0)
    t.train(20000)
    print("second finished", len(t.nodes) > 0, flush=True)
''')


def _run(threads, fail, batch=0):
    code = SCRIPT.format(root=ROOT, threads=threads, fail=fail, batch=batch)
    try:
        p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, out
    return p.returncode, p.stdout + p.stderr[-1500:]


def test_one_thread_a_failed_growth_raises_and_the_trainer_goes_on():
    rc, out = _run(threads=1, fail=1)
    assert rc is not None, f"hung (> {TIMEOUT} s): {out[-800:]}"
    assert rc == 0 and "first raised" in out and "second finished True" in out, (rc, out[-800:])


@pytest.mark.parametrize("batch", [0, 64])
def test_two_threads_a_failed_growth_is_a_python_exception_not_an_abort(batch):
    rc, out = _run(threads=2, fail=-1, batch=batch)
    assert rc is not None, f"hung (> {TIMEOUT} s): {out[-800:]}"
    assert rc == 0, f"exit code {rc:#x}: {out[-800:]}"
    assert "first raised RuntimeError" in out or "first raised MemoryError" in out, out[-800:]
