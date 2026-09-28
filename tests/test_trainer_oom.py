"""The trainers when memory runs out: an allocation that fails in a worker thread must end train() with a Python
exception, never abort the process (Windows 0xC0000409) or hang; in the batched and GPU modes the tables must stay
those of the last whole batch, so that training resumes exactly (from the object, or from a checkpoint).

Hooks: _debug_fail_table_growth(n) (the node tables of the CPU trainer and RNR) and _debug_fail_prepare(n) (the flat
/ GPU trainer's preparation of an iteration): the n-th from now fails with std::bad_alloc (n < 0: every one).  Each
scenario runs in a child process, so an abort fails the test instead of killing pytest.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "_debug_fail_prepare"),
                                reason="C++ core not built or without the test hooks")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HEAD = textwrap.dedent('''
    import os, sys
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
    spec = GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
''')


def run(body: str, **fmt) -> str:
    code = HEAD.format(root=ROOT) + textwrap.dedent(body).format(**fmt)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.returncode, p.stdout[-3000:], p.stderr[-3000:])
    return p.stdout


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_cpu_trainer_raises_instead_of_aborting(threads):
    out = run('''
        t = MCCFRTrainer(spec, bk, seed=1, backend="cpp", threads={threads})
        core._debug_fail_table_growth(-1)
        try:
            t.train(3000)
            print("no error")
        except RuntimeError as e:
            print("raised", "out of memory" in str(e), "do not save" in str(e), t._core.tables_consistent)
        core._debug_fail_table_growth(0)
        MCCFRTrainer(spec, bk, seed=1, backend="cpp", threads={threads}).train(500)  # the process goes on
        print("after ok")
    ''', threads=threads)
    assert "raised True True False" in out and "after ok" in out, out


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_batched_trainer_keeps_the_last_batch_and_resumes_exactly(threads):
    """A batch that fails is not applied: the tables are those of the previous batch, a checkpoint of them resumes
    to the tables of a run that never failed, bit for bit."""
    out = run('''
        import tempfile
        def export(t):
            return {{k: (list(r), list(s), v) for k, (_, r, s, v) in t._core.export_nodes().items()}}
        ref = MCCFRTrainer(spec, bk, seed=3, backend="cpp", threads={threads}).set_batch(64)
        ref.train(1600)
        t = MCCFRTrainer(spec, bk, seed=3, backend="cpp", threads={threads}).set_batch(64)
        t.train(320)
        core._debug_fail_table_growth(-1)
        try:
            t.train(1280)
            print("no error")
        except RuntimeError as e:
            print("raised", "was not applied" in str(e), t._core.tables_consistent, t.iteration % 64 == 0, 320 <= t.iteration < 1600)
        core._debug_fail_table_growth(0)
        ck = os.path.join(tempfile.mkdtemp(), "ck.bin")
        t.save_checkpoint(ck)
        u = MCCFRTrainer(spec, bk, seed=3, backend="cpp", threads={threads}).set_batch(64)
        u.load_checkpoint(ck)
        u.train(1600 - u.iteration)
        a, b = export(ref), export(u)
        print("resumed identical", {{k: v for k, v in b.items() if k in a}} == a and all(v[2] == 0 for k, v in b.items() if k not in a))
    ''', threads=threads)
    assert "raised True True True True" in out and "resumed identical True" in out, out


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["cpu", "emu"])
def test_flat_and_gpu_host_keep_the_last_batch_and_continue_exactly(threads, mode):
    """The flat trainer (CPU levels, or the GPU path with the device emulated): a preparation that fails ends
    train() with an error, the tables stay those of the last whole batch, and training continues from the same
    object to the tables of a run that never failed."""
    out = run('''
        def flat():
            ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), 5, True, {threads})
            ft.batch_size = 64
            ft.gpu_pass = 16
            ft.emulate_gpu = {emu}
            return ft
        def export(ft):
            return {{k: (list(r), list(s), v) for k, (r, s, v) in ft.export_nodes().items()}}
        ref = flat()
        ref.train(1280)
        ft = flat()
        core._debug_fail_prepare(700)  # the 700th preparation from now: in the 11th batch
        try:
            ft.train(1280)
            print("no error")
        except RuntimeError as e:
            print("raised", "out of memory" in str(e), ft.tables_consistent, ft.iteration % 64 == 0, 0 < ft.iteration < 1280)
        core._debug_fail_prepare(0)
        ft.train(1280 - ft.iteration)
        print("continued identical", export(ft) == export(ref))
        core._debug_fail_prepare(-1)
        try:
            flat().train(640)
        except RuntimeError as e:
            print("every preparation failing: raised", ft.iteration == 1280)
    ''', threads=threads, emu="True" if mode == "emu" else "False")
    assert "raised True True True True" in out and "continued identical True" in out and "failing: raised True" in out, out


def test_rnr_raises_instead_of_aborting():
    out = run('''
        from negpluribus.exploit.rnr import RNRTrainer
        for threads in (1, 4, 8):
            r = RNRTrainer(spec, bk, opponent_model=None, p_model=0.0, seed=1, backend="cpp", threads=threads)
            core._debug_fail_table_growth(-1)
            try:
                r.train(2000)
                print("no error")
            except RuntimeError as e:
                print("raised", "out of memory" in str(e))
            core._debug_fail_table_growth(0)
        print("after ok")
    ''')
    assert out.count("raised True") == 3 and "after ok" in out, out
