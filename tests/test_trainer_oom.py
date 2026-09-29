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
@pytest.mark.parametrize("mode", ["cpu", "emu", "emu-pool1", "emu-pool3"])
def test_flat_and_gpu_host_keep_the_last_batch_and_continue_exactly(threads, mode):
    """The flat trainer (CPU levels, or the GPU path with the device emulated: threads per batch, or the persistent
    preparation pool with a ring of 1 or 3 batches ahead): a preparation that fails ends train() with an error, the
    tables stay those of the last whole batch the device ran, and training continues from the same object to the
    tables of a run that never failed."""
    emu, pool, depth = mode != "cpu", mode.startswith("emu-pool"), int(mode[-1]) if mode.startswith("emu-pool") else 1
    out = run('''
        def flat():
            ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), 5, True, {threads})
            ft.batch_size = 64
            ft.gpu_pass = 16
            ft.emulate_gpu = {emu}
            ft.prep_pool = {pool}
            ft.prep_depth = {depth}
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
        f2 = flat()
        try:
            f2.train(640)
        except RuntimeError as e:
            print("every preparation failing: raised", f2.iteration == 0 and f2.tables_consistent)
        core._debug_fail_prepare(0)
    ''', threads=threads, emu=emu, pool=pool, depth=depth)
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


@pytest.mark.parametrize("batch", [0, 64])
def test_failed_growth_is_reported_and_retried(batch):
    """A node-table growth that fails is never lost: when it fails in the last worker's leave() or in end()
    (after the last lookup) train() still raises; the growth stays pending and is retried by the next train(),
    so the table never stays full with its request forgotten (inserts spun forever at the hard limit).  One
    thread, train(1) per call, the next growth failing: over many seeds every failure raises, and the same
    object then trains on."""
    out = run('''
        for seed in range(40):
            t = MCCFRTrainer(spec, bk, seed=seed, backend="cpp", threads=1)
            if {batch}:
                t.set_batch({batch})
            core._debug_fail_table_growth(1)
            raised = 0
            for i in range(300):
                try:
                    t.train({step})
                except RuntimeError as e:
                    assert "out of memory growing the node table" in str(e), str(e)
                    raised += 1
            core._debug_fail_table_growth(0)
            n = t.n_nodes
            t.train(3000)  # the growth is retried: no hang, the table grows
            print("seed", seed, "raised", raised, n < t.n_nodes)
    ''', batch=batch, step=batch or 1)
    lines = [ln for ln in out.splitlines() if ln.startswith("seed")]
    assert len(lines) == 40 and all(ln.split()[3] == "1" and ln.endswith("True") for ln in lines), out


def test_gpu_setup_error_on_resume_keeps_the_checkpoint(tmp_path):
    """train_blueprint.py --resume --gpu with a device that does not exist: the error comes before the GPU run holds
    the run's tables, so nothing is written -- the resumed checkpoint and its .gpu.json stay byte for byte (they were
    overwritten with untrained tables before), and a later --resume continues from them."""
    import hashlib

    script = os.path.join(ROOT, "scripts", "train_blueprint.py")
    base = [sys.executable, "-B", script, "--players", "2", "--stack", "10", "--street", "preflop", "--checkpoint-every", "1024",
            "--backend", "cpp", "--threads", "1", "--eval-deals", "0", "--tag", "t", "--data-dir", str(tmp_path), "--batch", "64", "--gpu"]
    p = subprocess.run(base + ["0", "--gpu-emulate", "--iters", "1024"], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    files = [tmp_path / "checkpoint_t.bin", tmp_path / "checkpoint_t.bin.gpu.json"]
    before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    p = subprocess.run(base + ["97", "--resume", "--iters", "1024"], capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "before the GPU run started: nothing written" in p.stdout + p.stderr, (p.stdout[-2000:], p.stderr[-2000:])
    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in files] == before
    p = subprocess.run(base + ["0", "--gpu-emulate", "--resume", "--iters", "1024"], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and "resumed from iteration 1,024" in p.stdout, (p.stdout[-2000:], p.stderr[-2000:])


def test_gpu_host_pool_edge_cases():
    """The GPU path's preparation pool (emulated device): a worker delayed after taking its chunk, with a short first
    batch, gives the tables of the path without the pool; a failure inside the device's batch leaves tables marked
    not whole, on both paths; the prepare-only benchmark touches no table (export after it used to read out of
    bounds); a pool whose thread cannot start falls back to the per-batch path and trains; the pool reports its
    preparation time."""
    out = run('''
        def flat(pool, threads=4):
            ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), 5, True, threads)
            ft.batch_size = 64
            ft.gpu_pass = 16
            ft.emulate_gpu = True
            ft.prep_pool = pool
            ft.prep_depth = 3
            return ft
        def export(ft):
            return {{k: (list(r), list(s), v) for k, (r, s, v) in ft.export_nodes().items()}}
        ref = flat(False)
        ref.train(37)  # a short first batch, then whole ones
        ref.train(600)
        core._debug_prep_delay_us(3000)
        ft = flat(True)
        ft.train(37)
        ft.train(600)
        core._debug_prep_delay_us(0)
        print("delayed pool identical", export(ft) == export(ref), ft.ms_prepare_total > 0)
        for pool in (False, True):
            ft = flat(pool)
            ft.train(128)
            core._debug_fail_device(2)
            try:
                ft.train(256)
                print("device failure: no error")
            except RuntimeError as e:
                print("device failure", pool, "do not save" in str(e), ft.tables_consistent)
            core._debug_fail_device(0)
        for pool in (False, True):
            ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), 5, True, 4)
            ft.batch_size = 64
            ft.prep_bench = True
            ft.prep_pool = pool
            ft.train(256)
            print("bench export", pool, len(export(ft)) >= 0, ft.iteration == 256)
        ref2 = flat(False)
        ref2.train(640)
        core._debug_fail_thread_start(1)
        ft = flat(True)
        ft.train(640)
        core._debug_fail_thread_start(0)
        print("start fallback identical", export(ft) == export(ref2))
    ''')
    assert "delayed pool identical True True" in out, out
    assert "device failure False True False" in out and "device failure True True False" in out, out
    assert "bench export False True True" in out and "bench export True True True" in out, out
    assert "start fallback identical True" in out, out


def test_first_batch_preparation_failure_says_where_the_tables_are(tmp_path):
    """The per-batch path: the first batch's preparation failing names the tables' iteration (a resumable
    checkpoint), as every later batch does; train_blueprint.py refuses a negative --gpu-prep-pool before any work."""
    out = run('''
        ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), 5, True, 2)
        ft.batch_size = 64
        ft.emulate_gpu = True
        ft.prep_pool = False
        ft.train(128)
        core._debug_fail_prepare(1)
        try:
            ft.train(128)
            print("no error")
        except RuntimeError as e:
            print("raised", "tables are those of iteration 128" in str(e), ft.tables_consistent)
        core._debug_fail_prepare(0)
    ''')
    assert "raised True True" in out, out
    p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "scripts", "train_blueprint.py"), "--players", "2", "--stack", "10",
                        "--street", "flop", "--backend", "cpp", "--iters", "10", "--tag", "t", "--data-dir", str(tmp_path),
                        "--gpu-prep-pool", "-1"], capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "--gpu-prep-pool" in p.stdout + p.stderr and "buckets:" not in p.stdout, p.stdout[-2000:]
