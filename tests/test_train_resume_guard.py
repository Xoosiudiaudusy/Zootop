"""scripts/train_blueprint.py --resume, checkpoints and snapshots, end to end on the push/fold game, one thread; the GPU
path runs on the CPU with --gpu-emulate (the GPU kernels' code on the host: the same numbers, no device used).

docs/gpu_training.md: a GPU checkpoint writes checkpoint_<tag>.bin.gpu.json (seed, batch, linear, linear_until) and a
resume with other values is refused; a resume with the same values continues bit for bit.  Checked:

  * a refused resume exits non-zero, names the settings and leaves the checkpoint, its run info and the blueprint
    untouched;
  * 1024 + --resume 512 iterations write the same blueprint and checkpoint bytes as 1536 in one run;
  * the GPU path's blueprint is the CPU batched mode's (the same average strategy).

Regression tests of defects, xfail(strict) while the defect is in master:

  * H1 (= QA-3): --resume without a checkpoint of the tag trains from iteration 0 without a word
    (train_blueprint.py:185-193 prints only when the file exists) and overwrites blueprint_<tag>.bin, the tag's
    "latest" blueprint; and the file resumed from is the NEWER of checkpoint_<tag>.bin / .json by mtime
    (fast/blueprint.py tagged_path), not the one further on.
  * M1 (= QA-2): with --gpu the checkpoint that ends the run gets no .it<N> snapshot (the loop skips its gpu_checkpoint()
    since "the final save below covers the last one", and that final save_outputs(snapshot=False) writes only
    blueprint_<tag>.bin); the CPU path writes it.  The final version of every GPU run (the 600M depth-grid points,
    pot64x at 1.6B) exists only under the name the next continuation of the tag overwrites.
  * M3: a CPU --resume keeps nothing it does not store: --batch, --linear-until and a --no-linear against a linear
    checkpoint go through silently (the GPU path refuses the same mismatches); at the trainer, linear_until and
    batch_size of a checkpoint are not restored.  (The checkpoint's linear flag winning over the constructor's is
    deliberate and tested in tests/test_persist.py:222; a fix that refuses instead must update that test.)
  * L1: CPU --batch with --checkpoint-every not a multiple of the batch splits batches at the checkpoints (a
    checkpointed run is another algorithm than the same run without checkpoints); the GPU path rounds it up.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "FlatTrainer"), reason="C++ core without the flat trainer")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "scripts", "train_blueprint.py")


def _run(data_dir, *extra, gpu=True):
    cmd = [sys.executable, "-B", SCRIPT, "--players", "2", "--stack", "10", "--street", "preflop", "--backend", "cpp",
           "--threads", "1", "--eval-deals", "0", "--tag", "g", "--data-dir", str(data_dir)]
    if gpu:
        cmd += ["--batch", "256", "--gpu-emulate"]
    env = dict(os.environ, NEGPLURIBUS_VERIFY_KEYS="0", NEGPLURIBUS_BUCKET_TABLES="")
    return subprocess.run(cmd + list(extra), capture_output=True, text=True, env=env, timeout=300)


def _sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _files(d):
    return {name: _sha(d / name) for name in ("checkpoint_g.bin", "checkpoint_g.bin.gpu.json", "blueprint_g.bin")}


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("gpu_resume")
    out = _run(d, "--iters", "1024", "--checkpoint-every", "512")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "EMULATED on the CPU" in out.stdout
    with open(d / "checkpoint_g.bin.gpu.json", encoding="utf-8") as f:
        info = json.load(f)
    assert info == {"seed": 0, "batch": 256, "linear": True, "linear_until": 0, "iteration": 1024}
    return d


@pytest.mark.parametrize("flags,what", [
    (["--seed", "1"], "seed"),
    (["--batch", "512"], "batch"),
    (["--linear-until", "300"], "linear_until"),
    (["--no-linear"], "linear"),
])
def test_a_resume_with_other_gpu_settings_is_refused_and_touches_nothing(first_run, flags, what):
    before = _files(first_run)
    out = _run(first_run, "--resume", "--iters", "256", *flags)  # a later --batch overrides _run's 256 (argparse)
    assert out.returncode != 0, out.stdout[-1500:]
    msg = out.stdout + out.stderr
    assert "other settings" in msg and what in msg, msg[-1500:]
    assert _files(first_run) == before


def test_a_resume_with_the_same_settings_continues_bit_for_bit(first_run, tmp_path):
    import shutil

    resumed = tmp_path / "resumed"
    shutil.copytree(first_run, resumed)
    out = _run(resumed, "--resume", "--iters", "512")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "resumed from iteration 1,024" in out.stdout and "GPU: resumed" in out.stdout
    straight = tmp_path / "straight"
    out = _run(straight, "--iters", "1536")
    assert out.returncode == 0, out.stderr[-2000:]
    a, b = _files(resumed), _files(straight)
    assert a["blueprint_g.bin"] == b["blueprint_g.bin"]
    assert a["checkpoint_g.bin"] == b["checkpoint_g.bin"]
    with open(resumed / "checkpoint_g.bin.gpu.json", encoding="utf-8") as f:
        assert json.load(f)["iteration"] == 1536


def test_the_gpu_path_writes_the_blueprint_of_the_cpu_batched_mode(first_run, tmp_path):
    """--gpu (here emulated) = --batch on the CPU (docs/gpu_training.md): the same average strategy in the blueprint."""
    from negpluribus.fast.binfmt import read_blueprint

    cpu = tmp_path / "cpu"
    out = _run(cpu, "--iters", "1024", "--checkpoint-every", "512", "--batch", "256", gpu=False)
    assert out.returncode == 0, out.stderr[-2000:]
    a = read_blueprint(str(first_run / "blueprint_g.bin"), verify=True).table
    b = read_blueprint(str(cpu / "blueprint_g.bin"), verify=True).table
    assert len(a) > 100 and a == b


@pytest.mark.xfail(strict=True, reason="M1 (= QA-2): train_blueprint.py GPU path skips the .it<N> snapshot of the last "
                                        "checkpoint (its final save_outputs(snapshot=False) 'covers' it); the CPU path writes it")
def test_the_gpu_path_keeps_a_snapshot_of_every_checkpoint_including_the_last(first_run):
    """The naming rule of the handoff (28.09 15:00): blueprint_<tag>.bin is the latest version and is overwritten when the
    tag trains on; fixed versions are the .it<N> snapshots.  The CPU path writes .it512 and .it1024 here
    (tests/test_persist.py checks it); a GPU run's final version must not exist only under the overwritable name."""
    assert (first_run / "blueprint_g.it512.bin").exists()
    assert (first_run / "blueprint_g.it1024.bin").exists()


def _iteration(path) -> int:
    from negpluribus.fast.blueprint import load_blueprint

    return load_blueprint(str(path), backend="cpp").iteration


def _ck_iteration(path) -> int:
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.engine import Street

    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,), max_raises_per_street=3)
    return MCCFRTrainer(spec, seed=0, backend="cpp", threads=1).load_checkpoint(str(path)).iteration


@pytest.mark.xfail(strict=True, reason="H1 (= QA-3): train_blueprint.py --resume without a checkpoint of the tag silently "
                                        "trains from iteration 0 and overwrites blueprint_<tag>.bin")
def test_resume_without_its_checkpoint_refuses_and_keeps_the_latest_blueprint(tmp_path):
    out = _run(tmp_path, "--iters", "3000", gpu=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert _iteration(tmp_path / "blueprint_g.bin") == 3000
    os.replace(tmp_path / "checkpoint_g.bin", tmp_path / "moved_away.bin")  # e.g. a data move, a mistyped folder
    out = _run(tmp_path, "--resume", "--iters", "100", gpu=False)
    assert out.returncode != 0, out.stdout[-1500:]
    assert _iteration(tmp_path / "blueprint_g.bin") == 3000


@pytest.mark.xfail(strict=True, reason="H1: --resume takes the newer FILE of checkpoint_<tag>.bin / .json "
                                        "(fast/blueprint.py tagged_path, by mtime), not the checkpoint further on")
def test_resume_continues_the_furthest_checkpoint_not_the_newest_file(tmp_path):
    out = _run(tmp_path, "--iters", "1000", "--json", gpu=False)  # checkpoint_g.bin and checkpoint_g.json at 1000
    assert out.returncode == 0, out.stderr[-2000:]
    out = _run(tmp_path, "--resume", "--iters", "2000", gpu=False)  # the binary goes on to 3000, the JSON stays at 1000
    assert out.returncode == 0 and "resumed from iteration 1,000" in out.stdout, out.stdout[-1500:]
    assert _ck_iteration(tmp_path / "checkpoint_g.bin") == 3000 and _ck_iteration(tmp_path / "checkpoint_g.json") == 1000
    st = os.stat(tmp_path / "checkpoint_g.bin")
    os.utime(tmp_path / "checkpoint_g.json", (st.st_atime + 60, st.st_mtime + 60))  # the older checkpoint, touched later
    out = _run(tmp_path, "--resume", "--iters", "100", gpu=False)
    assert out.returncode != 0 or "resumed from iteration 3,000" in out.stdout, out.stdout[-1500:]


@pytest.mark.parametrize("flags,what", [
    (["--batch", "128"], "batch"),
    (["--batch", "64", "--linear-until", "100"], "linear_until"),
    (["--batch", "64", "--no-linear"], "linear"),
])
@pytest.mark.xfail(strict=True, reason="M3: a CPU --resume neither stores nor checks --batch / --linear-until and lets a "
                                        "linear checkpoint override --no-linear silently (the GPU path refuses these)")
def test_a_cpu_resume_with_other_settings_is_refused(tmp_path, flags, what):
    out = _run(tmp_path, "--iters", "256", "--batch", "64", gpu=False)
    assert out.returncode == 0, out.stderr[-2000:]
    out = _run(tmp_path, "--resume", "--iters", "128", *flags, gpu=False)
    assert out.returncode != 0 and what in (out.stdout + out.stderr), out.stdout[-1500:]


@pytest.mark.parametrize("setting", ["linear_until", "batch_size"])
@pytest.mark.xfail(strict=True, reason="M3: csrc/persist.h checkpoints store neither linear_until nor batch_size; a trainer "
                                        "loading one keeps its own (a changed algorithm mid-run, without a word)")
def test_a_checkpoint_restores_or_refuses_the_settings_it_was_trained_with(tmp_path, setting):
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street

    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    a = MCCFRTrainer(spec, seed=0, backend="cpp", threads=1)
    if setting == "linear_until":
        a.set_linear_until(50)
    else:
        a.set_batch(64)
    a.train(256)
    a.save_checkpoint(str(tmp_path / "ck.bin"))
    b = MCCFRTrainer(spec, seed=0, backend="cpp", threads=1)
    try:
        b.load_checkpoint(str(tmp_path / "ck.bin"))
    except (ValueError, RuntimeError):
        return  # refused: fine
    assert getattr(b._core, setting) == getattr(a._core, setting)


@pytest.mark.xfail(strict=True, reason="L6: on the CPU, --seconds trains in one timed loop and ignores --checkpoint-every (no "
                                        "checkpoint or snapshot until the end)")
def test_a_timed_cpu_run_still_checkpoints(tmp_path):
    out = _run(tmp_path, "--seconds", "1.5", "--checkpoint-every", "2000", gpu=False)
    assert out.returncode == 0, out.stderr[-2000:]
    snapshots = [p.name for p in tmp_path.iterdir() if ".it" in p.name]
    assert snapshots, sorted(p.name for p in tmp_path.iterdir())


@pytest.mark.xfail(strict=True, reason="L1: CPU --batch with --checkpoint-every not a multiple of the batch splits batches at "
                                        "the checkpoints (train_blueprint.py trains in --checkpoint-every chunks)")
def test_checkpoints_do_not_change_a_cpu_batched_run(tmp_path):
    from negpluribus.fast.binfmt import read_blueprint

    plain, chunked = tmp_path / "plain", tmp_path / "chunked"
    assert _run(plain, "--iters", "256", "--batch", "64", gpu=False).returncode == 0
    out = _run(chunked, "--iters", "256", "--batch", "64", "--checkpoint-every", "100", gpu=False)
    assert out.returncode == 0, out.stderr[-2000:]
    a = read_blueprint(str(plain / "blueprint_g.bin")).table
    b = read_blueprint(str(chunked / "blueprint_g.bin")).table
    assert a == b
