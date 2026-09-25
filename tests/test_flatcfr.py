"""The GPU algorithm on the CPU (core.FlatTrainer, flatcfr.h): level-synchronous batched MCCFR on the flat
game must give the tables of Trainer's batched mode (mccfr.h) bit for bit."""
from __future__ import annotations

import os

import pytest

from negpluribus import fast
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import Street
from negpluribus.fast.trainer import core_bucketer, spec_to_dict

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "FlatTrainer"), reason="C++ core without the flat trainer (rebuild)")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _reference(spec, bk, seed, batch, iters, threads=2):
    tr = MCCFRTrainer(spec, bucketer=bk, seed=seed, backend="cpp", threads=threads).set_batch(batch)
    tr.train(iters)
    return {k: (list(r), list(s), v) for k, (_, r, s, v) in tr._core.export_nodes().items()}


GPU = hasattr(core, "cuda_available") and core.cuda_available()[0]
# the three ways to run the flat trainer: CPU levels, the GPU kernels emulated on the host, the GPU itself
MODES = ["cpu", "emu"] + (["gpu"] if GPU else [])


def _flat(spec, bk, seed, batch, iters, threads=1, pass_iterations=64, splits=None, mode="cpu"):
    from negpluribus.abstraction import EquityBucketer

    b = bk if bk is not None else EquityBucketer(n_buckets=spec.n_buckets)
    ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(b), seed, True, threads)
    ft.batch_size = batch
    ft.pass_iterations = pass_iterations
    ft.gpu_pass = pass_iterations if mode != "cpu" else 0
    if mode == "emu":
        ft.emulate_gpu = True
    elif mode == "gpu":
        ft.use_gpu(0)
    for n in (splits or [iters]):
        ft.train(n)
    assert ft.iteration == iters
    return {k: (list(r), list(s), v) for k, (r, s, v) in ft.export_nodes().items()}


@pytest.fixture(scope="module")
def flop_bucketer():
    from negpluribus.abstraction import EquityBucketer

    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    if os.path.exists(p):
        return EquityBucketer.load(p)
    return EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)


@pytest.fixture(scope="module")
def tiny_potential():
    from negpluribus.abstraction import PotentialAwareBucketer

    return PotentialAwareBucketer(n_buckets=8, samples=6, bins=10).fit(n_situations=120, seed=3)


@pytest.mark.parametrize("mode", MODES)
def test_push_fold_is_the_reference(mode):
    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    ref = _reference(spec, None, 4, 256, 3000)
    assert _flat(spec, None, 4, 256, 3000, mode=mode) == ref
    # threads, pass size and aligned splits change nothing
    assert _flat(spec, None, 4, 256, 3000, threads=3, pass_iterations=5, splits=[512, 2488], mode=mode) == ref


@pytest.mark.parametrize("mode", MODES)
def test_three_player_flop_is_the_reference(flop_bucketer, mode):
    spec = GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    ref = _reference(spec, flop_bucketer, 11, 32, 400)
    assert len(ref) > 100
    assert _flat(spec, flop_bucketer, 11, 32, 400, threads=2, pass_iterations=7, mode=mode) == ref


@pytest.mark.parametrize("mode", MODES)
def test_river_potential_is_the_reference(tiny_potential, mode):
    spec = GameSpec(n_players=2, stack_bb=20, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2, bucket_kind="potential")
    ref = _reference(spec, tiny_potential, 5, 64, 500)
    assert len(ref) > 500
    assert _flat(spec, tiny_potential, 5, 64, 500, threads=2, mode=mode) == ref


def test_cuda_available_reports_a_reason():
    ok, why = core.cuda_available()
    assert ok or why
