"""Checkpoint and blueprint files are written to ``<path>.tmp`` and renamed over ``<path>`` (csrc/binio.h BinWriter,
docs/backends.md): a save that fails never damages the file it would replace.  Binary and JSON checkpoints and the
binary blueprint, on a small push/fold game (C++ trainer, one thread).

The failure is a destination that cannot be replaced (read-only file, a directory in the way), as a disk that fills
up or a reader holding the file open on Windows would make it.

QA-5 (found while writing these tests, low severity): after a failed rename the complete ``<path>.tmp`` stays on disk
(BinWriter::finish nulls the file handle before renaming, so the destructor's clean-up never runs); for the production
tables that is a stray file of 0.3-5 GB next to the checkpoint.
"""
from __future__ import annotations

import hashlib
import os
import stat

import pytest

from negpluribus import fast
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")


def _spec():
    return GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))


def _sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _save(trainer, kind: str, path: str) -> None:
    if kind == "blueprint":
        trainer.save_blueprint(path)
    else:
        trainer.save_checkpoint(path)


KINDS = [("checkpoint", "ck.bin"), ("checkpoint", "ck.json"), ("blueprint", "bp.bin")]


@pytest.fixture()
def writable(tmp_path):
    """Makes every file under tmp_path writable again at the end (read-only files block the clean-up on Windows)."""
    yield tmp_path
    for root, _, files in os.walk(tmp_path):
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("kind,name", KINDS)
def test_a_stale_tmp_from_a_crashed_save_is_replaced(writable, kind, name):
    t = MCCFRTrainer(_spec(), seed=1, backend="cpp", threads=1).train(300)
    p = writable / name
    (writable / (name + ".tmp")).write_bytes(b"half a checkpoint" * 1000)
    _save(t, kind, str(p))
    assert not (writable / (name + ".tmp")).exists()
    assert core.file_kind(str(p)) == ("json" if name.endswith(".json") else kind)
    if kind == "checkpoint":
        again = MCCFRTrainer(_spec(), seed=1, backend="cpp", threads=1).load_checkpoint(str(p))
        assert again.iteration == 300


@pytest.mark.parametrize("kind,name", KINDS)
def test_a_save_that_cannot_replace_the_file_raises_and_keeps_the_previous_one(writable, kind, name):
    t = MCCFRTrainer(_spec(), seed=1, backend="cpp", threads=1).train(300)
    p = writable / name
    _save(t, kind, str(p))
    before = _sha(p)
    t.train(200)
    os.chmod(p, stat.S_IREAD)  # the rename over it fails
    with pytest.raises(RuntimeError):
        _save(t, kind, str(p))
    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    assert _sha(p) == before  # untouched
    if kind == "checkpoint":  # and still the complete earlier checkpoint
        assert MCCFRTrainer(_spec(), seed=1, backend="cpp", threads=1).load_checkpoint(str(p)).iteration == 300
    # a directory in the way: refused as well, nothing written under that name
    d = writable / ("dir_" + name)
    d.mkdir()
    with pytest.raises(RuntimeError):
        _save(t, kind, str(d))
    assert d.is_dir() and not any(d.iterdir())
    # a missing folder: refused before anything is written
    with pytest.raises(RuntimeError):
        _save(t, kind, str(writable / "no_such_folder" / name))
    assert not (writable / "no_such_folder").exists()


@pytest.mark.xfail(strict=True, reason="QA-5: csrc/binio.h BinWriter::finish() sets f_ = nullptr before the rename, so a "
                                        "failed rename leaves the whole <path>.tmp behind (the destructor only removes it "
                                        "while f_ is open)")
@pytest.mark.parametrize("kind,name", KINDS)
def test_a_failed_save_leaves_no_tmp_file(writable, kind, name):
    t = MCCFRTrainer(_spec(), seed=1, backend="cpp", threads=1).train(300)
    p = writable / name
    _save(t, kind, str(p))
    os.chmod(p, stat.S_IREAD)
    with pytest.raises(RuntimeError):
        _save(t, kind, str(p))
    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    assert not (writable / (name + ".tmp")).exists()
