"""The Python writers of the project's files must not destroy the file they replace when the write fails half-way
(a full disk, an exception while serialising): the C++ writers go through <path>.tmp + rename
(tests/test_checkpoint_safety.py), these open the destination with "w" and truncate it first.

L2 (defect hunter, low; confirmed): negpluribus/cfr/strategy.py BlueprintStrategy.save, abstraction/buckets.py
EquityBucketer.save, abstraction/potential.py PotentialAwareBucketer.save and the Python trainer's
cfr/mccfr.py save_checkpoint.  The buckets JSON is the one that matters most: every blueprint of a tag is keyed on its
fingerprint, and train_blueprint.py writes it next to the blueprints.  (train_blueprint.py's .gpu.json and
aivat_fast.root_table's np.savez are the same pattern; not exercised here.)

The failure is simulated: json.dump writes the first bytes and raises OSError, as a full disk would.
"""
from __future__ import annotations

import json

import pytest

L2 = "L2: {} opens the destination with 'w' (truncating it) instead of writing <path>.tmp and renaming it"


@pytest.fixture()
def failing_dump(monkeypatch):
    real = json.dump

    def dump(obj, f, *a, **kw):
        f.write('{"half": ')
        raise OSError(28, "No space left on device")

    def arm():
        monkeypatch.setattr(json, "dump", dump)

    def disarm():
        monkeypatch.setattr(json, "dump", real)

    return arm, disarm


def _writers():
    from negpluribus.abstraction import EquityBucketer, PotentialAwareBucketer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.cfr.strategy import BlueprintStrategy
    from negpluribus.engine import Street

    def blueprint(path):
        BlueprintStrategy({"P|BB|2|b0|r1": (["f", "c"], [0.25, 0.75])}).save(path)

    def ehs(path):
        bk = EquityBucketer(n_buckets=4, samples=30)
        bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
        bk.save(path)

    def potential(path):
        PotentialAwareBucketer(n_buckets=4, samples=3, bins=5).fit(n_situations=40, seed=1).save(path)

    def checkpoint(path):
        spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
        MCCFRTrainer(spec, seed=0, backend="python").train(50).save_checkpoint(path)

    return {"BlueprintStrategy.save": blueprint, "EquityBucketer.save": ehs,
            "PotentialAwareBucketer.save": potential, "MCCFRTrainer(python).save_checkpoint": checkpoint}


@pytest.mark.parametrize("writer", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=L2.format(name)))
    for name in ("BlueprintStrategy.save", "EquityBucketer.save", "PotentialAwareBucketer.save",
                 "MCCFRTrainer(python).save_checkpoint")
])
def test_a_failed_write_keeps_the_previous_file(tmp_path, failing_dump, writer):
    arm, disarm = failing_dump
    write = _writers()[writer]
    path = str(tmp_path / "file.json")
    write(path)
    before = open(path, "rb").read()
    json.loads(before)  # a complete file
    arm()
    with pytest.raises(OSError):
        write(path)
    disarm()
    assert open(path, "rb").read() == before
