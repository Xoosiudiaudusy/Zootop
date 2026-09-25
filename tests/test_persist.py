"""Binary checkpoints and blueprints, the C++ blueprint lookup, JSON written and read by C++
(csrc/persist.h, negpluribus/fast/blueprint.py, docs/backends.md "Binary checkpoints and blueprints").

What must hold, bit for bit:
  * resume from a binary checkpoint == resume from the JSON checkpoint of the same state ==
    an uninterrupted run (threads=1), down to identical checkpoint / blueprint file bytes;
  * the JSON the C++ core writes has the bytes json.dump wrote before; the JSON it reads gives the
    tables Python's json.load + import gave;
  * the C++ lookup returns the floats BlueprintStrategy.policy returns (JSON or binary file, or
    straight from the trainer), so an agent makes the same decisions on the same seeds;
  * the L1 diagnostic computed in C++ is the Python strategy_change() to the last bit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.cfr.strategy import BlueprintStrategy
from negpluribus.engine import Street

core = fast.core()
needs_core = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


@pytest.fixture(scope="module")
def flop_bucketer():
    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    if os.path.exists(p):
        return EquityBucketer.load(p)
    return EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)


def narrow_spec(players=3):
    # preflop and postflop share the raise names "r0.5" / "r1" (different action ids per street)
    return GameSpec(n_players=players, stack_bb=15, max_street=Street.FLOP, n_buckets=8,
                    preflop_fracs=(0.5, 1.0), postflop_fracs=(0.5, 1.0), max_raises_per_street=2)


def sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def old_checkpoint_json(trainer, path, rnr=False):
    """What the C++ trainers' save_checkpoint wrote before (Python dicts + json.dump)."""
    c = trainer._core
    d = {"iteration": trainer.iteration, "linear": trainer.linear}
    if rnr:
        d["nodes"] = {k: [list(a), list(r), list(s), v] for k, (a, r, s, v) in c.export_hero().items()}
        d["opp_nodes"] = {k: [list(a), list(r), list(s), v] for k, (a, r, s, v) in c.export_opp().items()}
    else:
        d["nodes"] = {k: [list(a), list(r), list(s), v] for k, (a, r, s, v) in c.export_nodes().items()}
    d.update(backend="cpp", threads=trainer.threads, rng_states=[list(st) for st in c.rng_states()])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)


def old_load_checkpoint_json(trainer, path):
    """What CppMCCFRTrainer.load_checkpoint did before (json.load + import_nodes)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    trainer.iteration = d["iteration"]
    trainer._core.import_nodes(d["nodes"], True)
    if d.get("rng_states"):
        trainer._core.set_rng_states([tuple(st) for st in d["rng_states"]])
    return trainer


# ================================================================= spellings
@needs_core
def test_json_spellings_and_rounding_are_those_of_cpython():
    rng = random.Random(11)
    vals = [0.0, -0.0, 1.0, 0.5, 1e-4, 1e-5, 9.999e-5, 1e15, 1e16, 1.5e16, 5e-324, 1.7976931348623157e308, 0.1, 1 / 3, 1e22, -2.5e-9]
    vals += [i / 64 for i in range(65)]
    vals += [rng.random() for _ in range(20000)]
    vals += [rng.random() * 10 ** rng.randint(-30, 30) * rng.choice((1, -1)) for _ in range(20000)]
    vals += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(20000)]
    vals = [v for v in vals if math.isfinite(v)] + [float("nan"), float("inf"), float("-inf")]
    assert [core.py_float_repr(v) for v in vals] == [json.dumps(v) for v in vals]
    strs = ["", 'a"b\\c', "\n\r\t\b\f\x00\x1f\x7f", "é日本𝄞", "F|BB|2|b3|c r0.5"]
    assert [core.py_json_string(s) for s in strs] == [json.dumps(s) for s in strs]
    rvals = [v for v in vals if not math.isnan(v)] + [k / 100000 + d for k in range(0, 100001, 97) for d in (0.0, 5e-6, -5e-6)]
    rvals += [i / 64 for i in range(1, 64, 2)]  # exact ties at the 6th decimal: half-even like Python
    for v in rvals:
        want = struct.pack("<d", round(v, 5))
        assert struct.pack("<d", core.round5(v)) == want, v
        assert struct.pack("<d", core.round5_exact(v)) == want, v


# ================================================================= checkpoints
@needs_core
@pytest.mark.parametrize("players", [2, 3])
def test_binary_resume_equals_json_resume_equals_uninterrupted_run(tmp_path, flop_bucketer, players):
    spec = narrow_spec(players)
    straight = MCCFRTrainer(spec, flop_bucketer, seed=3, backend="cpp", threads=1).train(500)
    first = MCCFRTrainer(spec, flop_bucketer, seed=3, backend="cpp", threads=1).train(300)
    first.save_checkpoint(str(tmp_path / "ck.bin"))
    first.save_checkpoint(str(tmp_path / "ck.json"))
    assert core.file_kind(str(tmp_path / "ck.bin")) == "checkpoint" and core.file_kind(str(tmp_path / "ck.json")) == "json"
    from_bin = MCCFRTrainer(spec, flop_bucketer, seed=99, backend="cpp", threads=1).load_checkpoint(str(tmp_path / "ck.bin"))
    from_json = MCCFRTrainer(spec, flop_bucketer, seed=99, backend="cpp", threads=1).load_checkpoint(str(tmp_path / "ck.json"))
    for t in (from_bin, from_json):
        assert t.iteration == 300 and t.linear
        assert t._core.rng_states() == first._core.rng_states()
        assert t.export_tables() == first.export_tables()
        t.train(200)
        assert t.iteration == 500
        assert t.export_tables() == straight.export_tables()
        assert t.strategy().table == straight.strategy().table
    # canonical files: the same contents give the same bytes, whatever the table's history
    for name, t in (("straight", straight), ("bin", from_bin), ("json", from_json)):
        t.save_checkpoint(str(tmp_path / f"{name}.bin"))
        t.save_blueprint(str(tmp_path / f"{name}_bp.bin"))
    assert sha(tmp_path / "straight.bin") == sha(tmp_path / "bin.bin") == sha(tmp_path / "json.bin")
    assert sha(tmp_path / "straight_bp.bin") == sha(tmp_path / "bin_bp.bin") == sha(tmp_path / "json_bp.bin")
    # the old JSON path (json.load + import_nodes) resumes to the same state as well
    old = old_load_checkpoint_json(MCCFRTrainer(spec, flop_bucketer, seed=99, backend="cpp", threads=1), str(tmp_path / "ck.json"))
    old.train(200)
    assert old.export_tables() == straight.export_tables()


@needs_core
def test_multithreaded_checkpoints_restore_every_stream(tmp_path, flop_bucketer):
    spec = narrow_spec(3)
    many = MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=4).train(400)
    many.save_checkpoint(str(tmp_path / "m.bin"))
    many.save_checkpoint(str(tmp_path / "m.json"))
    states = many._core.rng_states()
    assert len(states) == 4
    a = MCCFRTrainer(spec, flop_bucketer, seed=7, backend="cpp", threads=4).load_checkpoint(str(tmp_path / "m.bin"))
    b = MCCFRTrainer(spec, flop_bucketer, seed=7, backend="cpp", threads=4).load_checkpoint(str(tmp_path / "m.json"))
    for t in (a, b):
        assert t._core.rng_states() == states and t.iteration == 400
        assert t.export_tables() == many.export_tables()
    a.train(40)
    assert a.iteration == 440
    # the Python reference loads the JSON (and the export of the binary file)
    py = MCCFRTrainer(spec, flop_bucketer, seed=1).load_checkpoint(str(tmp_path / "m.json"))
    assert py.iteration == 400 and len(py.nodes) == many.n_nodes
    core.checkpoint_bin_to_json(str(tmp_path / "m.bin"), str(tmp_path / "m_export.json"))
    with open(tmp_path / "m_export.json", encoding="utf-8") as f:
        exported = json.load(f)
    with open(tmp_path / "m.json", encoding="utf-8") as f:
        written = json.load(f)
    assert exported == written  # same contents (keys in another order)


@needs_core
def test_json_checkpoints_have_the_bytes_of_json_dump(tmp_path, flop_bucketer):
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer

    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=2).train(300)
    t.save_checkpoint(str(tmp_path / "new.json"))
    old_checkpoint_json(t, str(tmp_path / "old.json"))
    assert sha(tmp_path / "new.json") == sha(tmp_path / "old.json")
    # blueprint JSON: the bytes of trainer.strategy().save()
    t.save_blueprint(str(tmp_path / "bp_new.json"))
    t.strategy().save(str(tmp_path / "bp_old.json"))
    assert sha(tmp_path / "bp_new.json") == sha(tmp_path / "bp_old.json")
    # the RNR trainer (hero and opponent tables)
    spec2 = GameSpec(n_players=2, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    bp = MCCFRTrainer(spec2, flop_bucketer, seed=0, backend="cpp", threads=1).train(200).strategy()
    r = RNRTrainer(spec2, flop_bucketer, OpponentModel(bp), 0.6, seed=3, warm_start=bp, backend="cpp", threads=2).train(200)
    r.save_checkpoint(str(tmp_path / "rnr_new.json"))
    old_checkpoint_json(r, str(tmp_path / "rnr_old.json"), rnr=True)
    assert sha(tmp_path / "rnr_new.json") == sha(tmp_path / "rnr_old.json")


@needs_core
def test_streamed_json_load_equals_json_load_on_odd_inputs(tmp_path, flop_bucketer):
    """Hand-made checkpoints: keys not in canonical form, escapes, NaN / Infinity, exponents, extra
    members and whitespace.  The C++ reader gives the table json.load + import_nodes gives."""
    spec = narrow_spec(3)
    base = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1).train(100)
    nodes = {k: [list(a), list(r), list(s), v] for k, (a, r, s, v) in base.export_tables().items()}
    nodes["weird \"key\" \\ é 𝄞"] = [["f", "c"], [float("inf"), -1e-300], [float("nan"), 1e300], 3]
    nodes["P|XYZ|3|b1|"] = [["c", "r0.5"], [0.0, -0.0], [5e-324, 1.5e16], 0]
    doc = {"backend": "python", "nodes": nodes, "extra": [1, {"x": None}], "linear": False, "iteration": 77}
    text = json.dumps(doc, indent=1, ensure_ascii=False)
    p = tmp_path / "odd.json"
    p.write_text(text, encoding="utf-8")
    a = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1).load_checkpoint(str(p))
    b = old_load_checkpoint_json(MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1), str(p))
    ea, eb = a.export_tables(), b.export_tables()
    assert set(ea) == set(eb) == set(nodes)
    for k in ea:
        assert repr(ea[k]) == repr(eb[k]), k  # repr: NaN compares unequal to itself
    assert a.iteration == 77 and a.linear is False  # the checkpoint's flag wins, as in the Python trainer
    # the Python trainer's own checkpoint (no RNG states): the C++ trainer keeps its seeded stream
    py = MCCFRTrainer(spec, flop_bucketer, seed=5).train(50)
    py.save_checkpoint(str(tmp_path / "py.json"))
    c = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1)
    states = c._core.rng_states()
    c.load_checkpoint(str(tmp_path / "py.json"))
    assert c._core.rng_states() == states and c.iteration == 50 and c.n_nodes == len(py.nodes)
    for bad in ('{"iteration": 1, "linear": true}', '{"iteration": 1, "linear": true, "nodes": {"k": [["c"], [1.0]]}}',
                '{"iteration": 1, "linear": true, "nodes": {"k": [["c"], [1.0], [1.0, 2.0], 1]}}', '{"iteration": 1, "lin'):
        q = tmp_path / "bad.json"
        q.write_text(bad, encoding="utf-8")
        with pytest.raises(RuntimeError):
            c.load_checkpoint(str(q))
        assert c.n_nodes == 0  # never half loaded


@needs_core
def test_linear_flag_of_a_checkpoint_reaches_the_core(tmp_path, flop_bucketer):
    """The Python trainer takes the checkpoint's linear flag; the C++ core now does too (before,
    only the Python attribute changed and the core kept its constructor's weighting)."""
    spec = narrow_spec(2)
    lin = MCCFRTrainer(spec, flop_bucketer, seed=2, backend="cpp", threads=1).train(100)
    lin.save_checkpoint(str(tmp_path / "lin.bin"))
    plain = MCCFRTrainer(spec, flop_bucketer, seed=2, backend="cpp", threads=1, linear=False)
    assert plain.linear is False and plain._core.linear is False
    plain.load_checkpoint(str(tmp_path / "lin.bin"))
    assert plain.linear is True and plain._core.linear is True
    plain.train(50)
    lin.train(50)
    assert plain.export_tables() == lin.export_tables()


@needs_core
def test_binary_checkpoint_of_another_game_is_refused(tmp_path, flop_bucketer):
    from negpluribus.abstraction import PotentialAwareBucketer

    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=1).train(50)
    p = str(tmp_path / "ck.bin")
    t.save_checkpoint(p)
    others = {
        "stack_bb": GameSpec(n_players=3, stack_bb=20, max_street=Street.FLOP, n_buckets=8, preflop_fracs=(0.5, 1.0),
                             postflop_fracs=(0.5, 1.0), max_raises_per_street=2),
        "players": narrow_spec(2),
        "postflop_fracs": GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8, preflop_fracs=(0.5, 1.0),
                                   postflop_fracs=(0.5, 1.0, 2.0), max_raises_per_street=2),
        "max_raises_per_street": GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8, preflop_fracs=(0.5, 1.0),
                                          postflop_fracs=(0.5, 1.0), max_raises_per_street=3),
    }
    for what, other in others.items():
        u = MCCFRTrainer(other, flop_bucketer, seed=1, backend="cpp", threads=1).train(5)
        n = u.n_nodes
        with pytest.raises(RuntimeError, match="refusing to resume.*" + what):
            u.load_checkpoint(p)
        assert u.n_nodes == n and u.iteration == 5  # untouched
    refit = EquityBucketer(flop_bucketer.n_buckets, flop_bucketer.samples)
    refit.boundaries = {k: [x + 1e-9 for x in v] for k, v in flop_bucketer.boundaries.items()}
    with pytest.raises(RuntimeError, match="fingerprint"):
        MCCFRTrainer(spec, refit, seed=1, backend="cpp", threads=1).load_checkpoint(p)
    pot = PotentialAwareBucketer(n_buckets=8, samples=4, bins=5).fit(n_situations=40, seed=1)
    pspec = GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8, preflop_fracs=(0.5, 1.0),
                     postflop_fracs=(0.5, 1.0), max_raises_per_street=2, bucket_kind="potential")
    with pytest.raises(RuntimeError, match="bucketer"):
        MCCFRTrainer(pspec, pot, seed=1, backend="cpp", threads=1).load_checkpoint(p)
    # the same game loads, and a JSON checkpoint (no identity recorded) is never refused
    MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=1).load_checkpoint(p)
    # preflop-only games ignore the bucketer (their keys use the 169 classes)
    pf = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    MCCFRTrainer(pf, seed=0, backend="cpp", threads=1).train(100).save_checkpoint(str(tmp_path / "pf.bin"))
    MCCFRTrainer(pf, EquityBucketer(4, 20), seed=0, backend="cpp", threads=1).load_checkpoint(str(tmp_path / "pf.bin"))
    # an RNR checkpoint is not an MCCFR checkpoint
    assert t._core.identity()["n_players"] == 3


@needs_core
def test_corrupt_or_truncated_files_are_rejected(tmp_path, flop_bucketer):
    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=1).train(100)
    p = tmp_path / "ck.bin"
    t.save_checkpoint(str(p))
    t.save_blueprint(str(tmp_path / "bp.bin"))
    assert not (tmp_path / "ck.bin.tmp").exists()
    data = p.read_bytes()
    for i, damaged in enumerate((data[: len(data) // 2], data[:-3], data[:500] + bytes([data[500] ^ 1]) + data[501:],
                                 data[: len(data) - 20] + bytes([data[-20] ^ 0x10]) + data[len(data) - 19:])):
        q = tmp_path / f"bad{i}.bin"
        q.write_bytes(damaged)
        u = MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=1)
        with pytest.raises(RuntimeError):
            u.load_checkpoint(str(q))
        assert u.n_nodes == 0 and u.iteration == 0
    b = (tmp_path / "bp.bin").read_bytes()
    (tmp_path / "bp_bad.bin").write_bytes(b[:200] + bytes([b[200] ^ 4]) + b[201:])
    with pytest.raises(RuntimeError):
        core.BlueprintTable.load(str(tmp_path / "bp_bad.bin"))
    with pytest.raises(ValueError):
        MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1).load_checkpoint(str(tmp_path / "bp.bin"))


@needs_core
def test_rnr_binary_checkpoint_resume(tmp_path, flop_bucketer):
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer

    spec = GameSpec(n_players=2, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    bp = MCCFRTrainer(spec, flop_bucketer, seed=0, backend="cpp", threads=1).train(300).strategy()
    model = OpponentModel(bp)
    model.theta[("flop", True)] = {"f": -0.5, "c": 0.7, "r": -0.2}

    def make():
        return RNRTrainer(spec, flop_bucketer, model, 0.6, seed=3, warm_start=bp, backend="cpp", threads=1)

    straight = make().train(200)
    first = make().train(200)
    first.save_checkpoint(str(tmp_path / "r.bin"))
    first.save_checkpoint(str(tmp_path / "r.json"))
    a = make().load_checkpoint(str(tmp_path / "r.bin"))
    b = make().load_checkpoint(str(tmp_path / "r.json"))
    straight.train(100)
    for t in (a, b):
        assert t.iteration == 200
        t.train(100)
        assert t._core.export_hero() == straight._core.export_hero()
        assert t._core.export_opp() == straight._core.export_opp()
    with pytest.raises(RuntimeError, match="RNR"):
        MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1).load_checkpoint(str(tmp_path / "r.bin"))


# ================================================================= blueprints
LEGAL_LISTS = [["f", "c", "r0.5", "r1", "a"], ["c", "r0.5", "a"], ["f", "c"], ["c"], ["a", "c", "f"],
               ["x", "c", "c"], ["r1"], [], ["f", "c", "r0.5", "r1", "a", "r2", "zz", "c"]]


def assert_same_policies(ref, other, keys):
    n = 0
    for k in keys:
        for legal in LEGAL_LISTS:
            assert other.policy(k, list(legal)) == ref.policy(k, list(legal)), (k, legal)
            n += 1
    return n


@needs_core
def test_cpp_lookup_gives_the_floats_of_blueprint_strategy(tmp_path, flop_bucketer):
    from negpluribus.fast.binfmt import read_blueprint
    from negpluribus.fast.blueprint import CppBlueprint, load_blueprint

    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=4, backend="cpp", threads=1).train(400)
    strat = t.strategy()
    keys = list(strat.table)
    probe = keys + ["nope", "P|BTN|3|b0|zz", keys[0] + " "]
    # in memory, full precision: the floats of trainer.strategy()
    mem = t.blueprint()
    assert isinstance(mem, CppBlueprint) and len(mem) == len(strat)
    assert assert_same_policies(strat, mem, probe) > 10000
    # the files: JSON (old and new writer) and binary hold round(p, 5); every reader agrees
    strat.save(str(tmp_path / "bp.json"))
    t.save_blueprint(str(tmp_path / "bp.bin"))
    ref = BlueprintStrategy.load(str(tmp_path / "bp.json"))
    for other in (load_blueprint(str(tmp_path / "bp.json")), load_blueprint(str(tmp_path / "bp.bin")),
                  load_blueprint(str(tmp_path / "bp.bin"), keys=True), t.blueprint(rounded=True),
                  read_blueprint(str(tmp_path / "bp.bin"), verify=True), load_blueprint(str(tmp_path / "bp.bin"), backend="python")):
        assert len(other) == len(ref)
        assert_same_policies(ref, other, probe)
    lk = load_blueprint(str(tmp_path / "bp.bin"))
    assert lk.rounded and lk.iteration == 400 and lk.identity["n_players"] == 3 and lk.get("nope") is None
    k = keys[5]
    assert lk.get(k) == ref.table[k] or (lk.get(k)[0] == list(ref.table[k][0]) and lk.get(k)[1] == list(ref.table[k][1]))
    assert k in lk and "nope" not in lk and 5 not in lk
    assert lk.to_strategy().table == {kk: (list(n), list(p)) for kk, (n, p) in ref.table.items()}
    st = lk.stats()
    assert st["key_string_bytes"] == 0 and st["bytes"] / len(lk) < 60
    # rounded probabilities are held as k (k / 100000.0 is the double round() gave): 4 bytes each
    assert st["packed"] and st["prob_bytes"] == 4 * st["actions"]
    assert load_blueprint(str(tmp_path / "bp.json")).stats()["packed"] and not mem.stats()["packed"]
    # exports: binary -> JSON (C++ and pure Python) and JSON -> binary -> JSON keep every value
    lk.save(str(tmp_path / "exp.json"))
    with open(tmp_path / "exp.json", encoding="utf-8") as f:
        exported = json.load(f)["table"]
    with open(tmp_path / "bp.json", encoding="utf-8") as f:
        original = json.load(f)["table"]
    assert exported == original
    read_blueprint(str(tmp_path / "bp.bin")).save(str(tmp_path / "exp_py.json"))
    assert sha(tmp_path / "exp_py.json") == sha(tmp_path / "exp.json")  # the same bytes from both writers
    load_blueprint(str(tmp_path / "bp.json"), keys=True).save(str(tmp_path / "from_json.bin"))
    assert_same_policies(ref, load_blueprint(str(tmp_path / "from_json.bin")), probe)


@needs_core
def test_tagged_files_pick_the_newest_format_and_dict_tools_read_binary(tmp_path, flop_bucketer):
    """Scripts that open blueprint_<tag> / checkpoint_<tag> by name take the .bin or the .json,
    whichever was written last, and dict-based tools get a BlueprintStrategy from either."""
    from negpluribus.fast.blueprint import load_blueprint, tagged_path

    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=2, backend="cpp", threads=1).train(200)
    assert tagged_path(str(tmp_path), "blueprint", "x") == str(tmp_path / "blueprint_x.json")  # neither: the old name
    t.save_blueprint(str(tmp_path / "blueprint_x.json"))
    t.save_blueprint(str(tmp_path / "blueprint_x.bin"))
    os.utime(tmp_path / "blueprint_x.json", (1_000_000_000, 1_000_000_000))
    assert tagged_path(str(tmp_path), "blueprint", "x") == str(tmp_path / "blueprint_x.bin")
    os.utime(tmp_path / "blueprint_x.bin", (900_000_000, 900_000_000))
    assert tagged_path(str(tmp_path), "blueprint", "x") == str(tmp_path / "blueprint_x.json")
    os.utime(tmp_path / "blueprint_x.json", (900_000_000, 900_000_000))
    assert tagged_path(str(tmp_path), "blueprint", "x") == str(tmp_path / "blueprint_x.bin")  # a tie: the binary
    d_bin = load_blueprint(str(tmp_path / "blueprint_x.bin"), backend="python")
    d_json = load_blueprint(str(tmp_path / "blueprint_x.json"), backend="python")
    assert isinstance(d_bin, BlueprintStrategy) and isinstance(d_json, BlueprintStrategy)
    assert d_bin.table == {k: (list(n), list(p)) for k, (n, p) in d_json.table.items()}


@needs_core
def test_json_blueprints_with_duplicates_and_odd_entries(tmp_path):
    from negpluribus.fast.blueprint import load_blueprint

    table = {
        "P|BTN|3|b0|": [["f", "c", "r1", "c"], [0.2, 0.3, 0.1, 0.4]],   # "c" twice: the last one counts
        "F|BB|3|b2|c": [["c", "r0.5", "a"], [0.0, 0.0, 0.0]],         # all zero: None
        "T|SB|2|b1|c/c": [["c", "a"], [1, 2]],                       # ints
        "R|BB|2|b7|c/c/c": [["f", "c", "a"], [0.5, 0.5]],           # zip() stops at the shorter list
        "weird key": [["c"], [1.0]],
    }
    text = '{"table": {' + ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in table.items())
    text += ', "P|BTN|3|b0|": [["f", "c"], [0.9, 0.1]]}, "other": 1}'  # a key given twice: the last wins
    p = tmp_path / "odd.json"
    p.write_text(text, encoding="utf-8")
    ref = BlueprintStrategy.load(str(p))
    cpp = load_blueprint(str(p))
    assert len(cpp) == len(ref) == 5
    assert_same_policies(ref, cpp, list(ref.table) + ["missing"])
    assert cpp.policy("P|BTN|3|b0|", ["c", "f"]) == ref.policy("P|BTN|3|b0|", ["c", "f"])
    assert cpp.stats()["packed"]
    # one value that is not a 5-decimal number (or a -0.0): the table keeps doubles, same floats
    for odd in ("0.1234567", "-0.0"):
        q = tmp_path / "odd2.json"
        q.write_text('{"table": {"P|BTN|3|b1|": [["f", "c", "a"], [' + odd + ', 0.5, 0.25]], "F|BB|3|b0|c": [["c"], [1.0]]}}',
                     encoding="utf-8")
        ref2, cpp2 = BlueprintStrategy.load(str(q)), load_blueprint(str(q))
        assert not cpp2.stats()["packed"]
        assert_same_policies(ref2, cpp2, list(ref2.table))
        got = cpp2.policy("P|BTN|3|b1|", ["f", "c", "a"])
        want = ref2.policy("P|BTN|3|b1|", ["f", "c", "a"])
        assert [struct.pack("<d", x) for x in got] == [struct.pack("<d", x) for x in want]  # the sign of zero too


@needs_core
def test_same_decisions_with_the_cpp_lookup_in_duplicate_matches(tmp_path, flop_bucketer):
    """BlueprintAgent on the dict and on the C++ lookup (loaded from the JSON or from the binary
    file of the same state): the same key, the same probabilities, the same action at every
    decision, the same result of every deal."""
    from negpluribus.agents import make_agent
    from negpluribus.agents.blueprint import BlueprintAgent
    from negpluribus.eval import duplicate_match
    from negpluribus.fast.blueprint import load_blueprint

    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=8, backend="cpp", threads=2).train(3000)
    t.strategy().save(str(tmp_path / "bp.json"))
    t.save_blueprint(str(tmp_path / "bp.bin"))

    class Recorder:
        def __init__(self, inner):
            self.inner, self.log = inner, []

        def policy(self, key, legal):
            p = self.inner.policy(key, legal)
            self.log.append((key, tuple(legal), None if p is None else tuple(p)))
            return p

    runs = {}
    for name, strat in (("dict", BlueprintStrategy.load(str(tmp_path / "bp.json"))), ("cpp_json", load_blueprint(str(tmp_path / "bp.json"))),
                        ("cpp_bin", load_blueprint(str(tmp_path / "bp.bin"))), ("mem_dict", t.strategy()), ("mem_cpp", t.blueprint())):
        rec = Recorder(strat)
        hero = BlueprintAgent(rec, flop_bucketer, spec.grid, seed=1, name="hero")
        acts = []
        orig = hero.act

        def act(obs, orig=orig, acts=acts):
            a = orig(obs)
            acts.append((a.type, a.amount))
            return a

        hero.act = act
        vils = [make_agent("tag", seed=10 + i, label=f"tag{i}") for i in range(2)]
        res = duplicate_match(hero, vils, n_deals=300, seed=4, sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street)
        runs[name] = (rec.log, acts, res.per_deal_bb, hero.n_decisions, hero.n_fallback)
    assert len(runs["dict"][0]) > 800
    assert runs["cpp_json"] == runs["dict"] == runs["cpp_bin"]
    assert runs["mem_cpp"] == runs["mem_dict"]


@needs_core
def test_strategy_change_in_cpp_is_the_python_number(flop_bucketer):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from train_blueprint import strategy_change
    finally:
        sys.path.pop(0)
    spec = narrow_spec(3)
    t = MCCFRTrainer(spec, flop_bucketer, seed=6, backend="cpp", threads=1).train(300)
    prev_dict, prev_cpp = t.strategy(), t.blueprint()
    t.train(300)
    want = strategy_change(prev_dict, t.strategy())
    got = t.strategy_change(prev_cpp)
    assert want is not None and got == want  # bit for bit
    assert t.strategy_change(t.blueprint()) == 0.0
    # nothing shared -> None
    other = MCCFRTrainer(narrow_spec(3), flop_bucketer, seed=1, backend="cpp", threads=1)
    assert other.strategy_change(prev_cpp) is None
    assert t.count_keys("P|BTN|3|", "|") == sum(1 for k in t.nodes if k.startswith("P|BTN|3|") and k.endswith("|"))


@needs_core
def test_train_blueprint_script_writes_binary_resumes_and_exports(tmp_path):
    """The script end to end on a small game: binary checkpoints and blueprints (+ JSON with
    --json), .it<N> snapshots, the L1 line, --resume from the binary file, export_json.py."""
    script = os.path.join(ROOT, "scripts", "train_blueprint.py")
    base = [sys.executable, "-B", script, "--players", "2", "--stack", "10", "--street", "preflop", "--iters", "2000",
            "--checkpoint-every", "1000", "--backend", "cpp", "--threads", "1", "--eval-deals", "20", "--tag", "t",
            "--data-dir", str(tmp_path)]
    env = dict(os.environ, NEGPLURIBUS_VERIFY_KEYS="1")
    out = subprocess.run(base + ["--json"], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr
    assert "mean L1 change vs previous n/a" in out.stdout and "mean L1 change vs previous 0." in out.stdout
    for name in ("checkpoint_t.bin", "blueprint_t.bin", "blueprint_t.it1000.bin", "blueprint_t.it2000.bin",
                 "checkpoint_t.json", "blueprint_t.json", "blueprint_t.it1000.json"):
        assert (tmp_path / name).exists(), name
    assert core.file_kind(str(tmp_path / "checkpoint_t.bin")) == "checkpoint"
    # the binary blueprint and its JSON twin give the same policies
    from negpluribus.fast.blueprint import load_blueprint

    ref = BlueprintStrategy.load(str(tmp_path / "blueprint_t.json"))
    assert_same_policies(ref, load_blueprint(str(tmp_path / "blueprint_t.bin")), list(ref.table))
    again = subprocess.run(base + ["--resume", "--iters", "1000"], capture_output=True, text=True, env=env, timeout=600)
    assert again.returncode == 0, again.stderr
    assert "resumed from iteration 2,000" in again.stdout
    exp = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "scripts", "export_json.py"), str(tmp_path / "checkpoint_t.bin"),
                          str(tmp_path / "exported.json")], capture_output=True, text=True, timeout=600)
    assert exp.returncode == 0, exp.stderr
    with open(tmp_path / "exported.json", encoding="utf-8") as f:
        assert json.load(f)["iteration"] == 3000
