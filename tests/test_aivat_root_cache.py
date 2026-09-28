"""The AIVAT root-table cache (negpluribus/eval/aivat_fast.py root_table, scripts/aivat_eval.py --root-cache).

The root table (u at the start of a hand, 93,769 classes x 2 seats, 256 rollouts each: minutes on the production
game) is cached in an .npz with the parameters it was built with and rebuilt when they differ.  aivat_eval.py passes
as identity the blueprint's and the buckets' file names and the game (fracs, raises, --stack-bb).  Checked here:

  * root_table: built once per (rollouts, seed, identity), reloaded bit for bit, rebuilt when any of them changes,
    never when only the game object changes (the identity is the caller's job) - on a stubbed core, and one real
    round trip on a tiny game;
  * aivat_eval.py: the identity it passes changes with --stack-bb, the bet grid, the blueprint and the buckets file
    (main() run in-process up to the root table, the heavy parts stubbed).

QA-4 (found while writing these tests, low severity): the identity holds only os.path.basename of --blueprint.  data/
blueprint_<tag>.bin is by convention the LATEST version of a tag (overwritten at every checkpoint, handoff 28.09
15:00), so after the tag trains on the cached root table of the older version is reused without a word, and two
different blueprints with the same file name in two folders share one cache.  AIVAT stays unbiased with any root
table (Lemma 1: every term has mean zero for any value function), so the cost is variance, not bias.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

from negpluribus import fast
from negpluribus.eval import aivat_fast

core = fast.core()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------------------- root_table on a stub core
class _StubTable:
    def __init__(self, v0, v1):
        self.values = (list(v0), list(v1))


def _stub_core(builds):
    class AivatRootTable:
        @staticmethod
        def build(game, rollouts, seed, threads):
            builds.append((game, rollouts, seed))
            return _StubTable([float(game), float(rollouts)], [float(seed), -1.0])

        @staticmethod
        def from_values(v0, v1):
            return _StubTable(v0, v1)

    return types.SimpleNamespace(AivatRootTable=AivatRootTable)


def test_root_table_is_cached_per_parameters_and_identity(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(aivat_fast, "_core", lambda: _stub_core(builds))
    p = str(tmp_path / "sub" / "root.npz")
    ident = {"blueprint": "blueprint_a.bin", "buckets": "buckets_a.json", "grid": ["1.0", "0.5,1.0", 2, 30]}

    a = aivat_fast.root_table(1, 4, seed=0, cache_path=p, identity=ident)
    assert len(builds) == 1 and os.path.exists(p)  # the folder is created
    with np.load(p, allow_pickle=False) as z:
        assert json.loads(str(z["meta"])) == {"rollouts": 4, "seed": 0, **ident}
    # the same parameters: read back, not rebuilt, even for another game object (the identity is the caller's job)
    b = aivat_fast.root_table(2, 4, seed=0, cache_path=p, identity=dict(ident))
    assert len(builds) == 1 and b.values == a.values
    # any change of the parameters or of the identity rebuilds (and the file then holds the new one)
    for kw in (dict(rollouts=8), dict(seed=1), dict(identity={**ident, "grid": ["1.0", "0.5,1.0", 2, 40]}),
               dict(identity={**ident, "blueprint": "blueprint_b.bin"}), dict(identity={**ident, "buckets": "b.json"}),
               dict(identity=None)):
        n = len(builds)
        args = {"rollouts": 4, "seed": 0, "identity": ident, **kw}
        aivat_fast.root_table(3, args["rollouts"], seed=args["seed"], cache_path=p, identity=args["identity"])
        assert len(builds) == n + 1, kw
    # without a cache path: always built
    aivat_fast.root_table(4, 4, seed=0)
    aivat_fast.root_table(4, 4, seed=0)
    assert [g for g, _, _ in builds[-2:]] == [4, 4]


def _tiny(tmp_path, seed=3, iters=600, name="blueprint_t.bin"):
    """A tiny HU 30bb game: buckets file (fixed E[HS] cut points), a trained binary blueprint."""
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street

    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=4, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    bk_path = tmp_path / "buckets_t.json"
    if not bk_path.exists():
        bk.save(str(bk_path))
    t = MCCFRTrainer(spec, bk, seed=seed, backend="cpp", threads=1).train(iters)
    bp_path = tmp_path / name
    bp_path.parent.mkdir(parents=True, exist_ok=True)
    t.save_blueprint(str(bp_path))
    return spec, bk, t, str(bk_path), str(bp_path)


@pytest.mark.skipif(core is None or not hasattr(core, "AivatRootTable"), reason="C++ core with AIVAT not built")
def test_root_table_cache_round_trip_on_the_core(tmp_path):
    from negpluribus.fast.trainer import core_bucketer

    spec, bk, t, _, _ = _tiny(tmp_path)
    game = aivat_fast.make_game(spec, core_bucketer(bk, cache_caps=(1 << 16,) * 3), t.blueprint())  # small caches
    p = str(tmp_path / "root.npz")
    built = aivat_fast.root_table(game, 1, seed=5, threads=1, cache_path=p, identity={"x": 1})
    again = aivat_fast.root_table(game, 1, seed=5, threads=1, cache_path=p, identity={"x": 1})
    assert built.n_classes == again.n_classes == 93_769
    v0, v1 = built.values
    w0, w1 = again.values
    assert list(v0) == list(w0) and list(v1) == list(w1)
    assert list(built.mean) == list(again.mean)


# ---------------------------------------------------------------------------------------- aivat_eval.py's identity
class _Captured(Exception):
    pass


def _load_script():
    spec = importlib.util.spec_from_file_location("aivat_eval_under_test", os.path.join(ROOT, "scripts", "aivat_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def identity_of(tmp_path, monkeypatch):
    """identity_of(blueprint, buckets, *flags) -> the identity aivat_eval.py main() hands to root_table."""
    if core is None or not hasattr(core, "AivatGame"):
        pytest.skip("C++ core with AIVAT not built")
    mod = _load_script()
    hands = tmp_path / "hands.jsonl"  # one heads-up hand of eval_archetypes --log-hands: the small blind folds
    hands.write_text(json.dumps({"opponent": "x", "agent": "blueprint", "deal": 0, "hero_seat": 0, "button": 0,
                                 "holes": [[0, 1], [2, 3]], "board": [], "events": [[0, 0, 0, 0]], "net_bb": -0.5,
                                 "luck_bb": None, "hero": []}) + "\n")
    # the bucket tables (minutes on a real abstraction) and the table wrapper are stubbed; nothing else is
    monkeypatch.setattr(core, "aivat_build_tables", lambda cbk, threads, streets: (core.BucketTables(), "stub"))
    monkeypatch.setattr(core, "TabulatedBucketer", lambda cbk, tables: cbk)

    def capture(game, rollouts, seed=0, threads=1, cache_path=None, identity=None):
        raise _Captured(identity)

    monkeypatch.setattr(mod, "root_table", capture)
    calls = [0]

    def run(blueprint, buckets, *flags):
        calls[0] += 1
        argv = ["aivat_eval.py", "--duel", str(hands), "--out", str(tmp_path / f"out{calls[0]}.jsonl"),
                "--blueprint", blueprint, "--buckets", buckets, "--tables", str(tmp_path / f"tables{calls[0]}"),
                "--root-cache", str(tmp_path / "root.npz"), "--threads", "1", "--stack-bb", "30",
                "--preflop-fracs", "1.0", "--postflop-fracs", "0.5,1.0", "--max-raises", "2", "--no-luck", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(_Captured) as e:
            mod.main()
        return e.value.args[0]

    return run


def test_aivat_eval_identity_follows_stack_grid_blueprint_and_buckets(tmp_path, identity_of):
    _, _, _, bk_path, bp_path = _tiny(tmp_path)
    base = identity_of(bp_path, bk_path)
    assert base["blueprint"] == os.path.basename(bp_path) and base["buckets"] == os.path.basename(bk_path)
    assert identity_of(bp_path, bk_path) == base  # deterministic
    assert identity_of(bp_path, bk_path, "--stack-bb", "40") != base
    assert identity_of(bp_path, bk_path, "--postflop-fracs", "0.5,1.0,2.0") != base
    assert identity_of(bp_path, bk_path, "--preflop-fracs", "1.0,3.0") != base
    assert identity_of(bp_path, bk_path, "--max-raises", "3") != base
    _, _, _, _, other = _tiny(tmp_path, seed=4, name="blueprint_other.bin")
    assert identity_of(other, bk_path) != base
    copy = tmp_path / "buckets_copy.json"
    copy.write_bytes(open(bk_path, "rb").read())
    assert identity_of(bp_path, str(copy)) != base


@pytest.mark.xfail(strict=True, reason="H3 (= QA-4): scripts/aivat_eval.py keys the root cache on os.path.basename(--blueprint): "
                                        "a re-trained blueprint under the same name reuses the old root table")
def test_aivat_eval_identity_changes_when_the_blueprint_file_is_retrained(tmp_path, identity_of):
    _, _, _, bk_path, bp_path = _tiny(tmp_path, seed=3, iters=600)
    before = identity_of(bp_path, bk_path)
    _tiny(tmp_path, seed=3, iters=1200)  # the tag trained on: the same file name, another blueprint
    assert identity_of(bp_path, bk_path) != before


@pytest.mark.xfail(strict=True, reason="H3 (= QA-4): two different blueprints with the same file name in two folders share one "
                                        "root-cache identity in scripts/aivat_eval.py")
def test_aivat_eval_identity_tells_same_named_blueprints_in_two_folders_apart(tmp_path, identity_of):
    _, _, _, bk_path, a = _tiny(tmp_path, seed=3, name=os.path.join("run_a", "blueprint_x.bin"))
    _, _, _, _, b = _tiny(tmp_path, seed=9, name=os.path.join("run_b", "blueprint_x.bin"))
    assert open(a, "rb").read() != open(b, "rb").read()
    assert identity_of(a, bk_path) != identity_of(b, bk_path)
