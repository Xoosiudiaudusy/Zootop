"""Evaluation tools must evaluate the agent that played: regressions for the defect hunter's H3, M6, M7 (xfail(strict)
while the defect is in master).

  * H3: AIVAT's known player is whatever --blueprint aivat_eval.py is given.  The hand log of eval_archetypes.py
    --log-hands names no blueprint (file, iteration or checksum) for a non-grid hero, so a log played by blueprint A
    is scored with blueprint B without a word (the root-cache side of it: tests/test_aivat_root_cache.py, QA-4).
  * M6: eval_archetypes.py and aivat_eval.py build the game from their command line (defaults: 3 raises per street,
    the production fracs) and never compare it with the identity stored in a binary blueprint (CppBlueprint.identity):
    a blueprint trained with 2 raises per street is played / modelled on a 3-raise grid, which differs only at the
    cap, silently.
  * M7: hand logs are not validated: aivat_eval.py scores a log with a duplicated (deal, seat) line as two hands, and
    aivat_fast.summarize(pairs=...) divides every deal by the LARGEST hands-per-deal count, so an incomplete deal is
    halved.

Tiny HU 30bb game (fixed E[HS] cut points, 1-thread C++ blueprints), main() of both scripts run in-process; in
aivat_eval.py the bucket tables and the root table (minutes on a real abstraction) are stubbed, and the run stops
where the root table would be built: a refusal must come before that.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "AivatGame") or not hasattr(core, "SubgameSearch"),
                                reason="C++ core with AIVAT and the search not built")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAME_FLAGS = ["--preflop-fracs", "1.0", "--postflop-fracs", "0.5,1.0", "--max-raises", "2"]


class _ReachedTheRootTable(Exception):
    pass


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test_identity", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """buckets.json; blueprint_a.bin (3000 iterations) and blueprint_b.bin (6000) of the same game, max 2 raises."""
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street

    d = tmp_path_factory.mktemp("eval_identity")
    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    bk.save(str(d / "buckets.json"))
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=4, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    t = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=1).train(3000)
    t.save_blueprint(str(d / "blueprint_a.bin"))
    t.train(3000)
    t.save_blueprint(str(d / "blueprint_b.bin"))
    return d


def _eval_archetypes(monkeypatch, capsys, *argv):
    mod = _load("eval_archetypes")
    monkeypatch.setattr(sys, "argv", ["eval_archetypes.py", "--spec", "2p_30bb_river", *argv])
    mod.main()
    return capsys.readouterr().out


@pytest.fixture(scope="module")
def log_of_a(files, tmp_path_factory):
    """A duplicate duel log (--log-hands) played by blueprint A as the hero against the random bettor."""
    import _pytest.monkeypatch

    mp = _pytest.monkeypatch.MonkeyPatch()
    log = files / "hands_a.jsonl"
    try:
        mod = _load("eval_archetypes")
        mp.setattr(sys, "argv", ["eval_archetypes.py", "--spec", "2p_30bb_river", *GAME_FLAGS,
                                 "--blueprint", str(files / "blueprint_a.bin"), "--buckets", str(files / "buckets.json"),
                                 "--opponents", "random", "--deals", "10", "--seed", "4", "--luck", "--log-hands", str(log)])
        mod.main()
    finally:
        mp.undo()
    rows = [json.loads(ln) for ln in open(log, encoding="utf-8") if ln.strip()]
    assert len(rows) == 20
    return log


@pytest.fixture()
def aivat(tmp_path, monkeypatch):
    """aivat(log, blueprint, *flags): runs aivat_eval.py main() on a duel log until the root table; returns
    ("root", identity) when it got there, ("refused", message) when it stopped before."""
    mod = _load("aivat_eval")
    monkeypatch.setattr(core, "aivat_build_tables", lambda cbk, threads, streets: (core.BucketTables(), "stub"))
    monkeypatch.setattr(core, "TabulatedBucketer", lambda cbk, tables: cbk)

    def capture(game, rollouts, seed=0, threads=1, cache_path=None, identity=None):
        raise _ReachedTheRootTable(identity)

    monkeypatch.setattr(mod, "root_table", capture)
    calls = [0]

    def run(log, blueprint, buckets, *flags):
        calls[0] += 1
        argv = ["aivat_eval.py", "--duel", str(log), "--out", str(tmp_path / f"out{calls[0]}.jsonl"), "--blueprint", str(blueprint),
                "--buckets", str(buckets), "--tables", str(tmp_path / f"tables{calls[0]}"), "--threads", "1",
                "--stack-bb", "30", "--no-luck", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        try:
            mod.main()
        except _ReachedTheRootTable as e:
            return "root", e.args[0]
        except (SystemExit, ValueError, RuntimeError) as e:
            return "refused", str(e)
        return "finished", None

    return run


def test_the_logged_blueprint_is_scored(files, log_of_a, aivat):
    """Control: the log's own blueprint and game reach the scoring."""
    kind, _ = aivat(log_of_a, files / "blueprint_a.bin", files / "buckets.json", *GAME_FLAGS)
    assert kind == "root"


@pytest.mark.xfail(strict=True, reason="H3: eval_archetypes --log-hands names no blueprint for a non-grid hero and "
                                        "aivat_eval.py models the known player with whatever --blueprint it is given")
def test_a_log_is_not_scored_with_another_blueprint(files, log_of_a, aivat):
    row = json.loads(open(log_of_a, encoding="utf-8").readline())
    assert any(k in row for k in ("blueprint", "hero_blueprint")) or any("blueprint" in d for d in row["hero"]), row.keys()
    kind, msg = aivat(log_of_a, files / "blueprint_b.bin", files / "buckets.json", *GAME_FLAGS)
    assert kind == "refused", msg


@pytest.mark.xfail(strict=True, reason="M6: aivat_eval.py builds the game from its flags (default max 3 raises) and never "
                                        "compares it with the blueprint's stored identity (2 raises here)")
def test_aivat_eval_refuses_a_grid_other_than_the_blueprints(files, log_of_a, aivat):
    flags = ["--preflop-fracs", "1.0", "--postflop-fracs", "0.5,1.0"]  # --max-raises left at its default 3
    kind, msg = aivat(log_of_a, files / "blueprint_a.bin", files / "buckets.json", *flags)
    assert kind == "refused" and "raise" in msg.lower(), (kind, msg)


@pytest.mark.xfail(strict=True, reason="M6: eval_archetypes.py plays a blueprint on the grid of its command line (default "
                                        "max 3 raises) without comparing it with the blueprint's stored identity")
def test_eval_archetypes_refuses_a_grid_other_than_the_blueprints(files, monkeypatch, capsys):
    from negpluribus.fast.blueprint import load_blueprint

    assert load_blueprint(str(files / "blueprint_a.bin")).identity["max_raises_per_street"] == 2
    with pytest.raises((SystemExit, ValueError)):
        _eval_archetypes(monkeypatch, capsys, "--preflop-fracs", "1.0", "--postflop-fracs", "0.5,1.0",
                         "--blueprint", str(files / "blueprint_a.bin"), "--buckets", str(files / "buckets.json"),
                         "--opponents", "random", "--deals", "2")


@pytest.mark.xfail(strict=True, reason="M7: aivat_eval.py load_hands does not refuse a duplicated (deal, seat) line")
def test_a_log_with_a_duplicated_hand_is_refused(files, log_of_a, aivat, tmp_path):
    lines = [ln for ln in open(log_of_a, encoding="utf-8") if ln.strip()]
    dup = tmp_path / "dup.jsonl"
    dup.write_text("".join(lines + lines[:1]), encoding="utf-8")  # deal 0, the first seat, twice
    kind, msg = aivat(dup, files / "blueprint_a.bin", files / "buckets.json", *GAME_FLAGS)
    assert kind == "refused", msg


@pytest.mark.xfail(strict=True, reason="M7: negpluribus/eval/aivat_fast.py summarize(pairs=...) divides every deal by the "
                                        "largest hands-per-deal count, not by its own")
def test_summarize_averages_each_deal_over_its_own_hands():
    from negpluribus.eval.aivat_fast import summarize

    results = [{"net": 0.0, "value": v} for v in (100.0, 300.0, 50.0)]  # deal 0: two hands, deal 1: one (incomplete)
    s = summarize(results, per=100.0, pairs=[0, 0, 1])["aivat"]
    assert s["n"] == 2 and s["bb100"] == pytest.approx(100 * (2.0 + 0.5) / 2)  # deal means 2.0 bb and 0.5 bb
