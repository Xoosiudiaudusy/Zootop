"""The counters scripts/eval_archetypes.py prints and logs (the check files of the search experiments are its output):
"off-map" for every hero, and for the search hero the summary line with "off the map" and "search errors N (fallback
to the blueprint)"; --log-hands keeps each decision's "played", "off_map" and "error".  An experiment with a non-zero
error counter is not counted (handoff 28.09), so the counter must reach the output.

main() runs in-process on a tiny HU 30bb game (fixed E[HS] cut points, a 1,500-iteration C++ blueprint), one search
thread, a few deals against the random bettor; the failing searches are simulated (every _search raises, as solve()
does when its node table cannot grow: tests/test_search_oom.py).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "SubgameSearch"), reason="C++ core with the search not built")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street

    d = tmp_path_factory.mktemp("eval_counters")
    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=4, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    bk.save(str(d / "buckets.json"))
    MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=1).train(1500).save_blueprint(str(d / "blueprint.bin"))
    return d


def _main(monkeypatch, capsys, files, *flags):
    from negpluribus.agents import core_search

    # the search agent's bucket caches at their production size take ~0.6 GB on their first inserts: small ones here
    load = core_search.SearchResources.load.__func__
    monkeypatch.setattr(core_search.SearchResources, "load",
                        classmethod(lambda cls, *a, **kw: load(cls, *a, **{**kw, "cache_caps": (1 << 16,) * 3})))
    spec = importlib.util.spec_from_file_location("eval_archetypes_under_test", os.path.join(ROOT, "scripts", "eval_archetypes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["eval_archetypes.py", "--spec", "2p_30bb_river", "--preflop-fracs", "1.0", "--postflop-fracs", "0.5,1.0",
            "--max-raises", "2", "--blueprint", str(files / "blueprint.bin"), "--buckets", str(files / "buckets.json"),
            "--opponents", "random", "--seed", "3", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    return capsys.readouterr().out


def _hands(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def test_blueprint_hero_reports_its_off_map_rate(monkeypatch, capsys, files):
    out = _main(monkeypatch, capsys, files, "--deals", "20")
    m = re.search(r"vs\s+random: .*off-map (\d+\.\d)%", out)
    assert m, out


def test_search_hero_summary_counts_failed_searches_and_the_hand_log_keeps_them(monkeypatch, capsys, files, tmp_path):
    from negpluribus.agents.core_search import CoreSearchAgent

    def failing(self, obs, reason):
        raise RuntimeError("search failed: out of memory growing the node table")

    monkeypatch.setattr(CoreSearchAgent, "_search", failing)
    log = tmp_path / "hands.jsonl"
    out = _main(monkeypatch, capsys, files, "--deals", "4", "--agent", "search", "--search-iterations", "30",
                "--search-threads", "1", "--no-luck", "--log-hands", str(log))
    m = re.search(r"off the map (\d+) \((\d+\.\d\d)%\); search errors (\d+) \(fallback to the blueprint\)", out)
    assert m, out
    errors = int(m.group(3))
    hands = _hands(log)
    assert len(hands) == 8
    failed = [d for h in hands for d in h["hero"] if d.get("played") == "blueprint after a search error"]
    assert errors == len(failed) > 0
    assert all(d["error"] == "RuntimeError: search failed: out of memory growing the node table" and "off_map" in d
               for d in failed)
    assert "errors: RuntimeError: search failed" in out
    assert int(m.group(1)) == sum(1 for h in hands for d in h["hero"] if d.get("off_map"))


def test_search_hero_without_failures_reports_zero_errors(monkeypatch, capsys, files, tmp_path):
    log = tmp_path / "hands.jsonl"
    out = _main(monkeypatch, capsys, files, "--deals", "3", "--agent", "search", "--search-iterations", "30",
                "--search-threads", "1", "--no-luck", "--log-hands", str(log))
    assert re.search(r"search errors 0 \(fallback to the blueprint\)", out), out
    hands = _hands(log)
    played = [d.get("played") for h in hands for d in h["hero"]]
    assert played and "blueprint after a search error" not in played
    postflop = [d for h in hands for d in h["hero"] if d.get("street", 0) > 0]
    assert all(d.get("played") == "search" for d in postflop)
