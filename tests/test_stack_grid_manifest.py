"""The depth-grid manifest data/stack_grid_pot16_s0.json (docs/stack_grid_design.md section 3, docs/stack_depth.md)
against the files it names: independent of the StackGridAgent code, so a re-measured matrix, a re-trained point or a
copied buckets file cannot silently break what the wrapper will read.

  * the points: sorted, one per depth, their blueprint and buckets files present;
  * the "boundaries" rule: an inclusive upper depth for every point, inside [its depth, the next point's depth), the
    grid's ends as the below / above points (review 6.5(b));
  * one abstraction for the whole grid: every point's buckets file is a byte copy of the shared bucketer, and every
    blueprint's header says it was trained on that bucketer (fingerprint), at its point's depth, on the manifest's
    bet grid, heads-up.

Only the headers of the blueprints are read (first 64 KiB), never the tables.  Skipped when the manifest or its files
are absent (data/ is a junction to D:; NEGPLURIBUS_TEST_DATA points the tests at another data folder).
"""
from __future__ import annotations

import filecmp
import json
import os
from fractions import Fraction

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("NEGPLURIBUS_TEST_DATA") or os.path.join(ROOT, "data")
MANIFEST = os.path.join(DATA, "stack_grid_pot16_s0.json")

pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason=f"needs {MANIFEST}")


def _resolve(p: str) -> str:
    """Manifest paths are written from the repository root ("data/..."): resolve them next to DATA."""
    if os.path.isabs(p):
        return p
    parts = p.replace("\\", "/").split("/")
    if parts[0] == "data":
        return os.path.join(DATA, *parts[1:])
    return os.path.join(os.path.dirname(DATA), *parts)


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def present(manifest):
    pts = [p for p in manifest["points"] if p.get("present", True)]
    missing = [p["blueprint"] for p in pts if not os.path.exists(_resolve(p["blueprint"]))]
    if missing:
        pytest.skip(f"blueprints of the manifest absent: {missing}")
    return pts


def _header_identity(path: str) -> dict:
    """The game a binary blueprint was saved for, from its header only (layout: negpluribus/fast/binfmt.py)."""
    from negpluribus.fast import binfmt

    with open(path, "rb") as f:
        head = f.read(1 << 16)
    r = binfmt._Reader(head, path)
    assert bytes(r.take(8)) == binfmt.BLUEPRINT_MAGIC, f"{path} is not a binary blueprint"
    assert r.u32() == 1, f"{path}: blueprint format version"
    r.u32()  # flags
    ident = binfmt._identity(r)
    assert ident is not None, f"{path}: no game identity in the header"
    return ident


def test_points_are_sorted_distinct_and_cover_the_design_depths(manifest):
    stacks = [Fraction(p["stack_bb"]) for p in manifest["points"]]
    assert stacks == sorted(stacks) and len(set(stacks)) == len(stacks)
    assert {20, 50, 100, 150, 200} <= {int(s) for s in stacks}  # the depths of steps 1-2 (docs/stack_depth.md)
    assert manifest["bucketer"] and manifest["grid"]["max_raises"] >= 1


def test_boundaries_rule_gives_every_depth_exactly_one_point(manifest):
    rule = manifest["rule"]
    assert rule["kind"] == "boundaries"
    pts = [p for p in manifest["points"] if p.get("present", True)]
    stacks = [Fraction(p["stack_bb"]) for p in pts]
    upper = {Fraction(k): Fraction(v) for k, v in rule["upper"].items()}
    all_stacks = {Fraction(p["stack_bb"]) for p in manifest["points"]}
    assert set(upper) <= all_stacks, f"upper depths for depths that are no points: {set(upper) - all_stacks}"
    assert set(stacks) <= set(upper), f"points without an upper depth: {set(stacks) - set(upper)}"
    for i, s in enumerate(stacks):
        nxt = stacks[i + 1] if i + 1 < len(stacks) else None
        assert s <= upper[s], f"the {s}bb point's upper depth {upper[s]} is below the point"
        if nxt is not None:
            assert upper[s] < nxt, f"the {s}bb point's upper depth {upper[s]} reaches the next point {nxt}"
    assert Fraction(rule.get("below_grid", stacks[0])) == stacks[0]
    assert Fraction(rule.get("above_grid", stacks[-1])) == stacks[-1]
    # the inclusive rule of the design: the first point whose upper depth is >= the effective stack
    def pick(eff):
        if eff < stacks[0]:
            return stacks[0]
        for s in stacks:
            if eff <= upper[s]:
                return s
        return stacks[-1]

    for s in stacks:  # a point's own depth and its upper depth are its own; one chip above goes to the next point
        assert pick(s) == s and pick(upper[s]) == s
        i = stacks.index(s)
        if i + 1 < len(stacks):
            assert pick(upper[s] + Fraction(1, 100)) == stacks[i + 1]
    assert pick(Fraction(1)) == stacks[0] and pick(Fraction(100_000)) == stacks[-1]


def test_every_point_shares_the_one_bucketer_byte_for_byte(manifest, present):
    shared = _resolve(manifest["bucketer"])
    assert os.path.exists(shared)
    for p in present:
        b = p.get("buckets")
        if b is None:
            continue
        assert os.path.exists(_resolve(b)), b
        assert filecmp.cmp(shared, _resolve(b), shallow=False), f"{b} differs from the grid's bucketer {manifest['bucketer']}"


def test_every_blueprint_was_trained_at_its_point_on_the_grid_and_bucketer(manifest, present):
    from negpluribus.abstraction import load_bucketer
    from negpluribus import fast

    grid = manifest["grid"]
    shared_fp = None
    if fast.core() is not None:
        from negpluribus.fast.trainer import core_bucketer

        shared_fp = core_bucketer(load_bucketer(_resolve(manifest["bucketer"])), cache_caps=(0, 0, 0)).identity["fingerprint"]
    fingerprints, schemes = set(), set()
    for p in present:
        path = _resolve(p["blueprint"])
        ident = _header_identity(path)
        where = f"{p['blueprint']} ({p['stack_bb']}bb point)"
        assert ident["n_players"] == 2, where
        assert Fraction(ident["stack_bb"]) == Fraction(p["stack_bb"]), f"{where}: trained at {ident['stack_bb']}bb"
        assert ident["preflop_fracs"] == [float(x) for x in grid["preflop"]], where
        assert ident["postflop_fracs"] == [float(x) for x in grid["postflop"]], where
        assert ident["max_raises_per_street"] == int(grid["max_raises"]), where
        fingerprints.add(ident["bucketer_fingerprint"])
        schemes.add(ident["key_scheme"])
    assert len(fingerprints) == 1, f"the points were trained on {len(fingerprints)} different bucketers"
    assert len(schemes) == 1
    if shared_fp is not None:
        assert fingerprints == {shared_fp}, "the blueprints' bucketer is not the manifest's shared bucketer"
