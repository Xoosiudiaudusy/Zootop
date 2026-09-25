"""Hand-class index and precomputed bucket tables (csrc/handindex.h, csrc/buckettable.h, fast/tables.py).

(a) the index is Waugh's class count on every street and agrees with canonical_form(): one index
per canonical form, a representative with the same form; (b) a tabulated street gives exactly
bucket()'s numbers, for E[HS] and potential-aware buckets; (c) files round-trip and are refused
by another bucketer; (d) training with a tabulated bucketer is bit-identical to training without.
Only the flop is tabulated here (1.29M classes, a few seconds with small sample counts); turn and
river are the same code on a bigger index (scripts/build_bucket_table.py checks them).
"""
from __future__ import annotations

import os
import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer, PotentialAwareBucketer
from negpluribus.abstraction.canonical import canonical_form
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "BucketTables"), reason="C++ core without bucket tables (rebuild)")

CLASSES = {3: 1_286_792, 4: 13_960_050, 5: 123_156_254}  # Waugh 2013, rounds {2, n}


@pytest.fixture(scope="module")
def ehs():
    return EquityBucketer(n_buckets=8, samples=12).fit(n_situations=200, seed=1)


@pytest.fixture(scope="module")
def pot():
    return PotentialAwareBucketer(n_buckets=8, samples=3, bins=10).fit(n_situations=120, seed=3)


def _core(bk):
    from negpluribus.fast.trainer import core_bucketer

    return core_bucketer(bk, cache_caps=(0, 0, 0))


@pytest.mark.parametrize("n_board", [3, 4, 5])
def test_index_counts_and_canonical_forms(n_board):
    ix = core.HandIndexer(n_board)
    assert ix.size == CLASSES[n_board]
    rng = random.Random(n_board)
    for _ in range(3000):
        cards = rng.sample(range(52), 2 + n_board)
        hole, board = cards[:2], cards[2:]
        i = ix.index(hole, board)
        assert 0 <= i < ix.size
        rh, rb = ix.unindex(i)
        assert ix.index(rh, rb) == i
        assert canonical_form(rh, rb) == canonical_form(hole, board)


def test_index_separates_forms():
    """Different canonical forms never share an index (random pairs on the flop)."""
    ix = core.HandIndexer(3)
    seen = {}
    rng = random.Random(7)
    for _ in range(20000):
        cards = rng.sample(range(52), 5)
        form = canonical_form(cards[:2], cards[2:])
        i = ix.index(cards[:2], cards[2:])
        assert seen.setdefault(i, form) == form


@pytest.fixture(scope="module")
def flop_tables(ehs, pot):
    out = {}
    for name, bk in (("ehs", ehs), ("pot", pot)):
        cbk = _core(bk)
        t = core.BucketTables()
        t.build(cbk, 1, threads=fast.default_threads())
        out[name] = (cbk, t)
    return out


@pytest.mark.parametrize("name", ["ehs", "pot"])
def test_flop_table_equals_bucket(flop_tables, name):
    cbk, t = flop_tables[name]
    assert t.has(1) and not t.has(2) and not t.has(3)
    rng = random.Random(11)
    for _ in range(5000):
        cards = rng.sample(range(52), 5)
        assert t.lookup(cards[:2], cards[2:]) == cbk.bucket(cards[:2], cards[2:])
    tb = core.TabulatedBucketer(cbk, t)
    assert tb.identity == cbk.identity
    for _ in range(2000):  # tabulated street from the table, the others from the wrapped bucketer
        for n in (3, 4, 5):
            cards = rng.sample(range(52), 2 + n)
            assert tb.bucket(cards[:2], cards[2:]) == cbk.bucket(cards[:2], cards[2:])


def test_file_roundtrip_and_identity_check(tmp_path, flop_tables):
    cbk, t = flop_tables["ehs"]
    other, _ = flop_tables["pot"]
    p = str(tmp_path / "t.npbt")
    t.save(p)
    u = core.BucketTables()
    u.load(p, cbk)
    rng = random.Random(3)
    for _ in range(2000):
        cards = rng.sample(range(52), 5)
        assert u.lookup(cards[:2], cards[2:]) == t.lookup(cards[:2], cards[2:])
    with pytest.raises(Exception):
        core.BucketTables().load(p, other)
    with pytest.raises(Exception):
        core.TabulatedBucketer(other, u)
    with open(p, "r+b") as f:  # a flipped byte in the flop table fails the checksum
        f.seek(200)
        b = f.read(1)
        f.seek(200)
        f.write(bytes([b[0] ^ 1]))
    with pytest.raises(Exception):
        core.BucketTables().load(p, cbk)


def test_training_with_table_is_bit_identical(tmp_path, monkeypatch, ehs, flop_tables):
    from negpluribus.fast.tables import table_name

    spec = GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=8)
    monkeypatch.delenv("NEGPLURIBUS_BUCKET_TABLES", raising=False)
    plain = MCCFRTrainer(spec, ehs, seed=5, backend="cpp", threads=1)
    plain.train(3000)
    cbk, t = flop_tables["ehs"]
    t.save(os.path.join(str(tmp_path), table_name(cbk)))
    monkeypatch.setenv("NEGPLURIBUS_BUCKET_TABLES", str(tmp_path))
    tab = MCCFRTrainer(spec, ehs, seed=5, backend="cpp", threads=1)
    assert type(tab._core_bucketer).__name__ == "TabulatedBucketer"
    tab.train(3000)
    a, b = plain.nodes.snapshot(), tab.nodes.snapshot()
    assert a.keys() == b.keys()
    for k in a:
        assert (a[k].regret, a[k].strategy_sum, a[k].visits) == (b[k].regret, b[k].strategy_sum, b[k].visits), k
