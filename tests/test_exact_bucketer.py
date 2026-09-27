"""Potential-aware buckets with exact flop / turn features (PotentialAwareBucketer(exact=True)).

exact=False changes nothing (fingerprint, file, numbers); exact=True is another bucketer whose
Python and C++ buckets agree; files round-trip the flag.  Full tables are checked by
scripts/build_bucket_table.py (bucket() on random hands against the table).
"""
from __future__ import annotations

import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import PotentialAwareBucketer, load_bucketer

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "exact_feature_many"), reason="C++ core without exact features (rebuild)")


def _core(bk):
    from negpluribus.fast.trainer import core_bucketer

    return core_bucketer(bk, cache_caps=(0, 0, 0))


@pytest.fixture(scope="module")
def mc():
    return PotentialAwareBucketer(n_buckets=8, samples=6, bins=10).fit(n_situations=120, seed=3)


@pytest.fixture(scope="module")
def ex():
    return PotentialAwareBucketer(n_buckets=8, samples=6, bins=10, exact=True).fit(n_situations=120, seed=3)


def test_off_changes_nothing(mc, tmp_path):
    assert "exact" not in mc.to_dict()
    same = PotentialAwareBucketer.from_dict(mc.to_dict())
    assert not same.exact and _core(same).identity == _core(mc).identity
    assert not _core(mc).exact


def test_exact_is_another_bucketer_and_round_trips(mc, ex, tmp_path):
    p = str(tmp_path / "b.json")
    ex.save(p)
    back = load_bucketer(p)
    assert back.exact and back.to_dict() == ex.to_dict()
    # the same centroids with and without the flag are two different bucketers
    twin = PotentialAwareBucketer.from_dict({**ex.to_dict(), "exact": False})
    assert _core(twin).identity["fingerprint"] != _core(ex).identity["fingerprint"]


def test_python_and_cpp_agree(ex):
    cbk = _core(ex)
    assert cbk.exact
    rng = random.Random(9)
    for n_board, n in ((3, 6), (4, 60), (5, 60)):
        for _ in range(n):
            cards = rng.sample(range(52), 2 + n_board)
            assert ex.bucket(cards[:2], cards[2:]) == cbk.bucket(cards[:2], cards[2:])


def test_exact_features_are_used(ex):
    cards = [0, 5, 10, 20, 30, 40]
    counts, _ = core.exact_feature(cards[:2], cards[2:], 10)
    from negpluribus.abstraction.canonical import canonical_key

    key = canonical_key(cards[:2], cards[2:])
    want, _ = core.exact_feature(list(key[:2]), list(key[3:]), 10)
    assert counts == want  # suit-invariant
    assert ex._histogram(list(key[:2]), list(key[3:]))[0] == want


def test_fit_uses_the_same_situations_and_river(mc, ex):
    """Same seed and n_situations: the same random deals (flop / turn feature keys), the same k-means
    code and the same river cut points; only the flop / turn feature function differs."""
    assert set(mc._features) == set(ex._features)
    assert mc.boundaries == ex.boundaries
    assert mc.n_buckets == ex.n_buckets and mc.bins == ex.bins
