"""Test growth allowance metadata on VarKey and Variable."""

import pickle

import pytest

from gpkit import Variable, VarKey


class TestGrowthVarKeyField:
    """VarKey carries an optional growth-fraction metadata field."""

    def test_growth_kwarg_stored_on_varkey(self):
        vk = VarKey("m", units="kg", growth=0.25)
        assert vk.growth == 0.25

    def test_growth_defaults_to_none(self):
        vk = VarKey("m", units="kg")
        assert vk.growth is None


class TestGrowthDoesNotAffectIdentity:
    """Growth metadata is not part of canonical identity (ref/hash/eq)."""

    def test_same_name_different_growth_are_equal(self):
        a = VarKey("m", units="kg", growth=0.25)
        b = VarKey("m", units="kg", growth=0.30)
        assert a == b
        assert hash(a) == hash(b)

    def test_growth_does_not_appear_in_ref(self):
        vk = VarKey("m", units="kg", growth=0.25)
        assert "growth" not in vk.ref
        assert "0.25" not in vk.ref


class TestGrowthKwargOnVariable:
    """Variable accepts growth=... and threads it to its VarKey."""

    def test_variable_accepts_growth_kwarg(self):
        m = Variable("m", "kg", growth=0.25)
        assert m.key.growth == 0.25

    def test_variable_without_growth_has_none(self):
        m = Variable("m", "kg")
        assert m.key.growth is None


class TestSiblingDedup:
    """Sibling Variables constructed twice yield equal VarKeys."""

    def test_sibling_construction_idempotent(self):
        m = Variable("m", "kg", growth=0.25)
        a = Variable("m_growth", units="kg", lineage=m.key.lineage)
        b = Variable("m_growth", units="kg", lineage=m.key.lineage)
        assert a.key == b.key
        assert hash(a.key) == hash(b.key)
        assert a.key.ref == b.key.ref

    def test_distinct_parents_yield_distinct_siblings(self):
        m = Variable("m", "kg", growth=0.25)
        p = Variable("P", "W", growth=0.10)
        m_growth = Variable("m_growth", units="kg", lineage=m.key.lineage)
        p_growth = Variable("P_growth", units="W", lineage=p.key.lineage)
        assert m_growth.key != p_growth.key


class TestPickleRoundTrip:
    """Growth metadata survives pickle round-trip."""

    def test_growth_survives_pickle(self):
        vk = VarKey("m", units="kg", growth=0.25)
        restored = pickle.loads(pickle.dumps(vk))
        assert restored.growth == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
