"""Test growth allowance metadata, helper, and theta singleton."""

import pickle

import pytest

from gpkit import Model, Variable, VarKey
from gpkit.nomials.growth import theta


class TestGrowthVarKeyField:
    """VarKey carries an optional growth-fraction metadata field."""

    def test_growth_kwarg_stored_on_varkey(self):
        vk = VarKey("m", units="kg", growth=0.25)
        assert vk.growth == 0.25

    def test_growth_defaults_to_none(self):
        vk = VarKey("m", units="kg")
        assert vk.growth is None

    def test_growth_survives_pickle(self):
        vk = VarKey("m", units="kg", growth=0.25)
        restored = pickle.loads(pickle.dumps(vk))
        assert restored.growth == 0.25


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


class TestThetaSingleton:
    """A single shared theta variable scales every growth allowance."""

    def test_theta_is_singleton(self):
        a = theta()
        b = theta()
        assert a.key == b.key

    def test_theta_default_value_is_one(self):
        assert theta().key.value == 1.0

    def test_theta_lineage(self):
        assert theta().key.lineage == (("growth", 0),)
        assert theta().key.name == "theta"

    def test_theta_is_dimensionless(self):
        assert theta().key.units is None

    def test_doubling_theta_doubles_allowance(self):
        m = Variable("m", "kg", growth=0.25)
        e = Variable("e", 100, "kg")
        model = Model(m, m.grown_from(e))
        model.substitutions[theta().key] = 2.0
        sol = model.solve(verbosity=0)
        assert sol[m.growth].to("kg").magnitude == pytest.approx(50.0)
        assert sol[m].to("kg").magnitude == pytest.approx(150.0)


class TestGrownFromHelper:
    """m.grown_from(expr) emits the right constraint set."""

    def test_returns_plain_two_constraint_list(self):
        m = Variable("m", "kg", growth=0.25)
        e = Variable("e", 100, "kg")
        cs = m.grown_from(e)
        assert isinstance(cs, list)
        assert len(cs) == 2

    def test_grown_from_without_growth_kwarg_raises(self):
        m = Variable("m", "kg")
        e = Variable("e", 100, "kg")
        with pytest.raises(ValueError, match="growth"):
            m.grown_from(e)

    def test_sibling_names_follow_convention(self):
        m = Variable("m", "kg", growth=0.25)
        assert m.growth.key.name == "m_growth"
        assert m.f_growth.key.name == "f_growth_m"

    def test_sibling_units(self):
        m = Variable("m", "kg", growth=0.25)
        assert m.growth.key.units == m.key.units
        assert m.f_growth.key.units is None

    def test_fraction_carries_growth_value(self):
        m = Variable("m", "kg", growth=0.25)
        assert m.f_growth.key.value == 0.25


class TestSingleLevelSolve:
    """End-to-end solve produces the expected growth-allowance bookkeeping."""

    def test_default_theta_yields_125_percent_total(self):
        m = Variable("m", "kg", growth=0.25)
        e = Variable("e", 100, "kg")
        sol = Model(m, m.grown_from(e)).solve(verbosity=0)
        assert sol[m].to("kg").magnitude == pytest.approx(125.0, rel=1e-4)
        assert sol[m.growth].to("kg").magnitude == pytest.approx(25.0, rel=1e-4)
        assert sol[m.f_growth].magnitude == pytest.approx(0.25)
        assert sol[theta()].magnitude == pytest.approx(1.0)

    def test_matches_hand_rolled_equivalent(self):
        m_auto = Variable("m", "kg", growth=0.25)
        e_auto = Variable("e", 100, "kg")
        sol_auto = Model(m_auto, m_auto.grown_from(e_auto)).solve(verbosity=0)

        m_hand = Variable("m_hand", "kg")
        e_hand = Variable("e_hand", 100, "kg")
        m_growth_hand = Variable("m_growth_hand", "kg")
        f_hand = Variable("f_hand", 0.25)
        sol_hand = Model(
            m_hand,
            [m_growth_hand >= f_hand * e_hand, m_hand >= e_hand + m_growth_hand],
        ).solve(verbosity=0)

        assert sol_auto[m_auto].to("kg").magnitude == pytest.approx(
            sol_hand[m_hand].to("kg").magnitude
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
