"Tests for classify_variables: the free/fixed/linked partition with values (#318)."

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpkit import Model, Var, Variable, VectorVariable
from gpkit.varinfo import VarInfo, VarKind, classify_variables
from gpkit.varkey import VarKey
from gpkit.varmap import VarMap, VarSet

# ---------------------------------------------------------------------------
# Scaffolding: raw keys, so most tests need neither a Model nor a solve
# ---------------------------------------------------------------------------


def vec(name, n):
    "The n element keys of a vector, as a VectorVariable would make them."
    veckey = VarKey(name=name, shape=(n,))
    return tuple(
        VarKey(name=name, shape=(n,), idx=(i,), veckey=veckey) for i in range(n)
    )


def sol_stub(values, sens=None):
    """A stand-in for Solution exposing only what classify_variables may read.

    It has no primal or constants, so a test fails loudly if classification ever
    starts coming from the solution instead of the substitutions.
    """
    return SimpleNamespace(
        variables=VarMap(values),
        sens=SimpleNamespace(variables=VarMap(sens or {})),
    )


def kinds(infos):
    return {info.key.name: info.kind for info in infos}


# ---------------------------------------------------------------------------
# The partition -- these two are the bugs that shipped downstream
# ---------------------------------------------------------------------------


class TestPartition:
    def test_wholly_fixed_vector_is_fixed(self):
        "The shipped bug: no substitution sits on the veckey, only on elements."
        qe = vec("q", 3)
        (info,) = classify_variables(VarSet(qe), {qe[0].veckey: [1.0, 2.0, 3.0]})
        assert info.kind is VarKind.FIXED

    def test_wholly_free_vector_is_free(self):
        (info,) = classify_variables(VarSet(vec("q", 3)), {})
        assert info.kind is VarKind.FREE

    def test_a_vector_yields_exactly_one_row(self):
        "The other shipped bug: an element row, or a blank veckey row, alongside it."
        qe = vec("q", 3)
        infos = classify_variables(VarSet(qe), {})
        assert len(infos) == 1
        assert infos[0].key is qe[0].veckey

    def test_rows_are_user_level_not_solver_level(self):
        qe = vec("q", 4)
        x = VarKey(name="x")
        vks = VarSet([*qe, x])
        assert len(vks) == 5  # four elements plus a scalar
        assert len(classify_variables(vks, {})) == 2  # one vector, one scalar

    def test_scalars(self):
        x, y = VarKey(name="x"), VarKey(name="y")
        infos = classify_variables(VarSet([x, y]), {x: 7.0})
        assert kinds(infos) == {"x": VarKind.FIXED, "y": VarKind.FREE}

    def test_linked_is_its_own_kind(self):
        "A callable substitution is fixed-but-computed, not fixed and not free."
        x, a = VarKey(name="x"), VarKey(name="a")
        infos = classify_variables(VarSet([x, a]), {a: 2.0, x: lambda c: 2 * c[a]})
        assert kinds(infos) == {"a": VarKind.FIXED, "x": VarKind.LINKED}

    def test_partially_fixed_vector_is_mixed(self):
        qe = vec("q", 3)
        (info,) = classify_variables(VarSet(qe), {qe[0]: 1.0})
        assert info.kind is VarKind.MIXED


# ---------------------------------------------------------------------------
# Identity and ordering
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_key_is_the_veckey_for_a_vector(self):
        qe = vec("q", 3)
        (info,) = classify_variables(VarSet(qe), {})
        assert info.key is qe[0].veckey
        assert info.vks == qe

    def test_a_scalar_is_its_own_sole_element(self):
        "So values[i] <-> vks[i] holds with no special case for scalars."
        x = VarKey(name="x")
        (info,) = classify_variables(VarSet([x]), {})
        assert info.vks == (x,)
        assert info.key is info.vks[0]

    def test_elements_are_in_index_order(self):
        'Not ref order: "q[10]" sorts before "q[2]".'
        (info,) = classify_variables(VarSet(vec("q", 12)), {})
        assert [k.idx for k in info.vks] == [(i,) for i in range(12)]

    def test_rows_partition_the_input(self):
        x = VarKey(name="x")
        vks = VarSet([*vec("q", 3), *vec("p", 2), x])
        infos = classify_variables(vks, {})
        seen = [k for info in infos for k in info.vks]
        assert len(seen) == len(set(seen)) == len(vks)
        assert set(seen) == set(vks)

    def test_order_is_deterministic_by_ref(self):
        x, a = VarKey(name="x"), VarKey(name="a")
        infos = classify_variables(VarSet([*vec("q", 2), x, a]), {})
        assert [i.key.name for i in infos] == ["a", "q", "x"]


# ---------------------------------------------------------------------------
# Values and sensitivities
# ---------------------------------------------------------------------------


class TestData:
    def test_free_and_unsolved_has_no_values(self):
        (info,) = classify_variables(VarSet(vec("q", 3)), {})
        assert info.values == (None, None, None)
        assert info.sensitivities == (None, None, None)

    def test_fixed_and_unsolved_takes_values_from_substitutions(self):
        qe = vec("q", 3)
        (info,) = classify_variables(VarSet(qe), {qe[0].veckey: [1.0, 2.0, 3.0]})
        assert info.values == (1.0, 2.0, 3.0)

    def test_values_align_with_vks(self):
        qe = vec("q", 3)
        (info,) = classify_variables(VarSet(qe), {qe[1]: 9.0})
        assert info.values[info.vks.index(qe[1])] == 9.0

    def test_mixed_vector_holes_land_on_the_free_indices(self):
        qe = vec("q", 3)
        (info,) = classify_variables(VarSet(qe), {qe[0]: 1.0, qe[2]: 3.0})
        assert info.values == (1.0, None, 3.0)

    def test_a_solution_supplies_values_and_sensitivities(self):
        x = VarKey(name="x")
        sol = sol_stub({x: 2.5}, {x: -0.5})
        (info,) = classify_variables(VarSet([x]), {x: 2.5}, sol)
        assert (info.values, info.sensitivities) == ((2.5,), (-0.5,))

    def test_missing_sensitivity_is_none(self):
        x = VarKey(name="x")
        (info,) = classify_variables(VarSet([x]), {}, sol_stub({x: 2.5}))
        assert info.sensitivities == (None,)

    def test_values_are_plain_floats(self):
        "parse_subs yields numpy scalars; the IR has to serialize."
        qe = vec("q", 2)
        (info,) = classify_variables(VarSet(qe), {qe[0].veckey: np.array([1.0, 2.0])})
        assert all(type(v) is float for v in info.values)
        json.dumps(info.values)

    def test_substitutions_beat_a_stale_key_value(self):
        "VarKey.value is a declaration-time carrier and goes stale (#321)."
        a = Variable("a", 3, "m")
        vks = VarSet([a.key])
        (info,) = classify_variables(vks, {a.key: 5.0})
        assert info.values == (5.0,)


# ---------------------------------------------------------------------------
# A VarMap's varset is its namespace; _data is what holds values (#55)
# ---------------------------------------------------------------------------


class TestBroaderNamespace:
    def test_a_key_with_no_value_still_gets_a_row(self):
        x, y = VarKey(name="x"), VarKey(name="y")
        subs = VarMap({x: 1.0})
        subs.register_keys(VarSet([x, y]))  # y is in the namespace, has no value
        infos = classify_variables(VarSet([x, y]), subs)
        assert kinds(infos) == {"x": VarKind.FIXED, "y": VarKind.FREE}

    def test_a_real_models_substitutions(self):
        "model.substitutions registers every key while holding only fixed values."
        from gpkit.examples.pipeline import Pipeline  # noqa: PLC0415

        m = Pipeline.default()
        assert len(m.substitutions) < len(m.vks)  # the condition that broke #55
        infos = classify_variables(m.vks, m.substitutions)
        assert infos  # no KeyError


# ---------------------------------------------------------------------------
# Agreement with the authorities that already exist
# ---------------------------------------------------------------------------


class ClassifyMe(Model):
    "Fixed and free, scalar and vector, in one solvable model."

    x = Var("m", "free scalar")
    c = Var("m", "fixed scalar", value=2.0)

    def setup(self):
        self.q = VectorVariable(3, "q", [0.1, 0.2, 0.3], "m", "fixed vector")
        self.v = VectorVariable(3, "v", "m", "free vector")
        self.cost = self.x
        return [
            self.x >= self.q.sum() + self.v.sum() + self.c,
            self.v >= 1e-3 * self.c,
        ]


class TestAgreement:
    def test_free_element_count_matches_n_free(self):
        "n_free is len(vks) - len(substitutions) in costed.py -- one authority."
        m = ClassifyMe()
        infos = classify_variables(m.vks, m.substitutions)
        assert not any(i.kind is VarKind.MIXED for i in infos)
        free = sum(len(i.vks) for i in infos if i.kind is VarKind.FREE)
        assert free == m.n_free

    def test_a_solution_does_not_change_the_partition(self):
        "Solving cannot repartition: primal/constants derive from parse_subs."
        m = ClassifyMe()
        sol = m.solve(verbosity=0)
        before = classify_variables(m.vks, m.substitutions)
        after = classify_variables(m.vks, m.substitutions, sol)
        assert [(i.key, i.vks, i.kind) for i in before] == [
            (i.key, i.vks, i.kind) for i in after
        ]
        assert any(i.values != (None,) * len(i.vks) for i in after)

    def test_classification_matches_the_report(self):
        """Bridge to report.py's _is_free_vk, which this will replace.

        Guards the overlap while both exist, so the migration can delete one.
        """
        m = ClassifyMe()
        sol = m.solve(verbosity=0)
        rep = m.report(solution=sol, fmt="dict")
        reported_free = {v["name"] for v in rep["free_variables"]}
        infos = classify_variables(VarSet(m.own_varkeys), m.substitutions, sol)
        computed_free = {i.key.name for i in infos if i.kind is VarKind.FREE}
        assert {n.replace("[:]", "") for n in reported_free} == computed_free


# ---------------------------------------------------------------------------
# The row stays free of anything surface-dependent
# ---------------------------------------------------------------------------


def test_varinfo_carries_no_rendered_fields():
    """A name, LaTeX or unit string depends on the output surface, so it is the
    renderer's to derive -- see #319. Adding one should be deliberate."""
    assert {f.name for f in dataclasses.fields(VarInfo)} == {
        "key",
        "vks",
        "kind",
        "values",
        "sensitivities",
    }


def test_varinfo_is_immutable():
    x = VarKey(name="x")
    (info,) = classify_variables(VarSet([x]), {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.kind = VarKind.FIXED
