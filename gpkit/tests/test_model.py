"""Tests for GP and SP classes"""

import sys
from io import StringIO

import numpy as np
import pytest

from gpkit import (
    ArrayVariable,
    Model,
    NamedVariables,
    SignomialEquality,
    SignomialsEnabled,
    Variable,
    VectorVariable,
    parse_variables,
    settings,
    units,
)
from gpkit.constraints.bounded import Bounded
from gpkit.constraints.relax import (
    ConstantsRelaxed,
    ConstraintsRelaxed,
    ConstraintsRelaxedEqually,
)
from gpkit.exceptions import (
    DualInfeasible,
    InvalidGPConstraint,
    InvalidPosynomial,
    PrimalInfeasible,
    UnboundedGP,
    UnknownInfeasible,
    UnnecessarySGP,
)
from gpkit.util.small_classes import CootMatrix

NDIGS = {"cvxopt": 5, "mosek_cli": 5, "mosek_conif": 3}
# name: decimal places of accuracy achieved in these tests

# pylint: disable=invalid-name,attribute-defined-outside-init
# pylint: disable=unused-variable,undefined-variable,exec-used


def get_ndig(solver_name):
    """Get the number of decimal places for a given solver"""
    return NDIGS.get(solver_name, 5)  # default to 5 if solver not found


class TestGP:
    """
    Test GeometricPrograms.
    This TestCase gets run once for each installed solver via the solver fixture.
    """

    def test_no_monomial_constraints(self, solver):
        x = Variable("x")
        ndig = get_ndig(solver)
        sol = Model(x, [x + 1 / x <= 3]).solve(solver=solver, verbosity=0)
        assert sol.cost == pytest.approx(0.381966, abs=10 ** (-ndig))

    def test_trivial_gp(self, solver):
        """
        Create and solve a trivial GP:
            minimize    x + 2y
            subject to  xy >= 1

        The global optimum is (x, y) = (sqrt(2), 1/sqrt(2)).
        """
        x = Variable("x")
        y = Variable("y")
        prob = Model(cost=(x + 2 * y), constraints=[x * y >= 1])
        sol = prob.solve(solver=solver, verbosity=0)
        assert isinstance(prob.latex(), str)
        # pylint: disable=protected-access
        assert isinstance(prob._repr_latex_(), str)
        assert sol["x"] == pytest.approx(np.sqrt(2.0), abs=10 ** (-get_ndig(solver)))
        assert sol["y"] == pytest.approx(
            1 / np.sqrt(2.0), abs=10 ** (-get_ndig(solver))
        )
        assert sol["x"] + 2 * sol["y"] == pytest.approx(
            2 * np.sqrt(2), abs=10 ** (-get_ndig(solver))
        )
        assert sol.cost == pytest.approx(2 * np.sqrt(2), abs=10 ** (-get_ndig(solver)))

    def test_dup_eq_constraint(self, solver):
        # from https://github.com/convexengineering/gpkit/issues/1551
        a = Variable("a", 1)
        b = Variable("b")
        c = Variable("c", 2)
        d = Variable("d")
        z = Variable("z", 0.5)

        # create a simple GP with equality constraints
        const = [
            z == b / a,
            z == d / c,
        ]

        # simple cost
        cost = a + b + c + d

        # create a model
        m = Model(cost, const)

        # solve the first version of the model (solves successfully)
        m.solve(verbosity=0, solver=solver)

        # add a redundant equality constraint
        m.extend([z == b / a])

        # solver will fail and attempt to debug
        m.solve(verbosity=0, solver=solver)

    def test_sigeq(self, solver):
        x = Variable("x")
        y = VectorVariable(1, "y")
        c = Variable("c")
        # test left vector input to sigeq
        with SignomialsEnabled():
            m = Model(
                c,
                [c >= (x + 0.25) ** 2 + (y - 0.5) ** 2, SignomialEquality(x**2 + x, y)],
            )
        sol = m.localsolve(solver=solver, verbosity=0)
        assert sol["x"] == pytest.approx(0.1639472, abs=10 ** (-get_ndig(solver)))
        assert sol["y"][0] == pytest.approx(0.1908254, abs=10 ** (-get_ndig(solver)))
        assert sol["c"] == pytest.approx(0.2669448, abs=10 ** (-get_ndig(solver)))
        # test right vector input to sigeq
        with SignomialsEnabled():
            m = Model(
                c,
                [c >= (x + 0.25) ** 2 + (y - 0.5) ** 2, SignomialEquality(y, x**2 + x)],
            )
        sol = m.localsolve(solver=solver, verbosity=0)
        assert sol["x"] == pytest.approx(0.1639472, abs=10 ** (-get_ndig(solver)))
        assert sol["y"][0] == pytest.approx(0.1908254, abs=10 ** (-get_ndig(solver)))
        assert sol["c"] == pytest.approx(0.2669448, abs=10 ** (-get_ndig(solver)))
        # test scalar input to sigeq
        z = Variable("z")
        with SignomialsEnabled():
            m = Model(
                c,
                [c >= (x + 0.25) ** 2 + (z - 0.5) ** 2, SignomialEquality(x**2 + x, z)],
            )
        sol = m.localsolve(solver=solver, verbosity=0)
        assert sol["x"] == pytest.approx(0.1639472, abs=10 ** (-get_ndig(solver)))
        assert sol["z"] == pytest.approx(0.1908254, abs=10 ** (-get_ndig(solver)))
        assert sol["c"] == pytest.approx(0.2669448, abs=10 ** (-get_ndig(solver)))

    def test_601(self, solver):
        # tautological monomials should solve but not pass to the solver
        x = Variable("x")
        y = Variable("y", 2)
        m = Model(x, [x >= 1, y == 2])
        m.solve(solver=solver, verbosity=0)
        assert len(list(m.as_hmapslt1({}))) == 3
        assert len(m.program.hmaps) == 2

    def test_cost_freeing(self, solver):
        "Test freeing a variable that's in the cost."
        x = Variable("x", 1)
        x_min = Variable("x_{min}", 2)
        intermediary = Variable("intermediary")
        m = Model(x, [x >= intermediary, intermediary >= x_min])
        with pytest.raises((PrimalInfeasible, UnknownInfeasible)):
            m.solve(solver=solver, verbosity=0)
        x = Variable("x", 1)
        x_min = Variable("x_{min}", 2)
        m = Model(x, [x >= x_min])
        with pytest.raises(PrimalInfeasible):
            m.solve(solver=solver, verbosity=0)
        del m.substitutions[m["x"]]
        assert m.solve(solver=solver, verbosity=0).cost == pytest.approx(2)
        del m.substitutions[m["x_{min}"]]
        with pytest.raises(UnboundedGP):
            m.solve(solver=solver, verbosity=0)
        gp = m.gp(checkbounds=False)
        with pytest.raises(DualInfeasible):
            gp.solve(solver=solver, verbosity=0)

    def test_simple_united_gp(self, solver):
        R = Variable("R", "nautical_miles")
        a0 = Variable("a0", 340.29, "m/s")
        theta = Variable("\\theta", 0.7598)
        t = Variable("t", 10, "hr")
        T_loiter = Variable("T_{loiter}", 1, "hr")
        T_reserve = Variable("T_{reserve}", 45, "min")
        M = VectorVariable(2, "M")

        prob = Model(
            1 / R, [t >= sum(R / a0 / M / theta**0.5) + T_loiter + T_reserve, M <= 0.76]
        )
        sol = prob.solve(solver=solver, verbosity=0)
        assert 0.000553226 / sol.cost == pytest.approx(1, abs=10 ** (-get_ndig(solver)))
        assert 340.29 / sol.constants["a0"] == pytest.approx(
            1, abs=10 ** (-get_ndig(solver))
        )
        assert 340.29 * a0.units / sol["a0"] == pytest.approx(
            1, abs=10 ** (-get_ndig(solver))
        )
        assert 1807.58 / sol.primal["R"] == pytest.approx(
            1, abs=10 ** (-get_ndig(solver))
        )
        assert 1807.58 * R.units / sol["R"] == pytest.approx(
            1, abs=10 ** (-get_ndig(solver))
        )

    def test_trivial_vector_gp(self, solver):
        "Create and solve a trivial GP with VectorVariables"
        x = VectorVariable(2, "x")
        y = VectorVariable(2, "y")
        prob = Model(cost=(sum(x) + 2 * sum(y)), constraints=[x * y >= 1])
        sol = prob.solve(solver=solver, verbosity=0)
        assert sol["x"].shape == (2,)
        assert sol["y"].shape == (2,)
        for x, y in zip(sol["x"], sol["y"]):
            assert x == pytest.approx(np.sqrt(2.0), abs=10 ** (-get_ndig(solver)))
            assert y == pytest.approx(1 / np.sqrt(2.0), abs=10 ** (-get_ndig(solver)))
        assert sol.cost / (4 * np.sqrt(2)) == pytest.approx(
            1.0, abs=10 ** (-get_ndig(solver))
        )

    def test_sensitivities(self, solver):
        W_payload = Variable("W_{payload}", 175 * (195 + 30), "lbf")
        f_oew = Variable("f_{oew}", 0.53, "-", "OEW/MTOW")
        fuel_per_nm = Variable("\\theta_{fuel}", 13.75, "lbf/nautical_mile")
        R = Variable("R", 3000, "nautical_miles", "range")
        mtow = Variable("MTOW", "lbf", "max take off weight")

        m = Model(
            61.3e6 * units.USD * (mtow / (1e5 * units.lbf)) ** 0.807,
            [mtow >= W_payload + f_oew * mtow + fuel_per_nm * R],
        )
        sol = m.solve(solver=solver, verbosity=0)
        senss = sol.sens.variables
        assert senss[f_oew] == pytest.approx(0.91, abs=0.01)
        assert senss[R] == pytest.approx(0.41, abs=0.01)
        assert senss[fuel_per_nm] == pytest.approx(0.41, abs=0.01)
        assert senss[W_payload] == pytest.approx(0.39, abs=0.01)

    def test_mdd_example(self, solver):
        Cl = Variable("Cl", 0.5, "-", "Lift Coefficient")
        Mdd = Variable("Mdd", "-", "Drag Divergence Mach Number")
        m1 = Model(1 / Mdd, [1 >= 5 * Mdd + 0.5, Mdd >= 0.00001])
        m2 = Model(1 / Mdd, [1 >= 5 * Mdd + 0.5])
        m3 = Model(1 / Mdd, [1 >= 5 * Mdd + Cl, Mdd >= 0.00001])
        sol1 = m1.solve(solver=solver, verbosity=0)
        sol2 = m2.solve(solver=solver, verbosity=0)
        sol3 = m3.solve(solver=solver, verbosity=0)
        # pylint: disable=no-member
        gp1, gp2, gp3 = [m.program for m in [m1, m2, m3]]
        assert gp1.data.A == CootMatrix(row=[0, 1, 2], col=[0, 0, 0], data=[-1, 1, -1])
        assert gp2.data.A == CootMatrix(row=[0, 1], col=[0, 0], data=[-1, 1])
        assert gp3.data.A == CootMatrix(row=[0, 1, 2], col=[0, 0, 0], data=[-1, 1, -1])
        assert (gp3.data.A.todense() == np.array([[-1, 1, -1]]).T).all()
        assert sol1[Mdd] == pytest.approx(sol2[Mdd])
        assert sol1[Mdd] == pytest.approx(sol3[Mdd])
        assert sol2[Mdd] == pytest.approx(sol3[Mdd])

    def test_additive_constants(self, solver):
        x = Variable("x")
        m = Model(1 / x, [1 >= 5 * x + 0.5, 1 >= 5 * x])
        m.solve(solver=solver, verbosity=0)
        # pylint: disable=no-member
        gp = m.program  # created by solve()
        assert gp.data.c[1] == 2 * gp.data.c[2]
        assert gp.data.A.data[1] == gp.data.A.data[2]

    def test_zeroing(self, solver):
        L = Variable("L")
        k = Variable("k", 0)
        with SignomialsEnabled():
            constr = [L - 5 * k <= 10]
        sol = Model(1 / L, constr).solve(solver, verbosity=0)
        assert sol[L] == pytest.approx(10, abs=10 ** (-get_ndig(solver)))
        assert sol.cost == pytest.approx(0.1, abs=10 ** (-get_ndig(solver)))
        assert sol.almost_equal(sol)

    @pytest.mark.parametrize(
        "solver", [s for s in settings["installed_solvers"] if s != "cvxopt"]
    )
    def test_singular(self, solver):
        """Create and solve GP with a singular A matrix.

        cvxopt cannot solve singular problems.
        """
        x = Variable("x")
        y = Variable("y")
        m = Model(y * x, [y * x >= 12])
        sol = m.solve(solver=solver, verbosity=0)
        assert sol.cost == pytest.approx(12, abs=10 ** (-get_ndig(solver)))

    def test_constants_in_objective_1(self, solver):
        "Issue 296"
        x1 = Variable("x1")
        x2 = Variable("x2")
        m = Model(1.0 + x1 + x2, [x1 >= 1.0, x2 >= 1.0])
        sol = m.solve(solver=solver, verbosity=0)
        assert sol.cost == pytest.approx(3, abs=10 ** (-get_ndig(solver)))

    def test_constants_in_objective_2(self, solver):
        "Issue 296"
        x1 = Variable("x1")
        x2 = Variable("x2")
        m = Model(x1**2 + 100 + 3 * x2, [x1 >= 10.0, x2 >= 15.0])
        sol = m.solve(solver=solver, verbosity=0)
        assert sol.cost / 245.0 == pytest.approx(1, abs=10 ** (-get_ndig(solver)))

    def test_terminating_constant_(self, solver):
        x = Variable("x")
        y = Variable("y", value=0.5)
        prob = Model(1 / x, [x + y <= 4])
        sol = prob.solve(verbosity=0)
        assert sol.cost == pytest.approx(1 / 3.5, abs=10 ** (-get_ndig(solver)))

    def test_exps_is_tuple(self, solver):
        "issue 407"
        x = Variable("x")
        m = Model(x, [x >= 1])
        m.solve(solver=solver, verbosity=0)
        assert isinstance(m.program.cost.exps, tuple)

    def test_sweep_not_modify_subs(self, solver):
        "make sure sweeping does not modify substitutions"
        A = Variable("A", "ft^2")
        d = Variable("d", 6, "in")
        m = Model(A, [A >= np.pi / 4 * d**2])
        assert m.substitutions == {d.key: 6}
        sol = m.sweep({d: [3, 12, 36]}, solver=solver, verbosity=0)
        assert sol[1][A] / A.units == pytest.approx(0.785398165)
        assert m.substitutions == {d.key: 6}

    @pytest.mark.parametrize(
        "solver", [s for s in settings["installed_solvers"] if s == "cvxopt"]
    )
    def test_cvxopt_kwargs(self, solver):
        """Test that kwargs can be passed to cvxopt solver."""
        x = Variable("x")
        m = Model(x, [x >= 12])
        sol = m.solve(solver=solver, verbosity=0, kktsolver="ldl")
        assert sol.cost == pytest.approx(12.0, abs=10 ** (-get_ndig(solver)))

    def test_printing_with_asymmetric_lineage_depth(self, solver):
        """Regression: sol.table() raised IndexError when the same model class
        appears at two different lineage depths in the same solution.

        NamedVariables.modelnums is keyed by (parent_lineage, class_name), so
        the same class in two distinct parent contexts each gets number 0:
          - standalone Inner() → lineage (("Inner", 0),) → lineagestr "Inner0"
          - Outer's Inner()   → lineage (("Outer", 0), ("Inner", 0)) → "Outer0.Inner0"
        Both produce "Inner0" at depth 1, causing a name collision. When
        _compute_collision_depths tried depth 2, lineages[-2] on the 1-element
        standalone list raised IndexError.
        """

        class _Inner(Model):
            def setup(self):
                v = Variable("v")
                return [v >= 1]

        class _Outer(Model):
            def setup(self):
                inner = _Inner()
                return [inner]

        outer = _Outer()
        inner_standalone = _Inner()
        m = Model(outer["v"] + inner_standalone["v"], [outer, inner_standalone])
        sol = m.solve(solver=solver, verbosity=0)
        tab = sol.table()  # Must not raise IndexError
        assert isinstance(tab, str)


class TestSP:
    "test case for SP class -- run for each installed solver via solver fixture"

    def test_sp_relaxation(self, solver):
        w = Variable("w")
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        with SignomialsEnabled():
            m = Model(x, [x + y >= w, x + y <= z / 2, y <= x, y >= 1], {z: 3, w: 3})
        r1 = ConstantsRelaxed(m)
        assert len(r1.vks) == 8
        with pytest.raises(ValueError):
            _ = Model(x * r1.relaxvars, r1)  # no "prod"
        sp = Model(x * r1.relaxvars.prod() ** 10, r1).sp(use_pccp=False)
        cost = sp.localsolve(verbosity=0, solver=solver).cost
        assert cost / 1024 == pytest.approx(1, abs=10 ** (-get_ndig(solver)))
        m.debug(verbosity=0, solver=solver)
        with SignomialsEnabled():
            m = Model(x, [x + y >= z, x + y <= z / 2, y <= x, y >= 1], {z: 3})
        m.debug(verbosity=0, solver=solver)
        r2 = ConstraintsRelaxed(m)
        assert len(r2.vks) == 7
        sp = Model(x * r2.relaxvars.prod() ** 10, r2).sp(use_pccp=False)
        cost = sp.localsolve(verbosity=0, solver=solver).cost
        assert cost / 1024 == pytest.approx(1, abs=10 ** (-get_ndig(solver)))
        with SignomialsEnabled():
            m = Model(x, [x + y >= z, x + y <= z / 2, y <= x, y >= 1], {z: 3})
        m.debug(verbosity=0, solver=solver)
        r3 = ConstraintsRelaxedEqually(m)
        assert len(r3.vks) == 4
        sp = Model(x * r3.relaxvar**10, r3).sp(use_pccp=False)
        cost = sp.localsolve(verbosity=0, solver=solver).cost
        assert cost / (32 * 0.8786796585) == pytest.approx(
            1, abs=10 ** (-get_ndig(solver))
        )

    def test_sp_bounded(self, solver):
        x = Variable("x")
        y = Variable("y")

        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.1])  # solves
        cost = m.localsolve(verbosity=0, solver=solver).cost
        assert cost == pytest.approx(0.9, abs=10 ** (-get_ndig(solver)))

        with SignomialsEnabled():
            m = Model(x, [x + y >= 1])  # dual infeasible
        with pytest.raises(UnboundedGP):
            m.localsolve(verbosity=0, solver=solver)
        gp = m.sp(checkbounds=False).gp()
        with pytest.raises(DualInfeasible):
            gp.solve(solver=solver, verbosity=0)

        with SignomialsEnabled():
            m = Model(x, Bounded([x + y >= 1]))
        sol = m.localsolve(verbosity=0, solver=solver)
        boundedness = sol.meta["boundedness"]
        # depends on solver, platform, whims of the numerical deities
        if "value near lower bound of 1e-30" in boundedness:  # pragma: no cover
            assert x.key in boundedness["value near lower bound of 1e-30"]
        else:  # pragma: no cover
            assert y.key in boundedness["value near upper bound of 1e+30"]

    def test_values_vs_subs(self, solver):
        # Substitutions update method
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        with SignomialsEnabled():
            constraints = [x + y >= z, y >= x - 1]
        m = Model(x + y * z, constraints)
        m.substitutions.update({"z": 5})
        sol = m.localsolve(verbosity=0, solver=solver)
        assert sol.cost == pytest.approx(13, abs=10 ** (-get_ndig(solver)))

        # Constant variable declaration method
        z = Variable("z", 5)
        with SignomialsEnabled():
            constraints = [x + y >= z, y >= x - 1]
        m = Model(x + y * z, constraints)
        sol = m.localsolve(verbosity=0, solver=solver)
        assert sol.cost == pytest.approx(13, abs=10 ** (-get_ndig(solver)))

    def test_initially_infeasible(self, solver):
        x = Variable("x")
        y = Variable("y")

        with SignomialsEnabled():
            sigc = x >= y + y**2 - y**3
            sigc2 = x <= y**0.5

        m = Model(1 / x, [sigc, sigc2, y <= 0.5])

        sol = m.localsolve(verbosity=0, solver=solver)
        assert sol.cost == pytest.approx(2**0.5, abs=10 ** (-get_ndig(solver)))
        assert sol[y] == pytest.approx(0.5, abs=10 ** (-get_ndig(solver)))

    def test_sp_substitutions(self, solver):
        x = Variable("x")
        y = Variable("y", 1)
        z = Variable("z", 4)

        old_stdout = sys.stdout
        sys.stdout = stringout = StringIO()

        with SignomialsEnabled():
            m1 = Model(x, [x + z >= y])
        with pytest.raises(UnnecessarySGP):
            m1.localsolve(verbosity=0, solver=solver)
        with pytest.raises(UnboundedGP):
            m1.solve(verbosity=0, solver=solver)

        with SignomialsEnabled():
            m2 = Model(x, [x + y >= z])
            m2.substitutions[y] = 1
            m2.substitutions[z] = 4
        sol = m2.solve(solver, verbosity=0)
        assert sol.cost == pytest.approx(3, abs=10 ** (-get_ndig(solver)))

        sys.stdout = old_stdout
        assert stringout.getvalue() == (
            f"Warning: SignomialConstraint {str(m1[0])} became the "
            "tautological constraint 0 <= 3 + x after substitution.\n"
            f"Warning: SignomialConstraint {str(m1[0])} became the "
            "tautological constraint 0 <= 3 + x after substitution.\n"
        )

    def test_tautological(self, solver):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        old_stdout = sys.stdout
        sys.stdout = stringout = StringIO()

        with SignomialsEnabled():
            m1 = Model(x, [x + y >= z, x >= y])
            m2 = Model(x, [x + 1 >= 0, x >= y])
        m1.substitutions.update({"z": 0, "y": 1})
        m2.substitutions.update({"y": 1})
        assert m1.solve(solver, verbosity=0).cost == pytest.approx(
            m2.solve(solver, verbosity=0).cost
        )

        sys.stdout = old_stdout
        assert stringout.getvalue() == (
            f"Warning: SignomialConstraint {str(m1[0])} became the "
            "tautological constraint 0 <= 1 + x after substitution.\n"
            f"Warning: SignomialConstraint {str(m2[0])} became the "
            "tautological constraint 0 <= 1 + x after substitution.\n"
        )

    def test_impossible(self, solver):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        with SignomialsEnabled():
            m1 = Model(x, [x + y >= z, x >= y])
        m1.substitutions.update({"x": 0, "y": 0})
        with pytest.raises(ValueError):
            _ = m1.localsolve(solver=solver)

    def test_trivial_sp(self, solver):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(x, [x >= 1 - y, y <= 0.1])
        with pytest.raises(InvalidGPConstraint):
            m.solve(verbosity=0, solver=solver)
        sol = m.localsolve(solver, verbosity=0)
        assert sol.primal["x"] == pytest.approx(0.9, abs=10 ** (-get_ndig(solver)))
        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.1])
        sol = m.localsolve(solver, verbosity=0)
        assert sol.primal["x"] == pytest.approx(0.9, abs=10 ** (-get_ndig(solver)))

    def test_tautological_spconstraint(self, solver):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z", 0)
        with SignomialsEnabled():
            m = Model(x, [x >= 1 - y, y <= 0.1, y >= z])
        with pytest.raises(InvalidGPConstraint):
            m.solve(verbosity=0, solver=solver)
        sol = m.localsolve(solver, verbosity=0)
        assert sol.primal["x"] == pytest.approx(0.9, abs=10 ** (-get_ndig(solver)))

    def test_relaxation(self, solver):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        m = Model(objective, constraints)
        m.localsolve(verbosity=0, solver=solver)

        # issue #257

        A = VectorVariable(2, "A")
        B = ArrayVariable([2, 2], "B")
        C = VectorVariable(2, "C")
        with SignomialsEnabled():
            constraints = [A <= B.dot(C), B <= 1, C <= 1]
        obj = 1 / A[0] + 1 / A[1]
        m = Model(obj, constraints)
        m.localsolve(verbosity=0, solver=solver)

    def test_issue180(self, solver):
        L = Variable("L")
        Lmax = Variable("L_{max}", 10)
        W = Variable("W")
        Wmax = Variable("W_{max}", 10)
        A = Variable("A", 10)
        Obj = Variable("Obj")
        a_val = 0.01
        a = Variable("a", a_val)
        with SignomialsEnabled():
            eqns = [
                L <= Lmax,
                W <= Wmax,
                L * W >= A,
                Obj >= a * (2 * L + 2 * W) + (1 - a) * (12 * W**-1 * L**-3),
            ]
        m = Model(Obj, eqns)
        spsol = m.solve(solver, verbosity=0)
        # now solve as GP
        m[-1] = Obj >= a_val * (2 * L + 2 * W) + (1 - a_val) * (12 * W**-1 * L**-3)
        del m.substitutions[m["a"]]
        gpsol = m.solve(solver, verbosity=0)
        assert spsol.cost == pytest.approx(gpsol.cost)

    def test_trivial_sp2(self, solver):
        x = Variable("x")
        y = Variable("y")

        # converging from above
        with SignomialsEnabled():
            constraints = [y + x >= 2, y >= x]
        objective = y
        x0 = 1
        y0 = 2
        m = Model(objective, constraints)
        sol1 = m.localsolve(x0={x: x0, y: y0}, verbosity=0, solver=solver)

        # converging from right
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        x0 = 2
        y0 = 1
        m = Model(objective, constraints)
        sol2 = m.localsolve(x0={x: x0, y: y0}, verbosity=0, solver=solver)

        assert sol1.primal["x"] == pytest.approx(
            sol2.primal["x"], abs=10 ** (-get_ndig(solver))
        )
        assert sol1.primal["y"] == pytest.approx(
            sol2.primal["x"], abs=10 ** (-get_ndig(solver))
        )

    def test_sp_initial_guess_sub(self, solver):
        x = Variable("x")
        y = Variable("y")
        x0 = 3
        y0 = 2
        with SignomialsEnabled():
            constraints = [y + x >= 4, y <= x]
        objective = x
        m = Model(objective, constraints)
        # Call to local solve with only variables
        sol = m.localsolve(x0={"x": x0, y: y0}, verbosity=0, solver=solver)
        assert sol[x] == pytest.approx(2, abs=10 ** (-get_ndig(solver)))
        assert sol.cost == pytest.approx(2, abs=10 ** (-get_ndig(solver)))

        # Call to local solve with only variable strings
        sol = m.localsolve(x0={"x": x0, "y": y0}, verbosity=0, solver=solver)
        assert sol["x"] == pytest.approx(2, abs=10 ** (-get_ndig(solver)))
        assert sol.cost == pytest.approx(2, abs=10 ** (-get_ndig(solver)))

        # Call to local solve with a mix of variable strings and variables
        sol = m.localsolve(x0={"x": x0, y: y0}, verbosity=0, solver=solver)
        assert sol.cost == pytest.approx(2, abs=10 ** (-get_ndig(solver)))

    def test_small_named_signomial(self, solver):
        x = Variable("x")
        z = Variable("z")
        local_ndig = 4
        nonzero_adder = 0.1
        with SignomialsEnabled():
            J = 0.01 * (x - 1) ** 2 + nonzero_adder
            with NamedVariables("SmallSignomial"):
                m = Model(z, [z >= J])
        sol = m.localsolve(verbosity=0, solver=solver)
        assert sol.cost == pytest.approx(nonzero_adder, abs=10 ** (-local_ndig))
        assert sol["x"] == pytest.approx(0.98725425, abs=10 ** (-get_ndig(solver)))

    def test_sigs_not_allowed_in_cost(self, solver):
        with SignomialsEnabled():
            x = Variable("x")
            y = Variable("y")
            J = 0.01 * ((x - 1) ** 2 + (y - 1) ** 2) + (x * y - 1) ** 2
            m = Model(J)
            with pytest.raises(InvalidPosynomial):
                m.localsolve(verbosity=0, solver=solver)

    def test_becomes_signomial(self, solver):
        "Test that a GP does not become an SP after substitutions"
        x = Variable("x")
        c = Variable("c")
        y = Variable("y")
        m = Model(x, [y >= 1 + c * x, y <= 0.5], {c: -1})
        with pytest.raises(InvalidGPConstraint):
            with SignomialsEnabled():
                m.gp()
        with pytest.raises(UnnecessarySGP):
            m.localsolve(solver=solver)

    def test_reassigned_constant_cost(self, solver):
        # for issue 1131
        x = Variable("x")
        x_min = Variable("x_min", 1)
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(y, [y + 0.5 >= x, x >= x_min, 6 >= y])
        m.localsolve(verbosity=0, solver=solver)
        del m.substitutions[x_min]
        m.cost = 1 / x_min
        assert x_min not in m.sp().gp().substitutions  # pylint: disable=no-member

    def test_unbounded_debugging(self, solver):
        "Test nearly-dual-feasible problems"
        x = Variable("x")
        y = Variable("y")
        m = Model(x * y, [x * y**1.01 >= 100])
        with pytest.raises((DualInfeasible, UnknownInfeasible)):
            m.solve(solver, verbosity=0)
        # test one-sided bound
        m = Model(x * y, Bounded(m, lower=0.001))
        sol = m.solve(solver, verbosity=0)
        bounds = sol.meta["boundedness"]
        assert bounds["sensitive to lower bound of 0.001"] == set([x.key])
        # end test one-sided bound
        m = Model(x * y, [x * y**1.01 >= 100])
        m = Model(x * y, Bounded(m))
        sol = m.solve(solver, verbosity=0)
        bounds = sol.meta["boundedness"]
        # depends on solver, platform, whims of the numerical deities
        if "sensitive to upper bound of 1e+30" in bounds:  # pragma: no cover
            assert y.key in bounds["sensitive to upper bound of 1e+30"]
        else:  # pragma: no cover
            assert x.key in bounds["sensitive to lower bound of 1e-30"]


class Thing(Model):
    "a thing, for model testing"

    def setup(self, length):
        a = self.a = VectorVariable(length, "a", "g/m")
        b = self.b = VectorVariable(length, "b", "m")
        c = Variable("c", 17 / 4.0, "g")
        return [a >= c / b]


class Thing2(Model):
    "another thing for model testing"

    def setup(self):
        return [Thing(2), Model()]


class Box(Model):
    """simple box for model testing

    Variables
    ---------
    h  [m]     height
    w  [m]     width
    d  [m]     depth
    V  [m**3]  volume

    Upper Unbounded
    ---------------
    w, d, h

    Lower Unbounded
    ---------------
    w, d, h
    """

    @parse_variables(__doc__, globals())
    def setup(self):
        return [V == h * w * d]


class BoxAreaBounds(Model):
    """for testing functionality of separate analysis models

    Lower Unbounded
    ---------------
    h, d, w
    """

    def setup(self, box):
        A_wall = Variable("A_{wall}", 100, "m^2", "Upper limit, wall area")
        A_floor = Variable("A_{floor}", 50, "m^2", "Upper limit, floor area")

        self.h, self.d, self.w = box.h, box.d, box.w

        return [
            2 * box.h * box.w + 2 * box.h * box.d <= A_wall,
            box.w * box.d <= A_floor,
        ]


class Sub(Model):
    "Submodel with mass, for testing"

    def setup(self):
        m = Variable("m", "lb", "mass")  # noqa: F841


class Widget(Model):
    "A model with two Sub models"

    def setup(self):
        m_tot = Variable("m_{tot}", "lb", "total mass")
        self.subA = Sub()
        self.subB = Sub()
        return [self.subA, self.subB, m_tot >= self.subA["m"] + self.subB["m"]]


class TestModelNoSolve:
    "model tests that don't require a solver"

    def test_modelname_added(self):
        t = Thing(2)
        for vk in t.vks:
            assert vk.lineage == (("Thing", 0),)

    def test_modelcontainmentprinting(self):
        t = Thing2()
        assert t["c"].key.models == ("Thing2", "Thing")
        assert isinstance(t.str_without(), str)
        assert isinstance(t.latex(), str)

    def test_no_naming_on_var_access(self):
        # make sure that analysis models don't add their names to
        # variables looked up from other models
        box = Box()
        area_bounds = BoxAreaBounds(box)
        M = Model(box["V"], [box, area_bounds])
        for var in ("h", "w", "d"):
            assert len(M.varkeys.by_name(var)) == 1

    def test_duplicate_submodel_varnames(self):
        w = Widget()
        # w has two Sub models, both with their own variable m
        assert len(w.varkeys.by_name("m")) == 2
        # keys for both submodel m's should be in the parent model varkeys
        assert w.subA["m"].key in w.varkeys
        assert w.subB["m"].key in w.varkeys
        # keys of w.variables_byname("m") should match m.varkeys
        m_vbn_keys = w.varkeys.by_name("m")
        assert w.subA["m"].key in m_vbn_keys
        assert w.subB["m"].key in m_vbn_keys
        # dig a level deeper, into the keymap
        assert len(w.varkeys.keys("m")) == 2

    def test_posy_simplification(self):
        "issue 525"
        D = Variable("D")
        mi = Variable("m_i")
        V = Variable("V", 1)
        m1 = Model(D + V, [V >= mi + 0.4, mi >= 0.1, D >= mi**2])
        m2 = Model(D + 1, [1 >= mi + 0.4, mi >= 0.1, D >= mi**2])
        gp1 = m1.gp()
        gp2 = m2.gp()
        # pylint: disable=no-member
        assert gp1.data.A == gp2.data.A
        assert gp1.data.c == gp2.data.c

    def test_partial_sub_signomial(self):
        "Test SP partial x0 initialization"
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.5])
        gp = m.sp().gp(x0={x: 0.5})  # pylint: disable=no-member
        (first_gp_constr_posy_exp,) = gp.hmaps[1]  # first after cost
        assert first_gp_constr_posy_exp[x.key] == -1.0 / 3

    def test_verify_docstring_constant_not_flagged_as_unbounded(self):
        """Regression: verify_docstring raised ValueError for an inherited constant.

        ConstraintSet.__init__ intentionally skips adding bounds for constants
        that have lineage AND are not in unique_varkeys (i.e., inherited from a
        parent model). Such constants end up in self.substitutions and varkeys,
        but NOT in bounded. The old count shortcut then fired:
            len(bounded) + len(missingbounds) != 2 * len(self.varkeys)
        and incorrectly added the inherited constant to missingbounds → ValueError.
        The fix skips keys in self.substitutions in the missing-bounds loop.
        """

        class _Child(Model):
            """Model with an inherited constant — should not need bound for rho.

            Upper Unbounded
            ---------------
            x
            """

            def setup(self, rho):
                x = self.x = Variable("x")
                # rho has lineage from _Parent's context and is not in
                # _Child.unique_varkeys, so ConstraintSet skips adding its bounds.
                return [x >= rho]

        class _Parent(Model):
            """SKIP VERIFICATION"""

            def setup(self):
                rho = Variable("rho", 1.225)
                self.child = _Child(rho)
                return [self.child]

        # Must construct without ValueError about inherited constant rho
        m = _Parent()
        assert m is not None
        assert "rho" in m.child.substitutions
