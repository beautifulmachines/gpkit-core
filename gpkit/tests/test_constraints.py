"Unit tests for Constraint, MonomialEquality and SignomialInequality"

import numpy as np
import pytest

from gpkit import (
    ConstraintSet,
    Model,
    Posynomial,
    SignomialsEnabled,
    Variable,
    VectorVariable,
)
from gpkit.constraints.bounded import Bounded
from gpkit.constraints.costed import CostedConstraintSet
from gpkit.constraints.loose import Loose
from gpkit.constraints.relax import (
    ConstantsRelaxed,
    ConstraintsRelaxed,
    ConstraintsRelaxedEqually,
)
from gpkit.constraints.tight import Tight
from gpkit.exceptions import InvalidGPConstraint, PrimalInfeasible
from gpkit.globals import NamedVariables
from gpkit.nomials import MonomialEquality, PosynomialInequality, SignomialInequality
from gpkit.units import DimensionalityError


class TestConstraint:
    "Tests for Constraint class"

    def test_uninited_element(self):
        x = Variable("x")

        class SelfPass(Model):
            "A model which contains itself!"

            def setup(self):
                ConstraintSet([self, x <= 1])

        with pytest.raises(ValueError):
            SelfPass()

    def test_bad_elements(self):
        x = Variable("x")
        with pytest.raises(ValueError):
            _ = Model(x, [x == "A"])
        with pytest.raises(ValueError):
            _ = Model(x, [x >= 1, x == "A"])
        with pytest.raises(ValueError):
            _ = Model(
                x,
                [
                    x >= 1,
                    x == "A",
                    x >= 1,
                ],
            )
        with pytest.raises(ValueError):
            _ = Model(x, [x == "A", x >= 1])
        v = VectorVariable(2, "v")
        with pytest.raises(ValueError):
            _ = Model(x, [v == "A"])
        with pytest.raises(TypeError):
            _ = Model(x, [v <= ["A", "B"]])
        with pytest.raises(TypeError):
            _ = Model(x, [v >= ["A", "B"]])

    def test_evalfn(self):
        x = Variable("x")
        x2 = Variable("x^2", evalfn=lambda solv: solv[x] ** 2)
        m = Model(x, [x >= 2])
        m.unique_varkeys = set([x2.key])
        sol = m.solve(verbosity=0)
        assert sol[x2] == pytest.approx(sol[x] ** 2)

    def test_relax_list(self):
        x = Variable("x")
        x_max = Variable("x_max", 1)
        x_min = Variable("x_min", 2)
        constraints = [x_min <= x, x <= x_max]
        ConstraintsRelaxed(constraints)
        ConstantsRelaxed(constraints)
        ConstraintsRelaxedEqually(constraints)

    def test_relax_linked(self):
        x = Variable("x")
        x_max = Variable("x_max", 1)
        x_min = Variable("x_min", lambda c: 2 * c[x_max])
        zero = Variable("zero", lambda c: 0 * c[x_max])
        constraints = ConstraintSet([x_min + zero <= x, x + zero <= x_max])
        _ = ConstantsRelaxed(constraints)
        NamedVariables.reset_modelnumbers()
        include_min = ConstantsRelaxed(constraints, include_only=["x_min"])
        NamedVariables.reset_modelnumbers()
        exclude_max = ConstantsRelaxed(constraints, exclude=["x_max"])
        assert str(include_min) == str(exclude_max)

    def test_equality_relaxation(self):
        x = Variable("x")
        m = Model(x, [x == 3, x == 4])
        rc = ConstraintsRelaxed(m)
        m2 = Model(rc.relaxvars.prod() * x**0.01, rc)
        assert m2.solve(verbosity=0)[x] == pytest.approx(3, abs=1e-3)

    def test_constraintget(self):
        x = Variable("x")
        x_ = Variable("x", lineage=[("_", 0)])
        xv = VectorVariable(2, "x")
        xv_ = VectorVariable(2, "x", lineage=[("_", 0)])
        assert Model(x, [x >= 1])["x"] == x
        with pytest.raises(ValueError):
            _ = Model(x, [x >= 1, x_ >= 1])["x"]
        with pytest.raises(ValueError):
            _ = Model(x, [x >= 1, xv >= 1])["x"]
        assert all(Model(xv.prod(), [xv >= 1])["x"] == xv)
        with pytest.raises(ValueError):
            _ = Model(xv.prod(), [xv >= 1, xv_ >= 1])["x"]
        with pytest.raises(ValueError):
            _ = Model(xv.prod(), [xv >= 1, x_ >= 1])["x"]

    def test_additive_scalar(self):
        "Make sure additive scalars simplify properly"
        x = Variable("x")
        c1 = 1 >= 10 * x
        c2 = 1 >= 5 * x + 0.5
        assert isinstance(c1, PosynomialInequality)
        assert isinstance(c2, PosynomialInequality)
        (c1hmap,) = c1.as_hmapslt1({})
        (c2hmap,) = c2.as_hmapslt1({})
        assert c1hmap == c2hmap

    def test_additive_scalar_gt1(self):
        "1 can't be greater than (1 + something positive)"
        x = Variable("x")

        with pytest.raises(PrimalInfeasible):
            _ = 1 >= 5 * x + 1.1

    def test_init(self):
        "Test Constraint __init__"
        x = Variable("x")
        y = Variable("y")
        c = PosynomialInequality(x, ">=", y**2)
        assert c.as_hmapslt1({}) == [(y**2 / x).hmap]
        assert c.left == x
        assert c.right == y**2
        c = PosynomialInequality(x, "<=", y**2)
        assert c.as_hmapslt1({}) == [(x / y**2).hmap]
        assert c.left == x
        assert c.right == y**2
        assert isinstance((1 >= x).latex(), str)

    def test_oper_overload(self):
        "Test Constraint initialization by operator overloading"
        x = Variable("x")
        y = Variable("y")
        c = y >= 1 + x**2
        assert c.as_hmapslt1({}) == [(1 / y + x**2 / y).hmap]
        assert c.left == y
        assert c.right == 1 + x**2
        # same constraint, switched operator direction
        c2 = 1 + x**2 <= y  # same as c
        assert c2.as_hmapslt1({}) == c.as_hmapslt1({})

    def test_sub_tol(self):
        "Test PosyIneq feasibility tolerance under substitutions"
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        PosynomialInequality.feastol = 1e-5
        m = Model(z, [x == z, x >= y], {x: 1, y: 1.0001})
        with pytest.raises(PrimalInfeasible):
            m.solve(verbosity=0)
        PosynomialInequality.feastol = 1e-3
        assert m.substitutions["x"] == m.solve(verbosity=0)["x"]


class TestCostedConstraint:
    "Tests for Costed Constraint class"

    def test_vector_cost(self):
        x = VectorVariable(2, "x")
        with pytest.raises(ValueError):
            CostedConstraintSet(x, [])
        _ = CostedConstraintSet(np.array(x[0]), [])

    def test_cost(self):
        v = Variable("v")
        assert CostedConstraintSet(v, []).cost == v


class TestMonomialEquality:
    "Test monomial equality constraint class"

    def test_init(self):
        "Test initialization via both operator overloading and __init__"
        x = Variable("x")
        y = Variable("y")
        mono = y**2 / x
        # operator overloading
        mec = x == y**2
        # __init__
        mec2 = MonomialEquality(x, y**2)
        assert mono.hmap in mec.as_hmapslt1({})
        assert mono.hmap in mec2.as_hmapslt1({})
        x = Variable("x", "ft")
        y = Variable("y")
        with pytest.raises(DimensionalityError):
            MonomialEquality(x, y)
        with pytest.raises(DimensionalityError):
            MonomialEquality(y, x)

    def test_vector(self):
        "Monomial Equalities with VectorVariables"
        x = VectorVariable(3, "x")
        assert not x == 3  # pylint: disable=unnecessary-negation
        assert x == x  # pylint: disable=comparison-with-itself

    def test_inheritance(self):
        "Make sure MonomialEquality inherits from the right things"
        f = Variable("f")
        m = Variable("m")
        a = Variable("a")
        mec = f == m * a
        assert isinstance(mec, MonomialEquality)

    def test_non_monomial(self):
        "Try to initialize a MonomialEquality with non-monomial args"
        x = Variable("x")
        y = Variable("y")

        with pytest.raises(TypeError):
            MonomialEquality(x * y, x + y)

    def test_str(self):
        "Test that MonomialEquality.__str__ returns a string"
        x = Variable("x")
        y = Variable("y")
        mec = x == y
        assert isinstance(mec.str_without(), str)

    def test_united_dimensionless(self):
        "Check dimensionless unit-ed variables work"
        x = Variable("x")
        y = Variable("y", "hr/day")
        c = MonomialEquality(x, y)
        assert isinstance(c, MonomialEquality)


class TestSignomialInequality:
    "Test Signomial constraints"

    def test_becomes_posy_sensitivities(self):
        # pylint: disable=invalid-name
        # model from #1165
        ujet = Variable("ujet")
        PK = Variable("PK")
        Dp = Variable("Dp", 0.662)
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1 / 0.7376)
        with SignomialsEnabled():
            m = Model(
                PK,
                [
                    mdot * ujet + fBLI * Dp >= 1,
                    PK >= 0.5 * mdot * ujet * (2 + ujet) + fBLI * fsurf * Dp,
                ],
            )
        var_senss = m.solve(verbosity=0).sens.variables
        assert var_senss[Dp] == pytest.approx(-0.16, abs=1e-2)
        assert var_senss[fBLI] == pytest.approx(-0.16, abs=1e-2)
        assert var_senss[fsurf] == pytest.approx(0.19, abs=1e-2)
        assert var_senss[mdot] == pytest.approx(-0.17, abs=1e-2)

        # Linked variable
        Dp = Variable("Dp", 0.662)
        mDp = Variable("-Dp", lambda c: -c[Dp])
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1 / 0.7376)
        m = Model(
            PK,
            [
                mdot * ujet >= 1 + fBLI * mDp,
                PK >= 0.5 * mdot * ujet * (2 + ujet) + fBLI * fsurf * Dp,
            ],
        )
        var_senss = m.solve(verbosity=0).sens.variables
        assert var_senss[Dp] == pytest.approx(-0.16, abs=1e-2)
        assert var_senss[fBLI] == pytest.approx(-0.16, abs=1e-2)
        assert var_senss[fsurf] == pytest.approx(0.19, abs=1e-2)
        assert var_senss[mdot] == pytest.approx(-0.17, abs=1e-2)

        # fixed negative variable
        Dp = Variable("Dp", 0.662)
        mDp = Variable("-Dp", -0.662)
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1 / 0.7376)
        m = Model(
            PK,
            [
                mdot * ujet >= 1 + fBLI * mDp,
                PK >= 0.5 * mdot * ujet * (2 + ujet) + fBLI * fsurf * Dp,
            ],
        )
        var_senss = m.solve(verbosity=0).sens.variables
        assert var_senss[Dp] + var_senss[mDp] == pytest.approx(-0.16, abs=1e-2)
        assert var_senss[fBLI] == pytest.approx(-0.16, abs=1e-2)
        assert var_senss[fsurf] == pytest.approx(0.19, abs=1e-2)
        assert var_senss[mdot] == pytest.approx(-0.17, abs=1e-2)

    def test_init(self):
        "Test initialization and types"
        drag = Variable("drag", units="N")
        x1, x2, x3 = (Variable(f"x_{i}", units="N") for i in range(3))
        with pytest.raises(TypeError):
            sc = drag >= x1 + x2 - x3
        with SignomialsEnabled():
            sc = drag >= x1 + x2 - x3
        assert isinstance(sc, SignomialInequality)
        assert not isinstance(sc, Posynomial)

    def test_posyslt1(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            sc = x + y >= x * y
        # make sure that the error type doesn't change on our users
        with pytest.raises(InvalidGPConstraint):
            _ = sc.as_hmapslt1({})


class TestLoose:
    "Test loose constraint set"

    def test_raiseerror(self):
        x = Variable("x")
        x_min = Variable("x_{min}", 2)
        m = Model(x, [Loose([x >= x_min]), x >= 1])
        Loose.raiseerror = True
        with pytest.raises(RuntimeWarning):
            m.solve(verbosity=0)
        Loose.raiseerror = False

    def test_posyconstr_in_gp(self):
        "Tests loose constraint set with solve()"
        x = Variable("x")
        x_min = Variable("x_{min}", 2)
        m = Model(x, [Loose([x >= x_min]), x >= 1])
        sol = m.solve(verbosity=0)
        warndata = sol.meta["warnings"]["Unexpectedly Tight Constraints"][0][1]
        assert warndata[-1] is m[0][0]
        assert warndata[0] == pytest.approx(+1, abs=1e-3)
        m.substitutions[x_min] = 0.5
        assert m.solve(verbosity=0).cost == pytest.approx(1)

    def test_posyconstr_in_sp(self):
        x = Variable("x")
        y = Variable("y")
        x_min = Variable("x_min", 1)
        y_min = Variable("y_min", 2)
        with SignomialsEnabled():
            sig_constraint = x + y >= 3.5
        m = Model(x * y, [Loose([x >= y]), x >= x_min, y >= y_min, sig_constraint])
        sol = m.localsolve(verbosity=0)
        warndata = sol.meta["warnings"]["Unexpectedly Tight Constraints"][0][1]
        assert warndata[-1] is m[0][0]
        assert warndata[0] == pytest.approx(+1, abs=1e-3)
        m.substitutions[x_min] = 2
        m.substitutions[y_min] = 1
        assert m.localsolve(verbosity=0).cost == pytest.approx(2.5, abs=1e-5)


class TestTight:
    "Test tight constraint set"

    def test_posyconstr_in_gp(self):
        "Tests tight constraint set with solve()"
        x = Variable("x")
        x_min = Variable("x_{min}", 2)
        m = Model(x, [Tight([x >= 1]), x >= x_min])
        sol = m.solve(verbosity=0)
        warndata = sol.meta["warnings"]["Unexpectedly Loose Constraints"][0][1]
        assert warndata[-1] is m[0][0]
        assert warndata[0] == pytest.approx(1, abs=1e-3)
        m.substitutions[x_min] = 0.5
        assert m.solve(verbosity=0).cost == pytest.approx(1)

    def test_posyconstr_in_sp(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            sig_constraint = x + y >= 0.1
        m = Model(x * y, [Tight([x >= y]), x >= 2, y >= 1, sig_constraint])
        sol = m.localsolve(verbosity=0)
        warndata = sol.meta["warnings"]["Unexpectedly Loose Constraints"][0][1]
        assert warndata[-1] is m[0][0]
        assert warndata[0] == pytest.approx(1, abs=1e-3)
        m.pop(1)
        assert m.localsolve(verbosity=0).cost == pytest.approx(1, abs=1e-5)

    def test_sigconstr_in_sp(self):
        "Tests tight constraint set with localsolve()"
        x = Variable("x")
        y = Variable("y")
        x_min = Variable("x_{min}", 2)
        y_max = Variable("y_{max}", 0.5)
        with SignomialsEnabled():
            m = Model(x, [Tight([x + y >= 1]), x >= x_min, y <= y_max])
        sol = m.localsolve(verbosity=0)
        warndata = sol.meta["warnings"]["Unexpectedly Loose Constraints"][0][1]
        assert warndata[-1] is m[0][0]
        assert warndata[0] > 0.5
        m.substitutions[x_min] = 0.5
        assert m.localsolve(verbosity=0).cost == pytest.approx(0.5, abs=1e-5)


class TestBounded:  # pylint: disable=too-few-public-methods
    "Test bounded constraint set"

    def test_substitution_issue905(self):
        x = Variable("x")
        y = Variable("y")
        m = Model(x, [x >= y], {"y": 1})
        bm = Model(m.cost, Bounded(m))
        sol = bm.solve(verbosity=0)
        assert sol.cost == pytest.approx(1.0)
        bm = Model(m.cost, Bounded(m, lower=1e-10))
        sol = bm.solve(verbosity=0)
        assert sol.cost == pytest.approx(1.0)
        bm = Model(m.cost, Bounded(m, upper=1e10))
        sol = bm.solve(verbosity=0)
        assert sol.cost == pytest.approx(1.0)
