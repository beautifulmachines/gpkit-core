"""Test substitution capability across gpkit"""

import pickle

import numpy as np
import numpy.testing as npt
import pytest
from adce import ADV, adnumber

import gpkit
from gpkit import (
    Model,
    NamedVariables,
    Signomial,
    SignomialsEnabled,
    Variable,
    VectorVariable,
)
from gpkit.exceptions import InvalidGPConstraint, InvalidPosynomial
from gpkit.units import DimensionalityError
from gpkit.util.small_scripts import mag


class TestNomialSubs:
    """Test substitution for nomial-family objects"""

    def test_vectorized_linked(self):
        class VectorLinked(Model):
            "simple vectorized link"

            def setup(self):
                self.y = y = Variable("y", 1)

                def vectorlink(c):
                    "linked vector function"
                    if isinstance(c[y], ADV):
                        return np.array(c[y]) + adnumber([1, 2, 3])
                    return c[y] + np.array([1, 2, 3])

                self.x = x = VectorVariable(3, "x")
                return [], {x: vectorlink}

        m = VectorLinked()
        assert m.substitutions[m.x[0].key](m.substitutions) == 2
        assert m.gp().substitutions[m.x[0].key] == 2
        assert m.gp().substitutions[m.x[1].key] == 3
        assert m.gp().substitutions[m.x[2].key] == 4

    def test_numeric(self):
        """Basic substitution of numeric value"""
        x = Variable("x")
        p = x**2
        assert p.sub({x: 3}) == 9
        assert p.sub({x.key: 3}) == 9
        assert p.sub({"x": 3}) == 9

    def test_dimensionless_units(self):
        x = Variable("x", 3, "ft")
        y = Variable("y", 1, "m")
        if x.units is not None:
            # units are enabled
            assert (x / y).value == pytest.approx(0.9144)

    def test_vector(self):
        x = Variable("x")
        y = Variable("y")
        z = VectorVariable(2, "z")
        p = x * y * z
        assert all(p.sub({x: 1, "y": 2}) == 2 * z)
        assert all(p.sub({x: 1, y: 2, "z": [1, 2]}) == z.sub({z: [2, 4]}))
        with pytest.raises(ValueError):
            z.sub({z: [1, 2, 3]})

        xvec = VectorVariable(3, "x", "m")
        xs = xvec[:2].sum()
        for x_ in ["x", xvec]:
            assert mag(xs.sub({x_: [1, 2, 3]}).c) == pytest.approx(3.0)

    def test_variable(self):
        """Test special single-argument substitution for Variable"""
        x = Variable("x")
        y = Variable("y")
        _ = x * y**2
        assert x.sub(3) == 3
        # make sure x was not mutated
        assert x == Variable("x")
        assert x.sub(3) != Variable("x")
        # also make sure the old way works
        assert x.sub({x: 3}) == 3
        # and for vectors
        xvec = VectorVariable(3, "x")
        assert xvec[1].sub(3) == 3

    def test_signomial(self):
        """Test Signomial substitution"""
        D = Variable("D", units="N")
        x = Variable("x", units="N")
        y = Variable("y", units="N")
        a = Variable("a")
        with SignomialsEnabled():
            sc = a * x + (1 - a) * y - D
            subbed = sc.sub({a: 0.1})
            assert subbed == 0.1 * x + 0.9 * y - D
            assert isinstance(subbed, Signomial)
            subbed = sc.sub({a: 2.0})
            assert isinstance(subbed, Signomial)
            assert subbed == 2 * x - y - D
            _ = a.sub({a: -1}).value  # fix monomial assumptions


class TestModelSubs:
    """Test substitution for Model objects"""

    def test_bad_gp_sub(self):
        x = Variable("x")
        y = Variable("y")
        with pytest.raises(ValueError, match="y must be fixed to a finite number"):
            Model(x, [y >= 1], {y: x})

    def test_sub_and_solve_refuse_alike(self):
        "One conversion serves both paths, so they refuse with one message"
        x, a = Variable("x"), Variable("a")
        with pytest.raises(ValueError, match="a must be fixed") as from_sub:
            (x * a).sub({a: "abc"})
        with pytest.raises(ValueError, match="a must be fixed") as from_solve:
            Model(x, [x >= a], {a: "abc"}).solve(verbosity=0)
        assert str(from_sub.value) == str(from_solve.value)

    def test_a_value_in_other_units_is_the_same_value(self):
        L, x = Variable("L", "m"), Variable("x", "m")
        in_cm = Model(x, [x >= L], {L: 30 * gpkit.units.cm}).solve(verbosity=0)
        in_m = Model(x, [x >= L], {L: 0.3}).solve(verbosity=0)
        assert mag(in_cm.cost) == pytest.approx(mag(in_m.cost))
        assert mag((2 * L).sub({L: 30 * gpkit.units.cm}).c) == pytest.approx(0.6)

    def test_quantity_sub(self):
        x = Variable("x", 1, "cm")
        y = Variable("y", 1)
        assert x.sub({x: 1 * gpkit.units.m}).c.magnitude == 100
        # NOTE: uncomment the below if requiring Quantity substitutions
        # with pytest.raises(ValueError): x.sub(x, 1)
        with pytest.raises(DimensionalityError):
            x.sub({x: 1 * gpkit.ureg.N})
        with pytest.raises(DimensionalityError):
            y.sub({y: 1 * gpkit.ureg.N})
        v = gpkit.VectorVariable(3, "v", "cm")
        subbed = v.sub({v: [1, 2, 3] * gpkit.ureg.m})
        assert [z.c.magnitude for z in subbed] == [100, 200, 300]
        v = VectorVariable(1, "v", "km")
        v_min = VectorVariable(1, "v_min", "km")
        m = Model(v.prod(), [v >= v_min], {v_min: [2 * gpkit.units("nmi")]})
        cost = m.solve(verbosity=0).cost
        assert cost / (2 * gpkit.ureg.nmi) == pytest.approx(1.0)
        m = Model(v.prod(), [v >= v_min], {v_min: np.array([2]) * gpkit.units("nmi")})
        cost = m.solve(verbosity=0).cost
        assert cost / (2 * gpkit.ureg.nmi) == pytest.approx(1.0)

    def test_phantoms(self):
        x = Variable("x")
        x_ = Variable("x", 1, lineage=[("test", 0)])
        xv = VectorVariable(2, "x", [1, 1], lineage=[("vec", 0)])
        m = Model(x, [x >= x_, x_ == xv.prod()])
        m.solve(verbosity=0)
        with pytest.raises(KeyError):
            _ = m.substitutions["x"]
        with pytest.raises(KeyError):
            _ = m.substitutions["y"]
        with pytest.raises(TypeError):
            _ = m["x"]
        assert x.key in m.varkeys.by_name("x")
        assert x_.key in m.varkeys.by_name("x")

    def test_persistence(self):
        x = gpkit.Variable("x")
        y = gpkit.Variable("y")
        ymax = gpkit.Variable("y_{max}", 0.1)

        with gpkit.SignomialsEnabled():
            m = gpkit.Model(x, [x >= 1 - y, y <= ymax])
            m.substitutions[ymax] = 0.2
            assert m.localsolve(verbosity=0).cost == pytest.approx(0.8, abs=1e-3)
            # VarKey values now persist across models (no value-nulling mutation)
            m = gpkit.Model(x, [x >= 1 - y, y <= ymax])
            assert m.localsolve(verbosity=0).cost == pytest.approx(0.9, abs=1e-3)
            m = gpkit.Model(x, [x >= 1 - y, y <= ymax])
            m.substitutions[ymax] = 0.1
            assert m.localsolve(verbosity=0).cost == pytest.approx(0.9, abs=1e-3)

    def test_united_sub_sweep(self):
        A = Variable("A", "USD")
        h = Variable("h", "USD/count")
        Q = Variable("Q", "count")
        Y = Variable("Y", "USD")
        m = Model(Y, [Y >= h * Q + A / Q])
        m.substitutions.update(
            {
                A: 500 * gpkit.units("USD"),
                h: 35 * gpkit.units("USD"),
            }
        )
        sols = m.sweep({Q: [50, 100, 500]}, verbosity=0)
        cost = [sol.cost.to("USD").magnitude for sol in sols]
        npt.assert_allclose(cost, [1760, 3505, 17501])

    def test_skipfailures(self):
        x = Variable("x")
        x_min = Variable("x_{min}", 1)
        m = Model(x, [x <= 1, x >= x_min])
        sweep = {x_min: [1, 2]}
        sol = m.sweep(sweep, verbosity=0, skipfailures=True)
        sol.table()
        assert len(sol) == 1

        subs = m.substitutions
        with pytest.raises(RuntimeWarning):
            sol = m.sweep(sweep, verbosity=0, skipfailures=False)
        assert m.substitutions is subs  # a failed sweep leaves the model as it was

        sweep[x_min][0] = 5  # so no sweeps solve
        with pytest.raises(RuntimeWarning):
            sol = m.sweep(sweep, verbosity=0, skipfailures=True)

    def test_vector_sweep(self):
        """Test sweep involving VectorVariables"""
        x = Variable("x")
        x_min = Variable("x_min", 1)
        y = VectorVariable(2, "y")
        m = Model(x, [x >= y.prod()])
        sweep = {y: np.reshape(np.meshgrid([2, 5, 9], [3, 7, 11]), (2, 9)).T}
        a = [sol.cost for sol in m.sweep(sweep, verbosity=0)]
        b = [6, 15, 27, 14, 35, 63, 22, 55, 99]
        npt.assert_allclose(a, b)
        x_min = Variable("x_min", 1)  # constant to check array indexing
        m = Model(x, [x >= y.prod(), x >= x_min])
        sweep = {"y": [[2, 5], [3, 5], [2, 7], [3, 7], [2, 11], [3, 11]]}
        sol = m.sweep(sweep, verbosity=0)
        b = [10, 15, 14, 21, 22, 33]
        for i, bi in enumerate(b):
            assert sol[i].constants[x_min] == 1
            assert sol[i].cost / bi == pytest.approx(1, abs=1e-6)
        m = Model(x, [x >= y.prod()])
        sweep = {y: [[2, 3, 9], [5, 7, 11]]}
        with pytest.raises(ValueError):
            m.sweep(sweep, verbosity=0)
        with pytest.raises(ValueError):  # points that differ in shape
            m.sweep({y: [[2, 3], [5, 7, 11]]}, verbosity=0)
        # a scalar point fills every element, as substituting a scalar does
        a = [sol.cost for sol in m.sweep({y: [1, 2, 3]}, verbosity=0)]
        npt.assert_allclose(a, [1, 4, 9])
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y[0]: 2})
        a = [sol.cost for sol in m.sweep({y[1]: [3, 5]}, verbosity=0)]
        b = [6, 10]
        npt.assert_allclose(a, b)
        # sweeping one element takes any number of points, not just len(y)
        a = [sol.cost for sol in m.sweep({y[1]: [3, 5, 7]}, verbosity=0)]
        npt.assert_allclose(a, [6, 10, 14])
        # create a numpy float array, then insert a sweep element
        m.substitutions.update({y: [2, 3]})
        a = [sol.cost for sol in m.sweep({y[1]: [3, 5]}, verbosity=0)]
        npt.assert_allclose(a, b)

    def test_calcconst(self):
        x = Variable("x", "hours")
        t_day = Variable("t_{day}", 12, "hours")
        t_night = Variable(
            "t_{night}", lambda c: 1 * gpkit.ureg.day - c.quantity(t_day), "hours"
        )
        _ = pickle.dumps(t_night)
        m = Model(x, [x >= t_day, x >= t_night])
        sol = m.solve(verbosity=0)
        assert sol[t_night] / gpkit.ureg.hours == pytest.approx(12)
        sol = m.sweep({t_day: [6, 8, 9, 13]}, verbosity=0)
        assert sol[0].sens[t_day] == pytest.approx(-1 / 3)
        assert sol[1].sens[t_day] == pytest.approx(-0.5, abs=1e-5)
        assert sol[2].sens[t_day] == pytest.approx(-0.6, abs=1e-4)
        assert sol[3].sens[t_day] == pytest.approx(+1, abs=1e-5)
        assert len(sol) == 4
        npt.assert_allclose([(s[t_day] + s[t_night]) / gpkit.ureg.hr for s in sol], 24)

    def test_vector_init(self):
        N = 6
        Weight = 50000
        xi_dist = (
            6
            * Weight
            / float(N)
            * (
                (np.array(range(1, N + 1)) - 0.5 / float(N)) / float(N)
                - (np.array(range(1, N + 1)) - 0.5 / float(N)) ** 2 / float(N) ** 2
            )
        )

        xi = VectorVariable(N, "xi", xi_dist, "N", "Constant Thrust per Bin")
        P = Variable("P", "N", "Total Power")
        phys_constraints = [P >= xi.sum()]
        objective = P
        eqns = phys_constraints
        m = Model(objective, eqns)
        sol = m.solve(verbosity=0)
        a, b = sol["xi"], xi_dist * gpkit.ureg.N
        assert all(abs(a - b) / (a + b) < 1e-7)

    def test_model_composition_units(self):
        class Above(Model):
            """A simple upper bound on x

            Lower Unbounded
            ---------------
            x
            """

            def setup(self):
                x = self.x = Variable("x", "ft")
                x_max = Variable("x_{max}", 1, "yard")
                self.cost = 1 / x
                return [x <= x_max]

        class Below(Model):
            """A simple lower bound on x

            Upper Unbounded
            ---------------
            x
            """

            def setup(self):
                x = self.x = Variable("x", "m")
                x_min = Variable("x_{min}", 1, "cm")
                self.cost = x
                return [x >= x_min]

        a, b = Above(), Below()
        concatm = Model(a.cost * b.cost, [a, b])
        concat_cost = concatm.solve(verbosity=0).cost
        yard, cm = gpkit.ureg("yard"), gpkit.ureg("cm")
        if not isinstance(a.x.key.units, str):
            assert a.solve(verbosity=0).cost * yard == pytest.approx(1)
            assert b.solve(verbosity=0).cost / cm == pytest.approx(1)
            assert concat_cost * yard / cm == pytest.approx(1)
        NamedVariables.reset_modelnumbers()
        a1, b1 = Above(), Below()
        assert a1.x.key.lineage == (("Above", 0),)
        m = Model(a1.x, [a1, b1, b1.x == a1.x])
        sol = m.solve(verbosity=0)
        if not isinstance(a1.x.key.units, str):
            assert sol.cost / cm == pytest.approx(1)
        a1, b1 = Above(), Below()
        # A second, independent root build of Above -- never composed with
        # the first (its `m` is discarded before this point) -- so it also
        # gets num 0: root-build numbering resets per build, not per thread.
        assert a1.x.key.lineage == (("Above", 0),)
        m = Model(b1.x, [a1, b1, b1.x == a1.x])
        sol = m.solve(verbosity=0)
        if not isinstance(b1.x.key.units, str):
            assert sol.cost / cm == pytest.approx(1)
        assert a1.x in sol.primal
        assert b1.x in sol.primal
        # a/b and a1/b1 are independent, never-composed builds of the same
        # classes, so they share the same VarKeys (both numbered 0).
        assert a.x in sol.primal
        assert b.x in sol.primal

    def test_getkey(self):
        class Top(Model):
            """Some high level model

            Upper Unbounded
            ---------------
            y
            """

            def setup(self):
                y = self.y = Variable("y")
                s = Sub()
                sy = s.y
                self.cost = y
                return [s, y >= sy, sy >= 1]

        class Sub(Model):
            """A simple sub model

            Upper Unbounded
            ---------------
            y
            """

            def setup(self):
                y = self.y = Variable("y")
                self.cost = y
                return [y >= 2]

        sol = Top().solve(verbosity=0)
        assert sol.cost == pytest.approx(2)

    def test_model_recursion(self):
        class Top(Model):
            """Some high level model

            Upper Unbounded
            ---------------
            x

            """

            def setup(self):
                sub = Sub()
                x = self.x = Variable("x")
                self.cost = x
                return sub, [x >= sub.y, sub.y >= 1]

        class Sub(Model):
            """A simple sub model

            Upper Unbounded
            ---------------
            y

            """

            def setup(self):
                y = self.y = Variable("y")
                self.cost = y
                return [y >= 2]

        sol = Top().solve(verbosity=0)
        assert sol.cost == pytest.approx(2)

    def test_vector_sub(self):
        x = VectorVariable(3, "x")
        y = VectorVariable(3, "y")
        ymax = VectorVariable(3, "ymax")

        with SignomialsEnabled():
            # issue1077 links to a case that failed for SPs only
            m = Model(x.prod(), [x + y >= 1, y <= ymax])

        m.substitutions["ymax"] = [0.3, 0.5, 0.8]
        m.localsolve(verbosity=0)

    def test_spsubs(self):
        x = Variable("x", 5)
        y = Variable("y", lambda c: 2 * c[x])
        z = Variable("z")
        w = Variable("w")

        with SignomialsEnabled():
            cnstr = [z + w >= y * x, w <= y]

        m = Model(z, cnstr)
        m.localsolve(verbosity=0)
        assert m.substitutions["y"], "__call__"

    def test_true_limit_with_vanishing_terms(self):
        "A true limit on a constant drops out, even if substitution leaves 0 <= 1"
        x, z = Variable("x"), Variable("z")
        sol = Model(x, [x >= 1, z <= 5], {z: 0}).solve(verbosity=0)
        assert sol.cost == pytest.approx(1)


class TestNomialMapSubs:
    "Tests substitutions of nomialmaps"

    def test_monomial_sub(self):
        z = Variable("z")
        w = Variable("w")

        with pytest.raises(ValueError):
            z.hmap.sub({z.key: w.key}, varkeys=z.vks)

    def test_sub_zero(self):
        z = Variable("z")
        w = Variable("w")

        p = 2 * w + z * w + 2

        assert p.sub({z: -2}) == 2
        with pytest.raises(ZeroDivisionError):
            Model(w, [w >= 1 / z], {z: 0}).gp()
        with pytest.raises(InvalidGPConstraint, match="finite"):
            Model(w, [w >= 1e200 * z], {z: 1e200}).gp()
        with pytest.raises(InvalidPosynomial, match="finite"):
            Model(1e200 * z * w, [w >= 1], {z: 1e200}).gp()

    @pytest.mark.parametrize(
        ("a_val", "b_val", "refused"),
        [
            (np.nan, 1, "a"),
            (lambda _: np.nan, 1, "a"),
            (np.inf, 0, "a"),
            (0, np.inf, "b"),
        ],
    )
    def test_a_nonfinite_value_is_refused(self, a_val, b_val, refused):
        "So no term can be 0*inf or inf - inf; a linked output is checked alike"
        x, a, b = Variable("x"), Variable("a"), Variable("b")
        with pytest.raises(ValueError, match=f"{refused} must be fixed to a finite"):
            Model(x, [x >= a * b, x >= 1], {a: a_val, b: b_val}).gp()

    def test_a_nonfinite_value_is_refused_by_nomial_sub(self):
        x, a, b = Variable("x"), Variable("a"), Variable("b")
        with SignomialsEnabled():
            p = a * x - b * x + 1
        with pytest.raises(ValueError, match="must be fixed to a finite"):
            p.sub({a: np.inf, b: np.inf})

    def test_log_sensitivity_at_zero(self):
        "d(cost)/dz is infinite here, but p*z**p, the log derivative of z**p, is 0"
        x, y, a, z = Variable("x"), Variable("y"), Variable("a"), Variable("z")
        with SignomialsEnabled():
            c = x + y * a >= 1 + z**0.5
        sol = Model(x, [c, x >= 0.1], {a: 0, z: 0}).solve(verbosity=0)
        assert sol.cost == pytest.approx(1)
        assert sol.sens.variables[z] == 0

    def test_sensitivity_to_constant_on_both_sides(self):
        x, p, q = Variable("x"), Variable("p"), Variable("q")
        with SignomialsEnabled():
            c = x * q + x * p >= 1 + 3 * q
        sol = Model(x, [c, x >= 0.01], {p: 1, q: 2}).solve(verbosity=0)
        # cost = (1 + 3q)/(q + p), so dlog(cost)/dlog(q) = 6/7 - 2/3
        assert sol.sens.variables[q] == pytest.approx(4 / 21, abs=1e-5)


class TestMergeDuringSubstitution:
    "Terms that merge before their last fixed variable is substituted"

    @pytest.fixture
    def xabc(self):
        return tuple(Variable(n) for n in "xabc")

    def test_constraint_sensitivities(self, xabc):
        x, a, b, c = xabc
        sol = Model(x, [x >= a * b * c + b * c], {a: 2, b: 3, c: 5}).solve(verbosity=0)
        # x = b*c*(a + 1)
        assert sol.sens.variables[a] == pytest.approx(2 / 3, abs=1e-5)
        assert sol.sens.variables[b] == pytest.approx(1, abs=1e-5)
        assert sol.sens.variables[c] == pytest.approx(1, abs=1e-5)
        assert sol.sens.variables[x] == pytest.approx(0, abs=1e-5)

    def test_cost_sensitivities(self, xabc):
        x, a, b, c = xabc
        m = Model(a * b * c * x + b * c * x, [x >= 1], {a: 2, b: 3, c: 5})
        sol = m.solve(verbosity=0)
        assert sol.sens.variables[a] == pytest.approx(2 / 3, abs=1e-5)
        assert sol.sens.variables[b] == pytest.approx(1, abs=1e-5)
        assert sol.sens.variables[c] == pytest.approx(1, abs=1e-5)

    def test_signomial_sensitivities(self, xabc):
        x, a, b, c = xabc
        y = Variable("y")
        with SignomialsEnabled():
            sig = a * b * c * x + b * c * x >= y + 1
        sol = Model(x, [sig, y >= 1], {a: 2, b: 3, c: 5}).solve(verbosity=0)
        # x = (y + 1)/(b*c*(a + 1))
        assert sol.sens.variables[a] == pytest.approx(-2 / 3, abs=1e-5)

    def test_expmap(self, xabc):
        "Every term maps to its exponents without the fixed variables"
        x, a, b, c = xabc
        p = a * b * c * x + b * c * x + c * x**2 + x
        fixed = {a.key: 2, b.key: 3, c.key: 5}
        hmap = p.hmap.sub(fixed, p.vks)
        for o_exp, exp in hmap.expmap.items():
            assert exp == {vk: e for vk, e in o_exp.items() if vk not in fixed}
            assert exp in hmap
        assert sum(hmap.csmap.values()) == pytest.approx(sum(hmap.values()))
