"""Tests for GP and SP classes"""

import sys
import unittest
from io import StringIO

import numpy as np

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


class TestGP(unittest.TestCase):
    """
    Test GeometricPrograms.
    This TestCase gets run once for each installed solver.
    """

    name = "TestGP_"
    # solver and ndig get set in loop at bottom this file, a bit hacky
    solver = None
    ndig = None

    def setup_method(self, method):  # pylint: disable=unused-argument
        """Set up test case with solver-specific precision"""
        self.ndig = get_ndig(self.solver)

    def test_no_monomial_constraints(self):
        x = Variable("x")
        sol = Model(x, [x + 1 / x <= 3]).solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"], 0.381966, self.ndig)

    def test_trivial_gp(self):
        """
        Create and solve a trivial GP:
            minimize    x + 2y
            subject to  xy >= 1

        The global optimum is (x, y) = (sqrt(2), 1/sqrt(2)).
        """
        x = Variable("x")
        y = Variable("y")
        prob = Model(cost=(x + 2 * y), constraints=[x * y >= 1])
        sol = prob.solve(solver=self.solver, verbosity=0)
        self.assertEqual(type(prob.latex()), str)
        # pylint: disable=protected-access
        self.assertEqual(type(prob._repr_latex_()), str)
        self.assertAlmostEqual(sol("x"), np.sqrt(2.0), self.ndig)
        self.assertAlmostEqual(sol("y"), 1 / np.sqrt(2.0), self.ndig)
        self.assertAlmostEqual(sol("x") + 2 * sol("y"), 2 * np.sqrt(2), self.ndig)
        self.assertAlmostEqual(sol["cost"], 2 * np.sqrt(2), self.ndig)

    def test_dup_eq_constraint(self):
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
        m.solve(verbosity=0, solver=self.solver)

        # add a redundant equality constraint
        m.extend([z == b / a])

        # solver will fail and attempt to debug
        m.solve(verbosity=0, solver=self.solver)

    def test_sigeq(self):
        x = Variable("x")
        y = VectorVariable(1, "y")
        c = Variable("c")
        # test left vector input to sigeq
        with SignomialsEnabled():
            m = Model(
                c,
                [c >= (x + 0.25) ** 2 + (y - 0.5) ** 2, SignomialEquality(x**2 + x, y)],
            )
        sol = m.localsolve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol("x"), 0.1639472, self.ndig)
        self.assertAlmostEqual(sol("y")[0], 0.1908254, self.ndig)
        self.assertAlmostEqual(sol("c"), 0.2669448, self.ndig)
        # test right vector input to sigeq
        with SignomialsEnabled():
            m = Model(
                c,
                [c >= (x + 0.25) ** 2 + (y - 0.5) ** 2, SignomialEquality(y, x**2 + x)],
            )
        sol = m.localsolve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol("x"), 0.1639472, self.ndig)
        self.assertAlmostEqual(sol("y")[0], 0.1908254, self.ndig)
        self.assertAlmostEqual(sol("c"), 0.2669448, self.ndig)
        # test scalar input to sigeq
        z = Variable("z")
        with SignomialsEnabled():
            m = Model(
                c,
                [c >= (x + 0.25) ** 2 + (z - 0.5) ** 2, SignomialEquality(x**2 + x, z)],
            )
        sol = m.localsolve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol("x"), 0.1639472, self.ndig)
        self.assertAlmostEqual(sol("z"), 0.1908254, self.ndig)
        self.assertAlmostEqual(sol("c"), 0.2669448, self.ndig)

    def test_601(self):
        # tautological monomials should solve but not pass to the solver
        x = Variable("x")
        y = Variable("y", 2)
        m = Model(x, [x >= 1, y == 2])
        m.solve(solver=self.solver, verbosity=0)
        self.assertEqual(len(list(m.as_hmapslt1({}))), 3)
        self.assertEqual(len(m.program.hmaps), 2)

    def test_cost_freeing(self):
        "Test freeing a variable that's in the cost."
        x = Variable("x", 1)
        x_min = Variable("x_{min}", 2)
        intermediary = Variable("intermediary")
        m = Model(x, [x >= intermediary, intermediary >= x_min])
        self.assertRaises(
            (PrimalInfeasible, UnknownInfeasible),
            m.solve,
            solver=self.solver,
            verbosity=0,
        )
        x = Variable("x", 1)
        x_min = Variable("x_{min}", 2)
        m = Model(x, [x >= x_min])
        self.assertRaises(PrimalInfeasible, m.solve, solver=self.solver, verbosity=0)
        del m.substitutions[m["x"]]
        self.assertAlmostEqual(m.solve(solver=self.solver, verbosity=0)["cost"], 2)
        del m.substitutions[m["x_{min}"]]
        self.assertRaises(UnboundedGP, m.solve, solver=self.solver, verbosity=0)
        gp = m.gp(checkbounds=False)
        self.assertRaises(DualInfeasible, gp.solve, solver=self.solver, verbosity=0)

    def test_simple_united_gp(self):
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
        sol = prob.solve(solver=self.solver, verbosity=0)
        almostequal = self.assertAlmostEqual
        almostequal(0.000553226 / sol["cost"], 1, self.ndig)
        almostequal(340.29 / sol["constants"]["a0"], 1, self.ndig)
        almostequal(340.29 / sol["variables"]["a0"], 1, self.ndig)
        almostequal(340.29 * a0.units / sol("a0"), 1, self.ndig)
        almostequal(1807.58 / sol["freevariables"]["R"], 1, self.ndig)
        almostequal(1807.58 * R.units / sol("R"), 1, self.ndig)

    def test_trivial_vector_gp(self):
        "Create and solve a trivial GP with VectorVariables"
        x = VectorVariable(2, "x")
        y = VectorVariable(2, "y")
        prob = Model(cost=(sum(x) + 2 * sum(y)), constraints=[x * y >= 1])
        sol = prob.solve(solver=self.solver, verbosity=0)
        self.assertEqual(sol("x").shape, (2,))
        self.assertEqual(sol("y").shape, (2,))
        for x, y in zip(sol("x"), sol("y")):
            self.assertAlmostEqual(x, np.sqrt(2.0), self.ndig)
            self.assertAlmostEqual(y, 1 / np.sqrt(2.0), self.ndig)
        self.assertAlmostEqual(sol["cost"] / (4 * np.sqrt(2)), 1.0, self.ndig)

    def test_sensitivities(self):
        W_payload = Variable("W_{payload}", 175 * (195 + 30), "lbf")
        f_oew = Variable("f_{oew}", 0.53, "-", "OEW/MTOW")
        fuel_per_nm = Variable("\\theta_{fuel}", 13.75, "lbf/nautical_mile")
        R = Variable("R", 3000, "nautical_miles", "range")
        mtow = Variable("MTOW", "lbf", "max take off weight")

        m = Model(
            61.3e6 * units.USD * (mtow / (1e5 * units.lbf)) ** 0.807,
            [mtow >= W_payload + f_oew * mtow + fuel_per_nm * R],
        )
        sol = m.solve(solver=self.solver, verbosity=0)
        senss = sol["sensitivities"]["variables"]
        self.assertAlmostEqual(senss[f_oew], 0.91, 2)
        self.assertAlmostEqual(senss[R], 0.41, 2)
        self.assertAlmostEqual(senss[fuel_per_nm], 0.41, 2)
        self.assertAlmostEqual(senss[W_payload], 0.39, 2)

    def test_mdd_example(self):
        Cl = Variable("Cl", 0.5, "-", "Lift Coefficient")
        Mdd = Variable("Mdd", "-", "Drag Divergence Mach Number")
        m1 = Model(1 / Mdd, [1 >= 5 * Mdd + 0.5, Mdd >= 0.00001])
        m2 = Model(1 / Mdd, [1 >= 5 * Mdd + 0.5])
        m3 = Model(1 / Mdd, [1 >= 5 * Mdd + Cl, Mdd >= 0.00001])
        sol1 = m1.solve(solver=self.solver, verbosity=0)
        sol2 = m2.solve(solver=self.solver, verbosity=0)
        sol3 = m3.solve(solver=self.solver, verbosity=0)
        # pylint: disable=no-member
        gp1, gp2, gp3 = [m.program for m in [m1, m2, m3]]
        self.assertEqual(
            gp1.A, CootMatrix(row=[0, 1, 2], col=[0, 0, 0], data=[-1, 1, -1])
        )
        self.assertEqual(gp2.A, CootMatrix(row=[0, 1], col=[0, 0], data=[-1, 1]))
        self.assertEqual(
            gp3.A, CootMatrix(row=[0, 1, 2], col=[0, 0, 0], data=[-1, 1, -1])
        )
        self.assertTrue((gp3.A.todense() == np.matrix([-1, 1, -1]).T).all())
        self.assertAlmostEqual(sol1(Mdd), sol2(Mdd))
        self.assertAlmostEqual(sol1(Mdd), sol3(Mdd))
        self.assertAlmostEqual(sol2(Mdd), sol3(Mdd))

    def test_additive_constants(self):
        x = Variable("x")
        m = Model(1 / x, [1 >= 5 * x + 0.5, 1 >= 5 * x])
        m.solve(verbosity=0)
        # pylint: disable=no-member
        gp = m.program  # created by solve()
        self.assertEqual(gp.cs[1], 2 * gp.cs[2])
        self.assertEqual(gp.A.data[1], gp.A.data[2])

    def test_zeroing(self):
        L = Variable("L")
        k = Variable("k", 0)
        with SignomialsEnabled():
            constr = [L - 5 * k <= 10]
        sol = Model(1 / L, constr).solve(self.solver, verbosity=0)
        self.assertAlmostEqual(sol(L), 10, self.ndig)
        self.assertAlmostEqual(sol["cost"], 0.1, self.ndig)
        self.assertTrue(sol.almost_equal(sol))

    @unittest.skipIf(
        settings["default_solver"] == "cvxopt",
        "cvxopt cannot solve singular problems",
    )
    def test_singular(self):  # pragma: no cover
        "Create and solve GP with a singular A matrix"
        x = Variable("x")
        y = Variable("y")
        m = Model(y * x, [y * x >= 12])
        sol = m.solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"], 12, self.ndig)

    def test_constants_in_objective_1(self):
        "Issue 296"
        x1 = Variable("x1")
        x2 = Variable("x2")
        m = Model(1.0 + x1 + x2, [x1 >= 1.0, x2 >= 1.0])
        sol = m.solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"], 3, self.ndig)

    def test_constants_in_objective_2(self):
        "Issue 296"
        x1 = Variable("x1")
        x2 = Variable("x2")
        m = Model(x1**2 + 100 + 3 * x2, [x1 >= 10.0, x2 >= 15.0])
        sol = m.solve(solver=self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"] / 245.0, 1, self.ndig)

    def test_terminating_constant_(self):
        x = Variable("x")
        y = Variable("y", value=0.5)
        prob = Model(1 / x, [x + y <= 4])
        sol = prob.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 1 / 3.5, self.ndig)

    def test_exps_is_tuple(self):
        "issue 407"
        x = Variable("x")
        m = Model(x, [x >= 1])
        m.solve(verbosity=0)
        self.assertEqual(type(m.program.cost.exps), tuple)

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
        self.assertEqual(gp1.A, gp2.A)
        self.assertTrue(gp1.cs == gp2.cs)


class TestSP(unittest.TestCase):
    "test case for SP class -- gets run for each installed solver"

    name = "TestSP_"
    solver = None
    ndig = None

    def setup_method(self, method):  # pylint: disable=unused-argument
        """Set up test case with solver-specific precision"""
        self.ndig = get_ndig(self.solver)

    def test_sp_relaxation(self):
        w = Variable("w")
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        with SignomialsEnabled():
            m = Model(x, [x + y >= w, x + y <= z / 2, y <= x, y >= 1], {z: 3, w: 3})
        r1 = ConstantsRelaxed(m)
        self.assertEqual(len(r1.vks), 8)
        with self.assertRaises(ValueError):
            _ = Model(x * r1.relaxvars, r1)  # no "prod"
        sp = Model(x * r1.relaxvars.prod() ** 10, r1).sp(use_pccp=False)
        cost = sp.localsolve(verbosity=0, solver=self.solver)["cost"]
        self.assertAlmostEqual(cost / 1024, 1, self.ndig)
        m.debug(verbosity=0, solver=self.solver)
        with SignomialsEnabled():
            m = Model(x, [x + y >= z, x + y <= z / 2, y <= x, y >= 1], {z: 3})
        m.debug(verbosity=0, solver=self.solver)
        r2 = ConstraintsRelaxed(m)
        self.assertEqual(len(r2.vks), 7)
        sp = Model(x * r2.relaxvars.prod() ** 10, r2).sp(use_pccp=False)
        cost = sp.localsolve(verbosity=0, solver=self.solver)["cost"]
        self.assertAlmostEqual(cost / 1024, 1, self.ndig)
        with SignomialsEnabled():
            m = Model(x, [x + y >= z, x + y <= z / 2, y <= x, y >= 1], {z: 3})
        m.debug(verbosity=0, solver=self.solver)
        r3 = ConstraintsRelaxedEqually(m)
        self.assertEqual(len(r3.vks), 4)
        sp = Model(x * r3.relaxvar**10, r3).sp(use_pccp=False)
        cost = sp.localsolve(verbosity=0, solver=self.solver)["cost"]
        self.assertAlmostEqual(cost / (32 * 0.8786796585), 1, self.ndig)

    def test_sp_bounded(self):
        x = Variable("x")
        y = Variable("y")

        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.1])  # solves
        cost = m.localsolve(verbosity=0, solver=self.solver)["cost"]
        self.assertAlmostEqual(cost, 0.9, self.ndig)

        with SignomialsEnabled():
            m = Model(x, [x + y >= 1])  # dual infeasible
        with self.assertRaises(UnboundedGP):
            m.localsolve(verbosity=0, solver=self.solver)
        gp = m.sp(checkbounds=False).gp()
        self.assertRaises(DualInfeasible, gp.solve, solver=self.solver, verbosity=0)

        with SignomialsEnabled():
            m = Model(x, Bounded([x + y >= 1]))
        sol = m.localsolve(verbosity=0, solver=self.solver)
        boundedness = sol["boundedness"]
        # depends on solver, platform, whims of the numerical deities
        if "value near lower bound of 1e-30" in boundedness:  # pragma: no cover
            self.assertIn(x.key, boundedness["value near lower bound of 1e-30"])
        else:  # pragma: no cover
            self.assertIn(y.key, boundedness["value near upper bound of 1e+30"])

    def test_values_vs_subs(self):
        # Substitutions update method
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        with SignomialsEnabled():
            constraints = [x + y >= z, y >= x - 1]
        m = Model(x + y * z, constraints)
        m.substitutions.update({"z": 5})
        sol = m.localsolve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["cost"], 13, self.ndig)

        # Constant variable declaration method
        z = Variable("z", 5)
        with SignomialsEnabled():
            constraints = [x + y >= z, y >= x - 1]
        m = Model(x + y * z, constraints)
        sol = m.localsolve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["cost"], 13, self.ndig)

    def test_initially_infeasible(self):
        x = Variable("x")
        y = Variable("y")

        with SignomialsEnabled():
            sigc = x >= y + y**2 - y**3
            sigc2 = x <= y**0.5

        m = Model(1 / x, [sigc, sigc2, y <= 0.5])

        sol = m.localsolve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["cost"], 2**0.5, self.ndig)
        self.assertAlmostEqual(sol(y), 0.5, self.ndig)

    def test_sp_substitutions(self):
        x = Variable("x")
        y = Variable("y", 1)
        z = Variable("z", 4)

        old_stdout = sys.stdout
        sys.stdout = stringout = StringIO()

        with SignomialsEnabled():
            m1 = Model(x, [x + z >= y])
        with self.assertRaises(UnnecessarySGP):
            m1.localsolve(verbosity=0, solver=self.solver)
        with self.assertRaises(UnboundedGP):
            m1.solve(verbosity=0, solver=self.solver)

        with SignomialsEnabled():
            m2 = Model(x, [x + y >= z])
            m2.substitutions[y] = 1
            m2.substitutions[z] = 4
        sol = m2.solve(self.solver, verbosity=0)
        self.assertAlmostEqual(sol["cost"], 3, self.ndig)

        sys.stdout = old_stdout
        self.assertEqual(
            stringout.getvalue(),
            (
                f"Warning: SignomialConstraint {str(m1[0])} became the "
                "tautological constraint 0 <= 3 + x after substitution.\n"
                f"Warning: SignomialConstraint {str(m1[0])} became the "
                "tautological constraint 0 <= 3 + x after substitution.\n"
            ),
        )

    def test_tautological(self):
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
        self.assertAlmostEqual(
            m1.solve(self.solver, verbosity=0)["cost"],
            m2.solve(self.solver, verbosity=0)["cost"],
        )

        sys.stdout = old_stdout
        self.assertEqual(
            stringout.getvalue(),
            (
                f"Warning: SignomialConstraint {str(m1[0])} became the "
                "tautological constraint 0 <= 1 + x after substitution.\n"
                f"Warning: SignomialConstraint {str(m2[0])} became the "
                "tautological constraint 0 <= 1 + x after substitution.\n"
            ),
        )

    def test_impossible(self):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        with SignomialsEnabled():
            m1 = Model(x, [x + y >= z, x >= y])
        m1.substitutions.update({"x": 0, "y": 0})
        with self.assertRaises(ValueError):
            _ = m1.localsolve(solver=self.solver)

    def test_trivial_sp(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(x, [x >= 1 - y, y <= 0.1])
        with self.assertRaises(InvalidGPConstraint):
            m.solve(verbosity=0, solver=self.solver)
        sol = m.localsolve(self.solver, verbosity=0)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)
        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.1])
        sol = m.localsolve(self.solver, verbosity=0)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)

    def test_tautological_spconstraint(self):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z", 0)
        with SignomialsEnabled():
            m = Model(x, [x >= 1 - y, y <= 0.1, y >= z])
        with self.assertRaises(InvalidGPConstraint):
            m.solve(verbosity=0, solver=self.solver)
        sol = m.localsolve(self.solver, verbosity=0)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)

    def test_relaxation(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        m = Model(objective, constraints)
        m.localsolve(verbosity=0, solver=self.solver)

        # issue #257

        A = VectorVariable(2, "A")
        B = ArrayVariable([2, 2], "B")
        C = VectorVariable(2, "C")
        with SignomialsEnabled():
            constraints = [A <= B.dot(C), B <= 1, C <= 1]
        obj = 1 / A[0] + 1 / A[1]
        m = Model(obj, constraints)
        m.localsolve(verbosity=0, solver=self.solver)

    def test_issue180(self):
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
        spsol = m.solve(self.solver, verbosity=0)
        # now solve as GP
        m[-1] = Obj >= a_val * (2 * L + 2 * W) + (1 - a_val) * (12 * W**-1 * L**-3)
        del m.substitutions[m["a"]]
        gpsol = m.solve(self.solver, verbosity=0)
        self.assertAlmostEqual(spsol["cost"], gpsol["cost"])

    def test_trivial_sp2(self):
        x = Variable("x")
        y = Variable("y")

        # converging from above
        with SignomialsEnabled():
            constraints = [y + x >= 2, y >= x]
        objective = y
        x0 = 1
        y0 = 2
        m = Model(objective, constraints)
        sol1 = m.localsolve(x0={x: x0, y: y0}, verbosity=0, solver=self.solver)

        # converging from right
        with SignomialsEnabled():
            constraints = [y + x >= 2, y <= x]
        objective = x
        x0 = 2
        y0 = 1
        m = Model(objective, constraints)
        sol2 = m.localsolve(x0={x: x0, y: y0}, verbosity=0, solver=self.solver)

        self.assertAlmostEqual(
            sol1["variables"]["x"], sol2["variables"]["x"], self.ndig
        )
        self.assertAlmostEqual(
            sol1["variables"]["y"], sol2["variables"]["x"], self.ndig
        )

    def test_sp_initial_guess_sub(self):
        x = Variable("x")
        y = Variable("y")
        x0 = 3
        y0 = 2
        with SignomialsEnabled():
            constraints = [y + x >= 4, y <= x]
        objective = x
        m = Model(objective, constraints)
        # Call to local solve with only variables
        sol = m.localsolve(x0={"x": x0, y: y0}, verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol(x), 2, self.ndig)
        self.assertAlmostEqual(sol["cost"], 2, self.ndig)

        # Call to local solve with only variable strings
        sol = m.localsolve(x0={"x": x0, "y": y0}, verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol("x"), 2, self.ndig)
        self.assertAlmostEqual(sol["cost"], 2, self.ndig)

        # Call to local solve with a mix of variable strings and variables
        sol = m.localsolve(x0={"x": x0, y: y0}, verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["cost"], 2, self.ndig)

    def test_small_named_signomial(self):
        x = Variable("x")
        z = Variable("z")
        local_ndig = 4
        nonzero_adder = 0.1
        with SignomialsEnabled():
            J = 0.01 * (x - 1) ** 2 + nonzero_adder
            with NamedVariables("SmallSignomial"):
                m = Model(z, [z >= J])
        sol = m.localsolve(verbosity=0, solver=self.solver)
        self.assertAlmostEqual(sol["cost"], nonzero_adder, local_ndig)
        self.assertAlmostEqual(sol("x"), 0.98725425, self.ndig)

    def test_sigs_not_allowed_in_cost(self):
        with SignomialsEnabled():
            x = Variable("x")
            y = Variable("y")
            J = 0.01 * ((x - 1) ** 2 + (y - 1) ** 2) + (x * y - 1) ** 2
            m = Model(J)
            with self.assertRaises(InvalidPosynomial):
                m.localsolve(verbosity=0, solver=self.solver)

    def test_partial_sub_signomial(self):
        "Test SP partial x0 initialization"
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(x, [x + y >= 1, y <= 0.5])
        gp = m.sp().gp(x0={x: 0.5})  # pylint: disable=no-member
        (first_gp_constr_posy_exp,) = gp.hmaps[1]  # first after cost
        self.assertEqual(first_gp_constr_posy_exp[x.key], -1.0 / 3)

    def test_becomes_signomial(self):
        "Test that a GP does not become an SP after substitutions"
        x = Variable("x")
        c = Variable("c")
        y = Variable("y")
        m = Model(x, [y >= 1 + c * x, y <= 0.5], {c: -1})
        with self.assertRaises(InvalidGPConstraint):
            with SignomialsEnabled():
                m.gp()
        with self.assertRaises(UnnecessarySGP):
            m.localsolve(solver=self.solver)

    def test_reassigned_constant_cost(self):
        # for issue 1131
        x = Variable("x")
        x_min = Variable("x_min", 1)
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(y, [y + 0.5 >= x, x >= x_min, 6 >= y])
        m.localsolve(verbosity=0, solver=self.solver)
        del m.substitutions[x_min]
        m.cost = 1 / x_min
        self.assertNotIn(x_min, m.sp().gp().substitutions)  # pylint: disable=no-member

    def test_unbounded_debugging(self):
        "Test nearly-dual-feasible problems"
        x = Variable("x")
        y = Variable("y")
        m = Model(x * y, [x * y**1.01 >= 100])
        with self.assertRaises((DualInfeasible, UnknownInfeasible)):
            m.solve(self.solver, verbosity=0)
        # test one-sided bound
        m = Model(x * y, Bounded(m, lower=0.001))
        sol = m.solve(self.solver, verbosity=0)
        bounds = sol["boundedness"]
        self.assertEqual(bounds["sensitive to lower bound of 0.001"], set([x.key]))
        # end test one-sided bound
        m = Model(x * y, [x * y**1.01 >= 100])
        m = Model(x * y, Bounded(m))
        sol = m.solve(self.solver, verbosity=0)
        bounds = sol["boundedness"]
        # depends on solver, platform, whims of the numerical deities
        if "sensitive to upper bound of 1e+30" in bounds:  # pragma: no cover
            self.assertIn(y.key, bounds["sensitive to upper bound of 1e+30"])
        else:  # pragma: no cover
            self.assertIn(x.key, bounds["sensitive to lower bound of 1e-30"])


class TestModelSolverSpecific(unittest.TestCase):
    "test cases run only for specific solvers"

    def test_cvxopt_kwargs(self):  # pragma: no cover
        if "cvxopt" not in settings["installed_solvers"]:
            return
        x = Variable("x")
        m = Model(x, [x >= 12])
        # make sure it"s possible to pass the kktsolver option to cvxopt
        sol = m.solve(solver="cvxopt", verbosity=0, kktsolver="ldl")
        self.assertAlmostEqual(sol["cost"], 12.0, NDIGS["cvxopt"])


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


class TestModelNoSolve(unittest.TestCase):
    "model tests that don't require a solver"

    def test_modelname_added(self):
        t = Thing(2)
        for vk in t.vks:
            self.assertEqual(vk.lineage, (("Thing", 0),))

    def test_modelcontainmentprinting(self):
        t = Thing2()
        self.assertEqual(t["c"].key.models, ("Thing2", "Thing"))
        self.assertIsInstance(t.str_without(), str)
        self.assertIsInstance(t.latex(), str)

    def test_no_naming_on_var_access(self):
        # make sure that analysis models don't add their names to
        # variables looked up from other models
        box = Box()
        area_bounds = BoxAreaBounds(box)
        M = Model(box["V"], [box, area_bounds])
        for var in ("h", "w", "d"):
            self.assertEqual(len(M.varkeys.by_name(var)), 1)

    def test_duplicate_submodel_varnames(self):
        w = Widget()
        # w has two Sub models, both with their own variable m
        self.assertEqual(len(w.varkeys.by_name("m")), 2)
        # keys for both submodel m's should be in the parent model varkeys
        self.assertIn(w.subA["m"].key, w.varkeys)
        self.assertIn(w.subB["m"].key, w.varkeys)
        # keys of w.variables_byname("m") should match m.varkeys
        m_vbn_keys = w.varkeys.by_name("m")
        self.assertIn(w.subA["m"].key, m_vbn_keys)
        self.assertIn(w.subB["m"].key, m_vbn_keys)
        # dig a level deeper, into the keymap
        self.assertEqual(len(w.varkeys.keys("m")), 2)


TESTS = [TestModelSolverSpecific, TestModelNoSolve]
MULTI_SOLVER_TESTS = [TestGP, TestSP]

for testcase in MULTI_SOLVER_TESTS:
    for solver in settings["installed_solvers"]:
        if solver:
            test = type(str(testcase.__name__ + "_" + solver), (testcase,), {})
            setattr(test, "solver", solver)
            setattr(test, "ndig", get_ndig(solver))
            TESTS.append(test)

if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests

    run_tests(TESTS, verbosity=0)
