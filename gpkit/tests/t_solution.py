"""Tests for Solution class"""

import unittest

import numpy as np

import gpkit
from gpkit import Model, SignomialsEnabled, Variable, VectorVariable
from gpkit.util.small_classes import Quantity, Strings


class TestSolution(unittest.TestCase):
    """Unit tests for the Solution class"""

    def test_getitem(self):
        A = Variable("A", "-", "Test Variable")
        prob = Model(A, [A >= 1])
        sol = prob.solve(verbosity=0)
        self.assertAlmostEqual(sol[A], 1.0, 8)

    def test_getitem_units(self):
        # test from issue541
        x = Variable("x", 10, "ft")
        y = Variable("y", "m")
        m = Model(y, [y >= x])
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol["y"] / sol["x"], 1.0, 6)
        self.assertAlmostEqual(sol[x] / sol[y], 1.0, 6)

    def test_call_vector(self):
        n = 5
        x = VectorVariable(n, "x")
        prob = Model(sum(x), [x >= 2.5])
        sol = prob.solve(verbosity=0)
        solx = sol[x]
        self.assertEqual(type(solx), Quantity)
        self.assertEqual(type(solx.magnitude), np.ndarray)
        self.assertEqual(solx.shape, (n,))
        for i in range(n):
            self.assertAlmostEqual(solx[i], 2.5, places=4)

    def test_subinto(self):
        h_max = Variable("h_max", 10, "m", "Length")
        a_min = Variable("A_min", 10, "m^2", "Area")
        p_max = Variable("P", "m", "Perimeter")
        h = Variable("h", "m", "Length")
        w = Variable("w", "m", "width")
        m = Model(12 / (w * h**3), [h <= h_max, h * w >= a_min, p_max >= 2 * h + 2 * w])
        p_vals = np.linspace(13, 24, 20)
        sweepsol = m.sweep({p_max: p_vals}, verbosity=0)
        p_sol = [sol.subinto(p_max) for sol in sweepsol]
        self.assertEqual(len(p_sol), 20)
        for pv, ps in zip(p_vals, p_sol):
            self.assertAlmostEqual(0 * gpkit.ureg.m, pv * gpkit.ureg.m - ps)

    def test_table(self):
        x = Variable("x")
        gp = Model(x, [x >= 12])
        sol = gp.solve(verbosity=0)
        tab = sol.table()
        self.assertTrue(isinstance(tab, Strings))

    def test_units_sub(self):
        # issue 809
        t = Variable("t", "N", "thrust")
        tmin = Variable("t_{min}", "N", "minimum thrust")
        m = Model(t, [t >= tmin])
        tminsub = 1000 * gpkit.ureg.lbf
        m.substitutions.update({tmin: tminsub})
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol[tmin], tminsub)
        self.assertFalse(
            "1000N" in sol.table().replace(" ", "").replace("[", "").replace("]", "")
        )

    def test_key_options(self):
        # issue 993
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(y, [y + 6 * x >= 13 + x**2])
        msol = m.localsolve(verbosity=0)
        spsol = m.sp().localsolve(verbosity=0)  # pylint: disable=no-member
        gpsol = m.program.gps[-1].solve(verbosity=0)
        self.assertEqual(msol[x], msol["x"])
        self.assertEqual(spsol[x], spsol["x"])
        self.assertEqual(gpsol[x], gpsol["x"])
        self.assertEqual(msol[x], spsol[x])
        self.assertEqual(msol[x], gpsol[x])


class TestResultsTable(unittest.TestCase):
    """TestCase for var_table()"""

    def test_result_access(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            sig = y + 6 * x >= 13 + x**2
        m = Model(y, [sig])
        sol = m.localsolve(verbosity=0)
        self.assertTrue(
            all((isinstance(gp.result.table(), Strings) for gp in m.program.gps))
        )
        self.assertAlmostEqual(sol.cost / 4.0, 1.0, 5)
        self.assertAlmostEqual(sol["x"] / 3.0, 1.0, 3)


TESTS = [TestSolution, TestResultsTable]

if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests

    run_tests(TESTS)
