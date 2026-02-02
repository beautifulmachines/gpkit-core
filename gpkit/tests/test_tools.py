"""Tests for tools module"""

import numpy as np
import pytest

from gpkit import Model, Variable, VectorVariable
from gpkit.solutions import Solution
from gpkit.tests.conftest import assert_logtol
from gpkit.tools.autosweep import BinarySweepTree
from gpkit.tools.tools import te_exp_minus1, te_secant, te_tangent


class TestTools:
    """TestCase for sweeps and tools"""

    def test_binary_sweep_tree(self):
        def dummy_sol(cost):
            vals = {"primal": None, "constants": None, "sens": None, "meta": {}}
            return Solution(cost=cost, **vals)

        bst0 = BinarySweepTree([1, 2], [dummy_sol(1), dummy_sol(8)], None, None)
        assert_logtol(bst0.sample_at([1, 1.5, 2])["cost"], [1, 3.375, 8], 1e-3)
        bst0.add_split(1.5, dummy_sol(4))
        assert_logtol(
            bst0.sample_at([1, 1.25, 1.5, 1.75, 2])["cost"],
            [1, 2.144, 4, 5.799, 8],
            1e-3,
        )

    def test_dual_objective(self):
        x = Variable("x")
        y = Variable("y")
        eqns = [x >= 1, y >= 1, x * y == 10]
        n = 4
        ws = Variable("w_{CO}")
        w_s = Variable("v_{CO}", lambda c: 1 - c[ws], "-")
        obj = ws * (x + y) + w_s * (y**-1 * x**-3)
        m = Model(obj, eqns)
        sols = m.sweep({ws: np.linspace(1 / n, 1 - 1 / n, n)}, verbosity=0)
        a = [sol.cost for sol in sols]
        b = np.array([1.58856898, 2.6410391, 3.69348122, 4.74591386])
        np.testing.assert_allclose(a, b, rtol=1e-7)

    def test_te_exp_minus1(self):
        """Test Taylor expansion of e^x - 1"""
        x = Variable("x")
        assert te_exp_minus1(x, 1) == x
        assert te_exp_minus1(x, 3) == x + x**2 / 2 + x**3 / 6
        assert te_exp_minus1(x, 0) == 0
        # make sure x was not modified
        assert x == Variable("x")
        # try for VectorVariable too
        y = VectorVariable(3, "y")
        assert te_exp_minus1(y, 1) == y
        assert te_exp_minus1(y, 3) == y + y**2 / 2 + y**3 / 6
        assert te_exp_minus1(y, 0) == 0
        # make sure y was not modified
        assert y == VectorVariable(3, "y")

    def test_te_secant(self):
        "Test Taylor expansion of secant(var)"
        x = Variable("x")
        assert te_secant(x, 1) == 1 + x**2 / 2
        a = te_secant(x, 2)
        b = 1 + x**2 / 2 + 5 * x**4 / 24
        assert all(abs(val) <= 1e-10 for val in (a.hmap - b.hmap).values())
        assert te_secant(x, 0) == 1
        # make sure x was not modified
        assert x == Variable("x")
        # try for VectorVariable too
        y = VectorVariable(3, "y")
        assert te_secant(y, 0) == 1  # truthy bc monomial constraint
        assert all(te_secant(y, 1) == 1 + y**2 / 2)
        assert all(te_secant(y, 2) == 1 + y**2 / 2 + 5 * y**4 / 24)
        # make sure y was not modified
        assert y == VectorVariable(3, "y")
        _ = te_secant(x, 13)  # to trigger the extension

    def test_te_tangent(self):
        "Test Taylor expansion of tangent(var)"
        x = Variable("x")
        assert te_tangent(x, 1) == x
        assert te_tangent(x, 3) == x + x**3 / 3 + 2 * x**5 / 15
        assert te_tangent(x, 0) == 0
        # make sure x was not modified
        assert x == Variable("x")
        # try for VectorVariable too
        y = VectorVariable(3, "y")
        assert te_tangent(y, 1) == y
        assert te_tangent(y, 3) == y + y**3 / 3 + 2 * y**5 / 15
        assert te_tangent(y, 0) == 0
        # make sure y was not modified
        assert y == VectorVariable(3, "y")
        with pytest.raises(NotImplementedError):
            _ = te_tangent(x, 16)
