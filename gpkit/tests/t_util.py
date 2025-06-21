"""Tests for utils module"""

import unittest

import numpy as np

from gpkit import Model, NomialArray, Variable, parse_variables


class OnlyVectorParse(Model):
    """
    Variables of length 3
    ---------------------
    x    [-]    just another variable
    """

    @parse_variables(__doc__, globals())
    def setup(self):
        pass


class Fuselage(Model):
    """The thing that carries the fuel, engine, and payload

    Variables
    ---------
    f                [-]             Fineness
    g          9.81  [m/s^2]         Standard gravity
    k                [-]             Form factor
    l                [ft]            Length
    mfac       2.0   [-]             Weight margin factor
    R                [ft]            Radius
    rhocfrp    1.6   [g/cm^3]        Density of CFRP
    rhofuel    6.01  [lbf/gallon]    Density of 100LL fuel
    S                [ft^2]          Wetted area
    t          0.024 [in]            Minimum skin thickness
    Vol              [ft^3]          Volume
    W                [lbf]           Weight

    Upper Unbounded
    ---------------
    k, W

    """

    # pylint: disable=undefined-variable, invalid-name
    @parse_variables(__doc__, globals())
    def setup(self, Wfueltot):
        return [
            f == l / R / 2,
            k >= 1 + 60 / f**3 + f / 400,
            3 * (S / np.pi) ** 1.6075
            >= 2 * (l * R * 2) ** 1.6075 + (2 * R) ** (2 * 1.6075),
            Vol <= 4 * np.pi / 3 * (l / 2) * R**2,
            Vol >= Wfueltot / rhofuel,
            W / mfac >= S * rhocfrp * t * g,
        ]


class TestTools(unittest.TestCase):
    """TestCase for math models"""

    def test_vector_only_parse(self):
        # pylint: disable=no-member
        m = OnlyVectorParse()
        self.assertTrue(hasattr(m, "x"))
        self.assertIsInstance(m.x, NomialArray)
        self.assertEqual(len(m.x), 3)

    def test_parse_variables(self):
        Fuselage(Variable("Wfueltot", 5, "lbf"))


TESTS = [TestTools]


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests

    run_tests(TESTS)
