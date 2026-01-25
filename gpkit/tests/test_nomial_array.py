"""Tests for NomialArray class"""

import warnings as pywarnings

import numpy as np
import pytest

import gpkit
from gpkit import Monomial, NomialArray, Posynomial, Variable, VectorVariable
from gpkit.constraints.set import ConstraintSet
from gpkit.units import DimensionalityError


class TestNomialArray:
    """TestCase for the NomialArray class.
    Also tests VectorVariable, since VectorVariable returns a NomialArray
    """

    def test_shape(self):
        x = VectorVariable((2, 3), "x")
        assert x.shape == (2, 3)
        assert isinstance(x.str_without(), str)
        assert isinstance(x.latex(), str)

    def test_ndim(self):
        x = VectorVariable((3, 4), "x")
        assert x.ndim == 2

    def test_array_mult(self):
        x = VectorVariable(3, "x", label="dummy variable")
        x_0 = Variable("x", idx=(0,), shape=(3,), label="dummy variable")
        x_1 = Variable("x", idx=(1,), shape=(3,), label="dummy variable")
        x_2 = Variable("x", idx=(2,), shape=(3,), label="dummy variable")
        p = x_0**2 + x_1**2 + x_2**2
        assert x.dot(x) == p
        m = NomialArray(
            [
                [x_0**2, x_0 * x_1, x_0 * x_2],
                [x_0 * x_1, x_1**2, x_1 * x_2],
                [x_0 * x_2, x_1 * x_2, x_2**2],
            ]
        )
        assert x.outer(x) == m

    # pylint: disable=no-member
    def test_elementwise_mult(self):
        m = Variable("m")
        x = VectorVariable(3, "x", label="dummy variable")
        x_0 = Variable("x", idx=(0,), shape=(3,), label="dummy variable")
        x_1 = Variable("x", idx=(1,), shape=(3,), label="dummy variable")
        x_2 = Variable("x", idx=(2,), shape=(3,), label="dummy variable")
        # multiplication with numbers
        v = NomialArray([2, 2, 3]).T
        p = NomialArray([2 * x_0, 2 * x_1, 3 * x_2]).T
        assert x * v == p
        # division with numbers
        p2 = NomialArray([x_0 / 2, x_1 / 2, x_2 / 3]).T
        assert x / v == p2
        # power
        p3 = NomialArray([x_0**2, x_1**2, x_2**2]).T
        assert x**2 == p3
        # multiplication with monomials
        p = NomialArray([m * x_0, m * x_1, m * x_2]).T
        assert x * m == p
        # division with monomials
        p2 = NomialArray([x_0 / m, x_1 / m, x_2 / m]).T
        assert x / m == p2
        assert isinstance(v.str_without(), str)
        assert isinstance(v.latex(), str)
        assert isinstance(p.str_without(), str)
        assert isinstance(p.latex(), str)

    def test_constraint_gen(self):
        x = VectorVariable(3, "x", label="dummy variable")
        x_0 = Variable("x", idx=(0,), shape=(3,), label="dummy variable")
        x_1 = Variable("x", idx=(1,), shape=(3,), label="dummy variable")
        x_2 = Variable("x", idx=(2,), shape=(3,), label="dummy variable")
        v = NomialArray([1, 2, 3]).T
        p = [x_0, x_1 / 2, x_2 / 3]
        constraint = ConstraintSet([x <= v])
        assert list(constraint.as_hmapslt1({})) == [e.hmap for e in p]

    def test_substition(self):  # pylint: disable=no-member
        x = VectorVariable(3, "x", label="dummy variable")
        c = {x: [1, 2, 3]}
        assert x.sub(c) == [Monomial({}, e) for e in [1, 2, 3]]
        p = x**2
        assert p.sub(c) == [Monomial({}, e) for e in [1, 4, 9]]
        d = p.sum()
        assert d.sub(c) == Monomial({}, 14)

    # pylint: disable=no-member
    def test_units(self):
        # inspired by gpkit issue #106
        c = VectorVariable(5, "c", "m", "Local Chord")
        constraints = c == 1 * gpkit.units.m
        assert len(constraints) == 5
        # test an array with inconsistent units
        with pywarnings.catch_warnings():  # skip the UnitStrippedWarning
            pywarnings.simplefilter("ignore")
            mismatch = NomialArray([1 * gpkit.units.m, 1 * gpkit.ureg.ft, 1.0])
        with pytest.raises(DimensionalityError):
            mismatch.sum()
        assert mismatch[:2].sum().c == 1.3048 * gpkit.ureg.m
        assert mismatch.prod().c == 1 * gpkit.ureg.m * gpkit.ureg.ft

    def test_sum(self):
        x = VectorVariable(5, "x")
        p = x.sum()
        assert isinstance(p, Posynomial)
        assert p == sum(x)

        x = VectorVariable((2, 3), "x")
        rowsum = x.sum(axis=1)
        colsum = x.sum(axis=0)
        assert isinstance(rowsum, NomialArray)
        assert isinstance(colsum, NomialArray)
        assert rowsum[0] == sum(x[0])
        assert colsum[0] == sum(x[:, 0])
        assert len(rowsum) == 2
        assert len(colsum) == 3

    def test_getitem(self):
        x = VectorVariable((2, 4), "x")
        assert isinstance(x[0][0], Monomial)
        assert isinstance(x[0, 0], Monomial)

    def test_prod(self):
        x = VectorVariable(3, "x")
        m = x.prod()
        assert isinstance(m, Monomial)
        assert m == x[0] * x[1] * x[2]
        assert m == np.prod(x)
        pows = NomialArray([x[0], x[0] ** 2, x[0] ** 3])
        assert pows.prod() == x[0] ** 6

    def test_outer(self):
        x = VectorVariable(3, "x")
        y = VectorVariable(3, "y")
        assert np.outer(x, y) == x.outer(y)
        assert np.outer(y, x) == y.outer(x)
        assert isinstance(x.outer(y), NomialArray)

    def test_empty(self):
        x = VectorVariable(3, "x")
        # have to create this using slicing, to get object dtype
        empty_posy_array = x[:0]
        with pytest.raises(ValueError):
            empty_posy_array.sum()
        with pytest.raises(ValueError):
            empty_posy_array.prod()
        assert len(empty_posy_array) == 0
        assert empty_posy_array.ndim == 1
