"""Test VarKey, Variable, VectorVariable, and ArrayVariable classes"""

import sys

import numpy as np
import pytest

import gpkit
from gpkit import (
    ArrayVariable,
    Monomial,
    NomialArray,
    Variable,
    VarKey,
    Vectorize,
    VectorVariable,
)
from gpkit.nomials import Variable as PlainVariable


class TestVarKey:
    """TestCase for the VarKey class"""

    def test_init(self):
        """Test VarKey initialization"""
        # test no-name init
        _ = ArrayVariable(1)
        # test protected field
        with pytest.raises(ValueError):
            _ = ArrayVariable(1, idx=5)
        # test type
        x = VarKey("x")
        assert isinstance(x, VarKey)
        # test no args
        x = VarKey()
        assert isinstance(x, VarKey)
        y = VarKey(**x.descr)
        assert x == y
        # test special 'name' keyword overwriting behavior
        x = VarKey("x", flavour="vanilla")
        assert x.name == "x"
        x = VarKey(name="x")
        assert x.name == "x"
        # pylint: disable=redundant-keyword-arg
        with pytest.raises(TypeError):
            VarKey("x", name="y")
        assert isinstance(x.latex(), str)
        assert isinstance(x.latex_unitstr(), str)
        # test index latex printing
        y = VectorVariable(2, "y")
        assert y[0].key.latex() == "{\\vec{y}}_{0}"

    def test_ast(self):  # pylint: disable=too-many-statements
        if sys.platform[:3] == "win":  # pragma: no cover
            return

        t = Variable("t")
        u = Variable("u")
        v = Variable("v")
        w = Variable("w")
        x = VectorVariable(3, "x")
        y = VectorVariable(3, "y")
        z = VectorVariable(3, "z")
        a = VectorVariable((3, 2), "a")

        assert str(3 * (x + y) * z) == "3·(x[:] + y[:])·z[:]"
        nni = 3
        ii = np.tile(np.arange(1, nni + 1), a.shape[1:] + (1,)).T
        assert str(w * NomialArray(ii) / nni)[:4] == "w·[["
        assert str(w * NomialArray(ii) / nni)[-4:] == "]]/3"
        assert str(NomialArray(ii) * w / nni)[:2] == "[["
        assert str(NomialArray(ii) * w / nni)[-6:] == "]]·w/3"
        assert str(w * ii / nni)[:4] == "w·[["
        assert str(w * ii / nni)[-4:] == "]]/3"
        assert str(w * (ii / nni))[:4] == "w·[["
        assert str(w * (ii / nni))[-2:] == "]]"
        assert str(w >= (x[0] * t + x[1] * u) / v) == "w ≥ (x[0]·t + x[1]·u)/v"
        assert str(x) == "x[:]"
        assert str(x * 2) == "x[:]·2"
        assert str(2 * x) == "2·x[:]"
        assert str(x + 2) == "x[:] + 2"
        assert str(2 + x) == "2 + x[:]"
        assert str(x / 2) == "x[:]/2"
        assert str(2 / x) == "2/x[:]"
        assert str(x**3) == "x[:]³"
        assert str(-x) == "-x[:]"
        assert str(x / y / z) == "x[:]/y[:]/z[:]"
        assert str(x / (y / z)) == "x[:]/(y[:]/z[:])"
        assert str(x <= y) == "x[:] ≤ y[:]"
        assert str(x >= y + z) == "x[:] ≥ y[:] + z[:]"
        assert str(x[:2]) == "x[:2]"
        assert str(x[:]) == "x[:]"
        assert str(x[1:]) == "x[1:]"
        assert str(y * [1, 2, 3]) == "y[:]·[1, 2, 3]"
        assert str(x[:2] == (y * [1, 2, 3])[:2]) == "x[:2] = (y[:]·[1, 2, 3])[:2]"
        assert str(y + [1, 2, 3]) == "y[:] + [1, 2, 3]"
        assert str(x == y + [1, 2, 3]) == "x[:] = y[:] + [1, 2, 3]"
        assert str(x >= y + [1, 2, 3]) == "x[:] ≥ y[:] + [1, 2, 3]"
        assert str(a[:, 0]) == "a[:,0]"
        assert str(a[2, :]) == "a[2,:]"
        g = 1 + 3 * a[2, 0] ** 2
        gstrbefore = str(g)
        g.ast = None
        gstrafter = str(g)
        assert gstrbefore == gstrafter

        cstr = str(2 * a >= a + np.ones((3, 2)) / 2)
        assert cstr == """2·a[:] ≥ a[:] + [[0.5 0.5]
           [0.5 0.5]
           [0.5 0.5]]"""

    def test_eq_neq(self):
        """Test boolean equality operators"""
        # no args
        vk1 = VarKey()
        vk2 = VarKey()
        assert vk1 != vk2
        # pylint: disable=unnecessary-negation  # testing __eq__ returns False
        assert not vk1 == vk2
        assert vk1 == vk1  # pylint: disable=comparison-with-itself
        v = VarKey("v")
        vel = VarKey("v")
        assert v == vel
        # pylint: disable=unnecessary-negation  # testing __ne__ returns False
        assert not v != vel
        assert vel == vel  # pylint: disable=comparison-with-itself
        x1 = Variable("x", 3, "m")
        x2 = Variable("x", 2, "ft")
        x3 = Variable("x", 2, "m")
        assert x2.key != x3.key
        assert x1.key == x3.key

    def test_repr(self):
        """Test __repr__ method"""
        for k in ("x", "$x$", "var_name", "var name", r"\theta", r"$\pi_{10}$"):
            var = VarKey(k)
            assert repr(var) == k

    def test_dict_key(self):
        """make sure variables are well-behaved dict keys"""
        v = VarKey()
        x = VarKey("$x$")
        d = {v: 1273, x: "foo"}
        assert d[v] == 1273
        assert d[x] == "foo"
        d = {VarKey(): None, VarKey(): 12}
        assert len(d) == 2

    def test_units_attr(self):
        """Make sure VarKey objects have a units attribute"""
        x = VarKey("x")
        for vk in (VarKey(), x, VarKey(**x.descr), VarKey(units="m")):
            assert "units" in vk.descr

    def test_hash_vector(self):
        """Make sure different vector keys don't have hash collisions"""
        t = VarKey("t")
        vec2 = VarKey("t", shape=(2,))
        vec3 = VarKey("t", shape=(3,))
        el2 = VarKey("t", shape=(2,), idx=(1,))
        el3 = VarKey("t", shape=(3,), idx=(1,))
        assert hash(t) != hash(vec3)
        assert hash(vec2) != hash(vec3)
        assert hash(el3) != hash(vec3)
        assert hash(el3) != hash(el2)
        assert hash(t) != hash(el2)


class TestVariable:
    """TestCase for the Variable class"""

    def test_init(self):
        """Test Variable initialization"""
        v = Variable("v")
        assert isinstance(v, PlainVariable)
        assert isinstance(v, Monomial)
        # test that operations on Variable cast to Monomial
        assert isinstance(3 * v, Monomial)
        assert not isinstance(3 * v, PlainVariable)

    def test_value(self):
        """Detailed tests for value kwarg of __init__"""
        a = Variable("a")
        b = Variable("b", value=4)
        c = a**2 + b
        assert b.value == 4
        assert isinstance(b.value, float)
        p1 = c.value
        p2 = a**2 + 4
        assert p1 == p2
        assert a.value == a

    def test_hash(self):
        x1 = Variable("x", "-", "first x")
        x2 = Variable("x", "-", "second x")
        assert hash(x1) == hash(x2)
        p1 = Variable("p", "psi", "first pressure")
        p2 = Variable("p", "psi", "second pressure")
        assert hash(p1) == hash(p2)
        xu = Variable("x", "m", "x with units")
        assert hash(x1) != hash(xu)

    def test_unit_parsing(self):
        x = Variable("x", "s^0.5/m^0.5")
        y = Variable("y", "(m/s)^-0.5")
        assert x.units == y.units

    def test_to(self):
        x = Variable("x", "ft")
        assert x.to("inch").c.magnitude == 12

    def test_eq_ne(self):
        # test for #1138
        w = Variable("W", 5, "lbf", "weight of 1 bag of sugar")
        assert w != w.key
        assert w.key != w
        # pylint: disable=unnecessary-negation  # testing __eq__ both operand orders
        assert not w == w.key
        assert not w.key == w


class TestVectorVariable:
    """TestCase for the VectorVariable class.
    Note: more relevant tests in t_posy_array."""

    def test_init(self):
        """Test VectorVariable initialization"""
        # test 1
        n = 3
        v = VectorVariable(n, "v", label="dummy variable")
        assert isinstance(v, NomialArray)
        v_mult = 3 * v
        for i in range(n):
            assert isinstance(v[i], PlainVariable)
            assert isinstance(v[i], Monomial)
            # test that operations on Variable cast to Monomial
            assert isinstance(v_mult[i], Monomial)
            assert not isinstance(v_mult[i], PlainVariable)

    def test_nomial_array_comp(self):
        x = VectorVariable(3, "x", label="dummy variable")
        x_0 = Variable("x", idx=(0,), shape=(3,), label="dummy variable")
        x_1 = Variable("x", idx=(1,), shape=(3,), label="dummy variable")
        x_2 = Variable("x", idx=(2,), shape=(3,), label="dummy variable")
        x2 = NomialArray([x_0, x_1, x_2])
        assert x == x2

    def test_issue_137(self):
        n = 20
        x_arr = np.arange(0, 5, 5 / n) + 1e-6
        x = VectorVariable(n, "x", x_arr, "m", "Beam Location")

        with pytest.raises(ValueError):
            _ = VectorVariable(2, "x", [1, 2, 3])

        with Vectorize(2):
            x = VectorVariable(3, "x", np.array([13, 15, 17]))
            assert x[0, 0].value == 13
            assert x[1, 0].value == 15
            assert x[2, 0].value == 17
            assert x[0, 0].value == x[0, 1].value
            assert x[1, 0].value == x[1, 1].value
            assert x[2, 0].value == x[2, 1].value

    def test_vectorize_shapes(self):
        with gpkit.Vectorize(3):
            with gpkit.Vectorize(5):
                y = gpkit.Variable("y")
                x = gpkit.VectorVariable(2, "x")
            z = gpkit.VectorVariable(7, "z")

        assert y.shape == (5, 3)
        assert x.shape == (2, 5, 3)
        assert z.shape == (7, 3)


class TestArrayVariable:
    """TestCase for the ArrayVariable class"""

    def test_is_vector_variable(self):
        """
        Make sure ArrayVariable is a shortcut to VectorVariable
        (we want to know if this changes).
        """
        assert ArrayVariable is VectorVariable

    def test_str(self):
        """Make sure string looks something like a numpy array"""
        x = ArrayVariable((2, 4), "x")
        assert str(x) == "x[:]"
