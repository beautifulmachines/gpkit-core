"""Tests for utils module"""

import pytest

from gpkit import Variable, units
from gpkit.util.repr_conventions import unitstr
from gpkit.util.small_classes import HashVector


class TestHashVector:
    """TestCase for the HashVector class"""

    def test_init(self):
        """Make sure HashVector acts like a dict"""
        # args and kwargs
        hv = HashVector([(2, 3), (1, 10)], dog="woof")
        assert isinstance(hv, dict)
        assert hv == {2: 3, 1: 10, "dog": "woof"}
        # no args
        assert not HashVector()
        # creation from dict
        assert HashVector({"x": 7}) == {"x": 7}

    def test_neg(self):
        """Test negation"""
        hv = HashVector(x=7, y=0, z=-1)
        assert -hv == {"x": -7, "y": 0, "z": 1}

    def test_pow(self):
        """Test exponentiation"""
        hv = HashVector(x=4, y=0, z=1)
        assert hv**0.5 == {"x": 2, "y": 0, "z": 1}
        with pytest.raises(TypeError):
            _ = hv**hv
        with pytest.raises(TypeError):
            _ = hv ** "a"

    def test_mul_add(self):
        """Test multiplication and addition"""
        a = HashVector(x=1, y=7)
        b = HashVector()
        c = HashVector(x=3, z=4)
        # nonsense multiplication
        with pytest.raises(TypeError):
            _ = a * set()
        # multiplication and addition by scalars
        r = a * 0
        assert r == HashVector(x=0, y=0)
        assert isinstance(r, HashVector)
        r = a - 2
        assert r == HashVector(x=-1, y=5)
        assert isinstance(r, HashVector)
        with pytest.raises(TypeError):
            _ = r + "a"
        # multiplication and addition by dicts
        assert a + b == a
        assert a + b + c == HashVector(x=4, y=7, z=4)


class TestSmallScripts:
    """TestCase for gpkit.small_scripts"""

    def test_unitstr(self):
        x = Variable("x", "ft")
        # pint issue 356
        footstrings = ("ft", "foot")  # backwards compatibility with pint 0.6
        assert unitstr(Variable("n", "count")) == "count"
        assert unitstr(x) in footstrings
        assert unitstr(x.key) in footstrings
        assert unitstr(Variable("y"), dimless="---") == "---"
        assert unitstr(None, dimless="--") == "--"

    def test_pint_366(self):
        # test for https://github.com/hgrecco/pint/issues/366
        assert unitstr(units("nautical_mile")) in ("nmi", "nautical_mile")
        assert units("nautical_mile") == units("nmi")
