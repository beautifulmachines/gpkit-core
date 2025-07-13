"""Unit tests for the VarSet class"""

import numpy as np
import pytest  # pylint: disable=import-error

from gpkit import Variable, VectorVariable
from gpkit.varmap import VarSet


# ---------- basic container behaviour ---------------------------------------
def test_empty_initialisation():
    vs = VarSet()
    assert len(vs) == 0
    assert not list(vs)
    assert "x" not in vs


def test_add_and_membership():
    x = Variable("x")
    vs = VarSet()
    vs.add(x.key)
    assert len(vs) == 1
    assert x.key in vs
    # membership by canonical name
    assert "x" in vs
    # by_name should return a set *containing* x
    assert vs.by_name("x") == {x.key}


def test_keys():
    x = Variable("x")
    y = Variable("y")
    vv = VectorVariable(3, "x")
    vs = VarSet({x.key, y.key})
    vs.add(vv[1].key)
    assert vs.keys("x") == {x.key, vv[1].key}
    assert vs.keys(x) == {x.key}
    assert vs.keys(vv) == {vv[1].key}  # because it's the only one we added
    assert vs.keys(vv[0]) == set()
    assert vs.keys("y") == {y.key}


def test_resolve():
    x = Variable("x", "m")
    x2 = Variable("x", "ft")
    z = Variable("z")
    vv = VectorVariable(4, "t")
    vs = VarSet({x.key, x2.key, z.key, vv.key})
    assert vs.resolve(vv) == vv.key
    assert vs.resolve(vv.key) == vv.key
    assert vs.resolve(z) == z.key
    assert vs.resolve("t") == vv.key
    with pytest.raises(KeyError):
        vs.resolve("x")
    with pytest.raises(KeyError):
        vs.resolve("y")


def test_clean():
    x = Variable("x")
    y = Variable("y")
    vs = VarSet({x.key, y.key})
    assert vs.clean({"x": 5, "y": 2}) == {x.key: 5, y.key: 2}


# ---------- vector handling --------------------------------------------------
def test_register_vector():
    vecx = VectorVariable(3, "X")
    vs = VarSet()
    # update with vector elements (scalar VarKeys)
    vs.update([xx.key for xx in vecx])

    # expect: all three element keys registered
    assert len(vs) == 3
    for xx in vecx:
        assert xx.key in vs
    # name look-up should return parent
    nameset = vs.by_name("X")
    assert nameset == {vecx.key}

    # by_vec mapping: parent key should yield an ndarray of element keys
    arr = vs.by_vec(vecx.key)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    for a, b in zip(arr, vecx):
        assert a == b.key


# ---------- mutating behaviour ----------------------------------------------
def test_discard_and_len():
    x = Variable("x")
    vecx = VectorVariable(3, "X")
    vs = VarSet([x.key, vecx[0].key, vecx[1].key])  # create with an iterable
    assert len(vs) == 3
    vs.discard(vecx[0].key)
    assert len(vs) == 2
    assert vecx[0] not in vs
    assert vecx[0].key not in vs
    # discarding a key that isn’t present should be silent
    vs.discard(vecx[2].key)
    vs.discard(Variable("y").key)
    assert len(vs) == 2


def test_varkeys_only():
    x = Variable("x")
    vs = VarSet()
    with pytest.raises(TypeError):
        vs.add(x)
    with pytest.raises(TypeError):
        vs.add("x")
