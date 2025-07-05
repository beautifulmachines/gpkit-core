import numpy as np
import pytest

from gpkit import Variable, VectorVariable
from gpkit.varmap import VarSet


# ---------- helpers ----------------------------------------------------------
@pytest.fixture
def scalar_and_vector():
    """Return a scalar VarKey 'x' and a 3-element vector VarKey 'X'."""
    x = Variable("x")
    X = VectorVariable(3, "X")
    return x, X


# ---------- basic container behaviour ---------------------------------------
def test_empty_initialisation():
    vs = VarSet()
    assert len(vs) == 0
    assert list(vs) == []
    assert "x" not in vs


def test_add_and_membership(scalar_and_vector):
    x, _ = scalar_and_vector
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


# ---------- vector handling --------------------------------------------------
def test_register_vector(scalar_and_vector):
    _, X = scalar_and_vector
    vs = VarSet()
    # update with vector elements (scalar VarKeys)
    vs.update([xx.key for xx in X])

    # expect: all three element keys registered
    assert len(vs) == 3
    for xx in X:
        assert xx.key in vs
    # name look-up should return parent
    nameset = vs.by_name("X")
    assert nameset == {X.key}

    # _by_vec mapping: parent key should yield an ndarray of element keys
    arr = vs._by_vec[X.key]  # private but worth sanity-checking
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    for a, b in zip(arr, X):
        assert a == b.key


# ---------- mutating behaviour ----------------------------------------------
def test_discard_and_len(scalar_and_vector):
    x, X = scalar_and_vector
    vs = VarSet([x.key, X[0].key, X[1].key])  # create with an iterable
    assert len(vs) == 3
    vs.discard(X[0].key)
    assert len(vs) == 2
    assert X[0] not in vs
    assert X[0].key not in vs
    # discarding a key that isn’t present should be silent
    vs.discard(X[2].key)
    assert len(vs) == 2


def test_varkeys_only():
    x = Variable("x")
    vs = VarSet()
    with pytest.raises(TypeError):
        vs.add(x)
    with pytest.raises(TypeError):
        vs.add("x")
