"""Test KeyDict class"""

import numpy as np
import pytest

from gpkit import Variable, VectorVariable, ureg
from gpkit.varkey import VarKey
from gpkit.varmap import VarMap, _compute_collision_depths


@pytest.fixture(name="vm")
def fixture_vm():
    """Fresh VarMap for each test."""
    return VarMap()


@pytest.fixture(name="x")
def fixture_x():
    """VarKey named 'x'."""
    return VarKey("x")


@pytest.fixture(name="y")
def fixture_y():
    """VarKey named 'y'."""
    return VarKey("y")


class TestVarMap:
    "TestCase for the VarMap class"

    def test_set_and_get(self, vm, x, y):
        vm[x] = 1
        vm[y] = 2
        assert vm[x] == 1
        assert vm[y] == 2
        # get by string -- TBD if this should be allowed
        assert vm["x"] == 1
        assert vm["y"] == 2

    def test_contains(self, vm, x, y):
        vm[x] = 1
        assert x in vm
        assert "x" in vm
        assert y not in vm
        assert "y" not in vm

    def test_getitem(self, vm):
        x = Variable("x", lineage=[("Motor", 0)])
        vm[x] = 52
        assert vm[x] == 52
        assert vm[x.key] == 52
        assert vm["x"] == 52
        # assert vm["Motor.x"] == 52
        assert "Someothermodelname.x" not in vm

    def test_failed_getitem(self, vm):
        with pytest.raises(KeyError):
            _ = vm["waldo"]
            # issue 893: failed __getitem__ caused state change
        assert "waldo" not in vm
        vm.update({Variable("waldo"): 5})
        assert vm["waldo"] == 5
        assert "waldo" in vm

    def test_keys_by_name(self, vm, x):
        x2 = VarKey(name="x", units="ft")
        vm[x] = 1
        vm[x2] = 3
        vks = vm.varset.by_name("x")
        assert x in vks
        assert x2 in vks
        assert len(vks) == 2

    def test_multiple_varkeys_same_name(self, vm, x):
        vm[x] = 1
        vm[VarKey(name="x", units="ft")] = 3
        with pytest.raises(KeyError):
            _ = vm["x"]  # Ambiguous

    def test_delitem(self, vm, x):
        vm[x] = 1
        del vm[x]
        assert x not in vm
        assert "x" not in vm
        # Add two, delete one
        x2 = VarKey(name="x", units="ft")
        vm[x] = 1
        vm[x2] = 2
        del vm[x]
        assert x2 in vm
        assert "x" in vm
        # now delete the second
        del vm[x2]
        assert "x" not in vm

    def test_vector(self, vm):
        x = VectorVariable(3, "x", "ft")
        vks = [v.key for v in x]
        vals = [4, 5, 6]
        for vk, val in zip(vks, vals):
            vm[vk] = val
        for vk, expected in zip(vks, vals):
            assert vm[vk] == expected
        assert vm[x] == [4, 5, 6]
        assert vm["x"] == [4, 5, 6]

    def test_vector_partial(self, vm):
        v = VectorVariable(3, "v")
        with pytest.raises(NotImplementedError):
            # can't set by vector if keys not known
            vm[v] = np.array([2, 3, 4])
        assert v[0].key.idx == (0,)  # legacy; belongs elsewhere
        vm[v[0]] = 6
        assert vm[v][0] == vm[v[0]]
        assert vm[v][0] == 6
        assert np.isnan(vm[v][1])
        del vm[v[0]]
        with pytest.raises(KeyError):
            _ = vm[v]

    def test_vector_delitem(self, vm):
        x = VectorVariable(3, "x", "ft")
        vm[x[0].key] = 1
        vm[x[1].key] = 2
        vm[x[2].key] = 3
        y = Variable("y", "kg")
        vm[y.key] = 5
        assert vm[x] == [1, 2, 3]
        nan = float("nan")
        del vm[x[1].key]
        np.testing.assert_equal(vm[x], [1, nan, 3])
        del vm[x[0].key]
        np.testing.assert_equal(vm[x], [nan, nan, 3])
        del vm[x[2].key]
        assert x not in vm
        assert y in vm

    def test_register_keys(self, vm, x, y):
        vm[x] = 1
        vm.register_keys({y})
        assert "y" in vm
        with pytest.raises(KeyError):
            _ = vm[y]
        with pytest.raises(KeyError):
            _ = vm["y"]
        vm["y"] = 6
        assert vm["y"] == 6
        assert vm[y] == 6

    def test_setitem_variable(self, vm):
        x = Variable("x")
        vm[x] = 6
        assert x in vm
        assert x.key in vm
        assert vm[x] == 6
        assert vm[x.key] == 6

    def test_setitem_unit(self, vm):
        x = Variable("h", "inch")
        vm[x] = 8.0
        assert vm[x] == 8.0
        assert str(vm.quantity(x)) == "8.0 inch"
        vm[x] = 6 * ureg.ft
        assert vm[x] == 72
        assert str(vm.quantity(x)) == "72.0 inch"

    def test_nonnumeric(self, vm):
        x = VectorVariable(2, "x")
        vm[x[1]] = "2"
        assert np.isnan(vm[x][0])
        assert vm[x[1]] == "2"
        assert x[0] not in vm
        assert x[1] in vm

    def test_setitem_lineage(self, vm):
        x = Variable("x", lineage=(("test", 0),))
        vm[x] = 1
        assert x in vm
        assert set(vm) == set([x.key])


class TestComputeCollisionDepths:
    """Tests for _compute_collision_depths in varmap.py."""

    def test_simple_depth1_resolution(self):
        """Two vars with distinct innermost model names resolve at depth 1."""
        vk_a = VarKey("m", lineage=(("SubA", 0),))
        vk_b = VarKey("m", lineage=(("SubB", 0),))
        result = _compute_collision_depths({"m": {vk_a, vk_b}})
        assert result[vk_a] == 1
        assert result[vk_b] == 1

    def test_asymmetric_lineage_no_indexerror(self):
        """Regression: IndexError when shorter lineage is a suffix of a longer one.

        The same model class in two different parent contexts each get num=0:
          standalone Inner: lineagestr "Inner0" (1 component)
          nested Outer.Inner: lineagestr "Outer0.Inner0" (2 components)
        Both have "Inner0" at depth 1 → collision. At depth 2, the standalone
        key only has 1 lineage component, so lineages[-2] raised IndexError.
        With the fix, the shorter lineage falls back to its full lineagestr,
        which differs from the longer one, resolving the collision.
        """
        vk_short = VarKey("x", lineage=(("Inner", 0),))
        vk_long = VarKey("x", lineage=(("Outer", 0), ("Inner", 0)))
        # Must not raise IndexError
        result = _compute_collision_depths({"x": {vk_short, vk_long}})
        assert vk_short in result
        assert vk_long in result

        # The resolved display strings must be distinct
        def display_str(vk, depth):
            parts = vk.lineagestr().split(".")
            return vk.lineagestr() if depth > len(parts) else ".".join(parts[-depth:])

        assert display_str(vk_short, result[vk_short]) != display_str(
            vk_long, result[vk_long]
        )
