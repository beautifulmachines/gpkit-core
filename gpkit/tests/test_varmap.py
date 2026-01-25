"""Test KeyDict class"""

import numpy as np
import pytest

from gpkit import Variable, VectorVariable, ureg
from gpkit.varkey import VarKey
from gpkit.varmap import VarMap


class TestVarMap:
    "TestCase for the VarMap class"

    def setup_method(self):
        self.x = VarKey("x")
        self.y = VarKey("y")
        self.vm = VarMap()

    def test_set_and_get(self):
        self.vm[self.x] = 1
        self.vm[self.y] = 2
        assert self.vm[self.x] == 1
        assert self.vm[self.y] == 2
        # get by string -- TBD if this should be allowed
        assert self.vm["x"] == 1
        assert self.vm["y"] == 2

    def test_contains(self):
        self.vm[self.x] = 1
        assert self.x in self.vm
        assert "x" in self.vm
        assert self.y not in self.vm
        assert "y" not in self.vm

    def test_getitem(self):
        x = Variable("x", lineage=[("Motor", 0)])
        self.vm[x] = 52
        assert self.vm[x] == 52
        assert self.vm[x.key] == 52
        assert self.vm["x"] == 52
        # assert self.vm["Motor.x"] == 52
        assert "Someothermodelname.x" not in self.vm

    def test_failed_getitem(self):
        with pytest.raises(KeyError):
            _ = self.vm["waldo"]
            # issue 893: failed __getitem__ caused state change
        assert "waldo" not in self.vm
        self.vm.update({Variable("waldo"): 5})
        assert self.vm["waldo"] == 5
        assert "waldo" in self.vm

    def test_keys_by_name(self):
        x2 = VarKey(name="x", units="ft")
        self.vm[self.x] = 1
        self.vm[x2] = 3
        vks = self.vm.varset.by_name("x")
        assert self.x in vks
        assert x2 in vks
        assert len(vks) == 2

    def test_multiple_varkeys_same_name(self):
        self.vm[self.x] = 1
        self.vm[VarKey(name="x", units="ft")] = 3
        with pytest.raises(KeyError):
            _ = self.vm["x"]  # Ambiguous

    def test_delitem(self):
        self.vm[self.x] = 1
        del self.vm[self.x]
        assert self.x not in self.vm
        assert "x" not in self.vm
        # Add two, delete one
        x2 = VarKey(name="x", units="ft")
        self.vm[self.x] = 1
        self.vm[x2] = 2
        del self.vm[self.x]
        assert x2 in self.vm
        assert "x" in self.vm
        # now delete the second
        del self.vm[x2]
        assert "x" not in self.vm

    def test_vector(self):
        x = VectorVariable(3, "x", "ft")
        vks = [v.key for v in x]
        vals = [4, 5, 6]
        for vk, val in zip(vks, vals):
            self.vm[vk] = val
        for vk, expected in zip(vks, vals):
            assert self.vm[vk] == expected
        assert self.vm[x] == [4, 5, 6]
        assert self.vm["x"] == [4, 5, 6]

    def test_vector_partial(self):
        v = VectorVariable(3, "v")
        with pytest.raises(NotImplementedError):
            # can't set by vector if keys not known
            self.vm[v] = np.array([2, 3, 4])
        assert v[0].key.idx == (0,)  # legacy; belongs elsewhere
        self.vm[v[0]] = 6
        assert self.vm[v][0] == self.vm[v[0]]
        assert self.vm[v][0] == 6
        assert np.isnan(self.vm[v][1])
        del self.vm[v[0]]
        with pytest.raises(KeyError):
            _ = self.vm[v]

    def test_vector_delitem(self):
        x = VectorVariable(3, "x", "ft")
        self.vm[x[0].key] = 1
        self.vm[x[1].key] = 2
        self.vm[x[2].key] = 3
        y = Variable("y", "kg")
        self.vm[y.key] = 5
        assert self.vm[x] == [1, 2, 3]
        nan = float("nan")
        del self.vm[x[1].key]
        np.testing.assert_equal(self.vm[x], [1, nan, 3])
        del self.vm[x[0].key]
        np.testing.assert_equal(self.vm[x], [nan, nan, 3])
        del self.vm[x[2].key]
        assert x not in self.vm
        assert y in self.vm

    def test_register_keys(self):
        self.vm[self.x] = 1
        self.vm.register_keys({self.y})
        assert "y" in self.vm
        with pytest.raises(KeyError):
            _ = self.vm[self.y]
        with pytest.raises(KeyError):
            _ = self.vm["y"]
        self.vm["y"] = 6
        assert self.vm["y"] == 6
        assert self.vm[self.y] == 6

    def test_setitem_variable(self):
        x = Variable("x")
        self.vm[x] = 6
        assert x in self.vm
        assert x.key in self.vm
        assert self.vm[x] == 6
        assert self.vm[x.key] == 6

    def test_setitem_unit(self):
        x = Variable("h", "inch")
        self.vm[x] = 8.0
        assert self.vm[x] == 8.0
        assert str(self.vm.quantity(x)) == "8.0 inch"
        self.vm[x] = 6 * ureg.ft
        assert self.vm[x] == 72
        assert str(self.vm.quantity(x)) == "72.0 inch"

    def test_nonnumeric(self):
        x = VectorVariable(2, "x")
        self.vm[x[1]] = "2"
        assert np.isnan(self.vm[x][0])
        assert self.vm[x[1]] == "2"
        assert x[0] not in self.vm
        assert x[1] in self.vm

    def test_setitem_lineage(self):
        x = Variable("x", lineage=(("test", 0),))
        self.vm[x] = 1
        assert x in self.vm
        assert set(self.vm) == set([x.key])
