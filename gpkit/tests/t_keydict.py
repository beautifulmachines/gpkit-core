"""Test KeyDict class"""

import unittest

import numpy as np

from gpkit import Variable, VectorVariable, ureg
from gpkit.tests.helpers import run_tests
from gpkit.varkey import VarKey
from gpkit.varmap import VarMap, _get_nested_item, _make_nested_list, _set_nested_item


class TestVarMap(unittest.TestCase):
    "TestCase for the VarMap class"

    def setUp(self):
        self.x = VarKey("x")
        self.y = VarKey("y")
        self.vm = VarMap()

    def test_nonnumeric(self):
        x = VectorVariable(2, "x")
        vm = VarMap()
        vm[x[1]] = "2"
        self.assertTrue(np.isnan(vm[x][0]))
        self.assertEqual(vm[x[1]], "2")
        self.assertNotIn(x[0], vm)
        self.assertIn(x[1], vm)

    def test_set_and_get(self):
        self.vm[self.x] = 1
        self.vm[self.y] = 2
        self.assertEqual(self.vm[self.x], 1)
        self.assertEqual(self.vm[self.y], 2)
        # get by string -- TBD if this should be allowed
        self.assertEqual(self.vm["x"], 1)
        self.assertEqual(self.vm["y"], 2)

    def test_getitem(self):
        x = Variable("x", lineage=[("Motor", 0)])
        self.vm[x] = 52
        self.assertEqual(self.vm[x], 52)
        self.assertEqual(self.vm[x.key], 52)
        self.assertEqual(self.vm["x"], 52)
        # self.assertEqual(self.vm["Motor.x"], 52)
        self.assertNotIn("x.Someothermodelname", self.vm)

    def test_failed_getitem(self):
        with self.assertRaises(KeyError):
            _ = self.vm["waldo"]
            # issue 893: failed __getitem__ caused state change
        self.assertNotIn("waldo", self.vm)
        waldo = Variable("waldo")
        self.vm.update({waldo: 5})
        res = self.vm["waldo"]
        self.assertEqual(res, 5)
        self.assertIn("waldo", self.vm)

    def test_keys_by_name(self):
        x2 = VarKey(name="x", units="ft")
        self.vm[self.x] = 1
        self.vm[x2] = 3
        vks = self.vm.keys_by_name("x")
        self.assertIn(self.x, vks)
        self.assertIn(x2, vks)
        self.assertEqual(len(vks), 2)

    def test_multiple_varkeys_same_name(self):
        self.vm[self.x] = 1
        self.vm[VarKey(name="x", units="ft")] = 3
        with self.assertRaises(KeyError):
            _ = self.vm["x"]  # Ambiguous

    def test_delitem(self):
        self.vm[self.x] = 1
        del self.vm[self.x]
        self.assertNotIn(self.x, self.vm)
        self.assertNotIn("x", self.vm)
        # Add two, delete one
        x2 = VarKey(name="x", units="ft")
        self.vm[self.x] = 1
        self.vm[x2] = 2
        del self.vm[self.x]
        self.assertIn(x2, self.vm)
        self.assertIn("x", self.vm)
        # now delete the second
        del self.vm[x2]
        self.assertNotIn("x", self.vm)

    def test_contains(self):
        self.vm[self.x] = 1
        self.assertIn(self.x, self.vm)
        self.assertIn("x", self.vm)
        self.assertNotIn(self.y, self.vm)
        self.assertNotIn("y", self.vm)

    def test_vector(self):
        x = VectorVariable(3, "x", "ft")
        vks = [v.key for v in x]
        vals = [4, 5, 6]
        for vk, val in zip(vks, vals):
            self.vm[vk] = val
        for vk, expected in zip(vks, vals):
            self.assertEqual(self.vm[vk], expected)
        self.assertEqual(self.vm[x], [4, 5, 6])
        self.assertEqual(self.vm["x"], [4, 5, 6])

    def test_vector_original(self):
        v = VectorVariable(3, "v")
        with self.assertRaises(NotImplementedError):
            # can't set by vector if keys not known
            self.vm[v] = np.array([2, 3, 4])
        self.assertEqual(v[0].key.idx, (0,))
        self.vm[v[0]] = 6
        self.assertEqual(self.vm[v][0], self.vm[v[0]])
        self.assertEqual(self.vm[v][0], 6)
        self.assertTrue(np.isnan(self.vm[v][1]))

    def test_vector_delitem(self):
        x = VectorVariable(3, "x", "ft")
        self.vm[x[0].key] = 1
        self.vm[x[1].key] = 2
        self.vm[x[2].key] = 3
        y = Variable("y", "kg")
        self.vm[y.key] = 5
        self.assertEqual(self.vm[x], [1, 2, 3])
        nan = float("nan")
        del self.vm[x[1].key]
        np.testing.assert_equal(self.vm[x], [1, nan, 3])
        del self.vm[x[0].key]
        np.testing.assert_equal(self.vm[x], [nan, nan, 3])
        del self.vm[x[2].key]
        self.assertNotIn(x, self.vm)
        self.assertIn(y, self.vm)

    def test_register_keys(self):
        self.vm[self.x] = 1
        self.vm.register_keys({self.y})
        self.assertIn("y", self.vm)
        with self.assertRaises(KeyError):
            _ = self.vm[self.y]
        with self.assertRaises(KeyError):
            _ = self.vm["y"]
        self.vm["y"] = 6
        self.assertEqual(self.vm["y"], 6)
        self.assertEqual(self.vm[self.y], 6)

    def test_setitem_variable(self):
        x = Variable("x")
        self.vm[x] = 6
        self.assertIn(x, self.vm)
        self.assertIn(x.key, self.vm)
        self.assertEqual(self.vm[x], 6)
        self.assertEqual(self.vm[x.key], 6)

    def test_setitem_unit(self):
        x = Variable("h", "inch")
        self.vm[x] = 8.0
        self.assertEqual(self.vm[x], 8.0)
        self.assertEqual(str(self.vm.quantity(x)), "8.0 inch")
        self.vm[x] = 6 * ureg.ft
        self.assertEqual(self.vm[x], 72)
        self.assertEqual(str(self.vm.quantity(x)), "72.0 inch")

    def test_setitem_lineage(self):
        x = Variable("x", lineage=(("test", 0),))
        self.vm[x] = 1
        self.assertIn(x, self.vm)
        self.assertEqual(set(self.vm), set([x.key]))


class TestNestedList(unittest.TestCase):
    "TestCase for nested list private methods in varmap.py"

    def test_make(self):
        x = _make_nested_list((2, 3))
        expected = [[None] * 3, [None] * 3]
        self.assertEqual(x, expected)

    def test_set_item(self):
        x = _make_nested_list((3, 2))
        _set_nested_item(x, (1, 0), 5)
        expected = [[None] * 2, [5, None], [None] * 2]
        self.assertEqual(x, expected)

    def test_get_item(self):
        x = [[2, 3], [4, 8], [1, 9]]
        self.assertEqual(_get_nested_item(x, (1, 1)), 8)
        self.assertEqual(_get_nested_item(x, (2, 1)), 9)
        self.assertEqual(_get_nested_item(x, (1, 0)), 4)

    def test_single_dim(self):
        x = _make_nested_list((4,))
        self.assertEqual(x, [None] * 4)
        _set_nested_item(x, (1,), 4)
        _set_nested_item(x, (2,), 3)
        self.assertEqual(x, [None, 4, 3, None])
        self.assertEqual(_get_nested_item(x, (0,)), None)
        self.assertEqual(_get_nested_item(x, (1,)), 4)
        self.assertEqual(_get_nested_item(x, (2,)), 3)
        self.assertEqual(_get_nested_item(x, (3,)), None)


TESTS = [TestVarMap, TestNestedList]


if __name__ == "__main__":  # pragma: no cover
    run_tests(TESTS)
