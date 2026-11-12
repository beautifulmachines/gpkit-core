"""Regression tests for specific past bugs, kept as small whole models."""

import numpy as np
import pytest

from gpkit import ConstraintSet, Model, Variable, Vectorize
from gpkit.tests.degenerate_box import Box


def test_nearly_degenerate_box_solves():
    "A near-tight constraint at near-zero dual still solves to the known optimum"
    m = Box()
    sol = m.solve(verbosity=0)
    assert sol.cost.magnitude == pytest.approx(0.003674, rel=0.01)
    assert sol.sens.variables[m.gamma] == pytest.approx(0, abs=1e-2)


def test_linked_functions_and_subs_in_vectorization():
    "Non-array linked functions and substitutions inside Vectorize (gpkit #1513)"

    class Vehicle(Model):
        "Vehicle model"

        def setup(self):
            self.a = a = Variable("a")
            return [a >= 1]

    class Fleet2(Model):
        "Fleet model (composed of multiple Vehicles)"

        def setup(self):
            x = Variable("x")

            def myfun(c):
                return [np.array(c[x]) - 1, np.ones(x.key.shape)]

            with Vectorize(2):
                y = Variable("y", myfun)
                self.Vehicle = Vehicle()

            self.z = z = Variable("z")
            constraints = [z >= sum(y / x * self.Vehicle.a), self.Vehicle]
            return constraints, {"x": 4}

    class System(Model):
        "System model"

        def setup(self):
            with Vectorize(1):
                self.Fleet2 = Fleet2()
            self.cost = sum(self.Fleet2.z)
            return [self.Fleet2]

    class System2(Model):
        "System model with more fleets"

        def setup(self):
            with Vectorize(3):
                self.Fleet2 = Fleet2()
            self.cost = self.Fleet2.z.sum()
            return [self.Fleet2]

    System().solve(verbosity=0)
    System2().solve(verbosity=0)

    class Simple(Model):
        "Simple model"

        def setup(self):
            self.x = x = Variable("x")
            y = Variable("y", 1)
            z = Variable("z", 2)
            return [x >= y + z]

    class Cake(Model):
        "Cake model"

        def setup(self):
            with Vectorize(3):
                s = Simple()
            self.cost = s.x.sum()
            return ConstraintSet([s])

    m = Cake()
    m.substitutions.update({"z": lambda v: np.array(v["y"]) ** 2})
    sol = m.sweep({"y": [[1, 2, 3]]}, verbosity=0)
    assert sol[0].cost == pytest.approx(20.0, rel=1e-2)


def test_returned_substitutions_broadcast_over_vectorized_submodels():
    "Substitutions returned from setup() broadcast into Vectorize (gpkit #1522)"

    def broadcast_substitution(key, array):
        "Broadcasts input into the shape of a given key"
        return np.broadcast_to(array, reversed(key.key.shape)).T

    class Pie(Model):
        "Pie model"

        def setup(self):
            self.x = x = Variable("x")
            z = Variable("z")
            return [x >= z], {"z": 1}

    class Cake(Model):
        "Cake model, containing a vector of Pies"

        def setup(self):
            self.y = y = Variable("y")
            with Vectorize(2):
                s = Pie()
            constraints = [y >= s.x, s]
            return constraints, {"x": broadcast_substitution(s.x, [2, 3])}

    class Yum(Model):
        "Dessert system containing n Cakes"

        def setup(self, n):
            with Vectorize(n):
                cake = Cake()
            self.cost = cake.y.sum()
            return ConstraintSet([cake])

    Yum(5).solve(verbosity=0)
    sol = Yum(1).solve(verbosity=0)
    assert sol.cost == pytest.approx(3.0, rel=1e-2)
