"Tests non-array linked functions & subs in a vectorization environment"

import numpy as np

from gpkit import ConstraintSet, Model, Variable, Vectorize


class Vehicle(Model):
    "Vehicle model"

    def setup(self):
        self.a = a = Variable("a")
        constraints = [a >= 1]
        return constraints


class System(Model):
    "System model"

    def setup(self):
        with Vectorize(1):
            self.Fleet2 = Fleet2()
        constraints = [self.Fleet2]
        self.cost = sum(self.Fleet2.z)
        return constraints


class Fleet2(Model):
    "Fleet model (composed of multiple Vehicles)"

    def setup(self):
        x = Variable("x")

        def myfun(c):
            return [np.array(c[x]) - 1, np.ones(x.shape)]

        with Vectorize(2):
            y = Variable("y", myfun)
            self.Vehicle = Vehicle()

        self.z = z = Variable("z")
        substitutions = {"x": 4}
        constraints = [
            z >= sum(y / x * self.Vehicle.a),
            self.Vehicle,
        ]
        return constraints, substitutions


m = System()
sol = m.solve(verbosity=0)
print(sol.table())


# now with more fleets per system
class System2(Model):
    "System model"

    def setup(self):
        with Vectorize(3):
            self.Fleet2 = Fleet2()
        constraints = [self.Fleet2]
        self.cost = sum(self.Fleet2.z)
        return constraints


m = System2()
sol = m.solve(verbosity=0)
print(sol.table())


# now testing substitutions


class Simple(Model):
    "Simple model"

    def setup(self):
        self.x = x = Variable("x")
        y = Variable("y", 1)
        z = Variable("z", 2)
        constraints = [
            x >= y + z,
        ]
        return constraints


class Cake(Model):
    "Cake model"

    def setup(self):
        with Vectorize(3):
            s = Simple()
        c = ConstraintSet([s])
        self.cost = s.x.sum()
        return c


m = Cake()
m.substitutions.update({"z": lambda v: np.array(v["y"]) ** 2})
sol = m.sweep({"y": [[1, 2, 3]]}, verbosity=0)
print(sol.table())
