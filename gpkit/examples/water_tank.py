"Minimizes rectangular tank surface area for a particular volume."

from gpkit import Model, Var, VectorVariable


class WaterTank(Model):
    "Minimize surface area of a rectangular tank for a fixed water volume."

    M = Var("kg", "mass of water", value=100)
    rho = Var("kg/m^3", "density of water", value=1000)
    A = Var("m^2", "surface area")
    V = Var("m^3", "volume")

    def setup(self):
        d = VectorVariable(3, "d", "m", "dimension vector")

        self.cost = self.A
        return [
            self.A >= 2 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]),
            self.V == d[0] * d[1] * d[2],
            self.M == self.V * self.rho,
        ]


if __name__ == "__main__":
    m = WaterTank()
    sol = m.solve(verbosity=0)
    print(sol.summary())
