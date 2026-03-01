"Maximizes box volume given area and aspect ratio constraints."

from gpkit import Model, Var


class Box(Model):
    "Box volume maximization."

    # Parameters
    alpha = Var("-", "lower limit, wall aspect ratio", value=2)
    beta = Var("-", "upper limit, wall aspect ratio", value=10)
    gamma = Var("-", "lower limit, floor aspect ratio", value=2)
    delta = Var("-", "upper limit, floor aspect ratio", value=10)
    A_wall = Var("m^2", "upper limit, wall area", value=200)
    A_floor = Var("m^2", "upper limit, floor area", value=50)

    # Free variables
    h = Var("m", "height")
    w = Var("m", "width")
    d = Var("m", "depth")

    def setup(self):
        h, w, d = self.h, self.w, self.d
        self.cost = 1 / (h * w * d)
        return [
            self.A_wall >= 2 * h * w + 2 * h * d,
            self.A_floor >= w * d,
            h / w >= self.alpha,
            h / w <= self.beta,
            d / w >= self.gamma,
            d / w <= self.delta,
        ]


m = Box()
print(m.solve(verbosity=0).table())
