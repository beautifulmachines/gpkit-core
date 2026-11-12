"""A box-volume GP whose optimum is nearly degenerate, kept as a test fixture.

From Section 2.4 of Boyd et al.'s GP tutorial. With these numbers the floor
aspect ratio constraint d/w >= gamma is almost tight (d/w = 2.0018 against 2)
at a dual of about 3e-4, which exercises how near-zero sensitivities are
handled. The numbers match the paper, so they are not tuned away.
"""

from gpkit import Model, Var


class Box(Model):
    "Maximize box volume given wall area, floor area, and aspect ratio constraints."

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
