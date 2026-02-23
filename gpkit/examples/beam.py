"""
A simple beam example with fixed geometry. Solves the discretized
Euler-Bernoulli beam equations for a constant distributed load
"""

import numpy as np

from gpkit import Model, Var, Variable, Vectorize, ureg
from gpkit.util.small_scripts import mag

eps = 2e-4  # has to be quite large for consistent cvxopt printouts;
#  normally you'd set this to something more like 1e-20


class Beam(Model):
    """Discretization of the Euler beam equations for a distributed load."""

    EI = Var("N*m^2", "Bending stiffness")
    dx = Var("m", "Length of an element")
    L = Var("m", "Overall beam length", value=5)

    # Boundary condition variables (small nonzero to keep GP feasible)
    V_tip = Var("N", "Tip loading", value=eps)
    M_tip = Var("N*m", "Tip moment", value=eps)
    th_base = Var("-", "Base angle", value=eps)
    w_base = Var("m", "Base deflection", value=eps)

    def setup(self, N=4):
        # N-element distributed variables (N depends on setup arg, so use Vectorize)
        with Vectorize(N):
            q = Variable("q", 100, "N/m", "Distributed load")
            V = Variable("V", "N", "Internal shear")
            M = Variable("M", "N*m", "Internal moment")
            th = Variable("th", "-", "Slope")
            w = Variable("w", "m", "Displacement")

        # minimize tip displacement (the last w)
        self.cost = self.w_tip = w[-1]
        return {
            "definition of dx": self.L == (N - 1) * self.dx,
            "boundary_conditions": [
                V[-1] >= self.V_tip,
                M[-1] >= self.M_tip,
                th[0] >= self.th_base,
                w[0] >= self.w_base,
            ],
            # below: trapezoidal integration to form a piecewise-linear
            #        approximation of loading, shear, and so on
            # shear and moment increase from tip to base (left > right)
            "shear integration": V[:-1] >= V[1:] + 0.5 * self.dx * (q[:-1] + q[1:]),
            "moment integration": M[:-1] >= M[1:] + 0.5 * self.dx * (V[:-1] + V[1:]),
            # slope and displacement increase from base to tip (right > left)
            "theta integration": th[1:]
            >= th[:-1] + 0.5 * self.dx * (M[1:] + M[:-1]) / self.EI,
            "displacement integration": w[1:]
            >= w[:-1] + 0.5 * self.dx * (th[1:] + th[:-1]),
        }


b = Beam(N=6, substitutions={"L": 6, "EI": 1.1e4, "q": 110 * np.ones(6)})
sol = b.solve(verbosity=0)
print(sol.summary(vecn=6))
w_gp = sol["w"]  # deflection along beam

L, EI, q = sol["L"], sol["EI"], sol["q"]
x = np.linspace(0, mag(L), len(q)) * ureg.m  # position along beam
q = q[0]  # assume uniform loading for the check below
w_exact = q / (24 * EI) * x**2 * (x**2 - 4 * L * x + 6 * L**2)  # analytic soln
assert max(abs(w_gp - w_exact)) <= 1.1 * ureg.cm

PLOT = False
if PLOT:  # pragma: no cover
    import matplotlib.pyplot as plt

    x_exact = np.linspace(0, L, 1000)
    w_exact = q / (24 * EI) * x_exact**2 * (x_exact**2 - 4 * L * x_exact + 6 * L**2)
    plt.plot(x, w_gp, color="red", linestyle="solid", marker="^", markersize=8)
    plt.plot(x_exact, w_exact, color="blue", linestyle="dashed")
    plt.xlabel("x [m]")
    plt.ylabel("Deflection [m]")
    plt.axis("equal")
    plt.legend(["GP solution", "Analytical solution"])
    plt.show()
