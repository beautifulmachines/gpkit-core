"""Composite pressure vessel: a signomial constraint with a guessed first split"""

import numpy as np

from gpkit import Model, Variable, guess

r = Variable("r", "m", "radius")
L = Variable("L", "m", "overall length, end caps included")
t = Variable("t", "m", "cylinder wall thickness")
m = Variable("m", "kg", "mass")
P = Variable("P", 200e5, "Pa", "pressure")
V = Variable("V", 0.01, "m^3", "internal volume")
sf = Variable("sf", 2, "-", "safety factor")
sigma = Variable("sigma", 1.2e9, "Pa", "composite strength")
rho_c = Variable("rho_c", 1600, "kg/m^3", "composite density")
rho_l = Variable("rho_l", 4430, "kg/m^3", "liner density")
t_l = Variable("t_l", 0.5e-3, "m", "liner thickness")

# the hemispherical caps carry half the cylinder's hoop stress, so they are
# half as thick, and their mass is subtracted from the cylinder's
cap = 2 * np.pi * rho_c * r**2 * t
cyl = 2 * np.pi * rho_c * r * t * L
liner = 2 * np.pi * rho_l * r * L * t_l

vessel = Model(
    m,
    [
        V + (2 / 3) * np.pi * r**3 <= np.pi * r**2 * L,
        t >= P * r * sf / sigma,
        # a zero share for the caps: the first GP counts them at full thickness
        guess({m: 1, cap: 0}) >= cyl + liner,
        L >= 2 * r,
    ],
)

# the first GP is conservative, so its solution is already a feasible design
conservative = vessel.local_gp().solve(verbosity=0)
print(f"Conservative design: {conservative.cost:.4g~P}")

# each later GP is condensed at the previous solution: the cost never rises
sol = vessel.localsolve(verbosity=0)
print(vessel.table(sol))
print("\nmass of each GP solve, each a feasible design")
print(", ".join(f"{res.cost:.4g~P}" for res in vessel.program.results))
