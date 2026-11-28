"""Composite pressure vessel: a signomial constraint with a guessed first split"""

from gpkit import Model, Variable, guess, pi

r = Variable("r", "m", "radius")
L = Variable("L", "m", "overall length, end caps included")
t = Variable("t", "m", "cylinder wall thickness")
m = Variable("m", "kg", "mass")
P = Variable("P", "Pa", "pressure", value=30e6)
V = Variable("V", "m^3", "internal volume", value=0.05)
sf = Variable("sf", "-", "safety factor", value=2.25)
sigma = Variable("sigma", "Pa", "composite strength", value=1.5e9)
rho_c = Variable("rho_c", "kg/m^3", "composite density", value=1550)
rho_l = Variable("rho_l", "kg/m^3", "liner density", value=2700)
t_l = Variable("t_l", "m", "liner thickness", value=1e-3)

# the hemispherical caps carry half the cylinder's hoop stress, so they are
# half as thick, and their mass is subtracted from the cylinder's
cap = 2 * pi * rho_c * r**2 * t
cyl = 2 * pi * rho_c * r * t * L
liner = 2 * pi * rho_l * r * L * t_l

vessel = Model(
    m,
    [
        V + (2 / 3) * pi * r**3 <= pi * r**2 * L,
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
