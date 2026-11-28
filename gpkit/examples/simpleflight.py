"Minimizes airplane drag for a simple drag and structure model."

import pickle

from gpkit import Model, Variable, pi

# Constants
k = Variable("k", "-", "form factor", value=1.2)
e = Variable("e", "-", "Oswald efficiency factor", value=0.95)
mu = Variable("\\mu", "kg/m/s", "viscosity of air", value=1.78e-5)
rho = Variable("\\rho", "kg/m^3", "density of air", value=1.23)
tau = Variable("\\tau", "-", "airfoil thickness to chord ratio", value=0.12)
N_ult = Variable("N_{ult}", "-", "ultimate load factor", value=3.8)
V_min = Variable("V_{min}", "m/s", "takeoff speed", value=22)
C_Lmax = Variable("C_{L,max}", "-", "max CL with flaps down", value=1.5)
S_wetratio = Variable("(\\frac{S}{S_{wet}})", "-", "wetted area ratio", value=2.05)
W_W_coeff1 = Variable(
    "W_{W_{coeff1}}", "1/m", "Wing Weight Coefficent 1", value=8.71e-5
)
W_W_coeff2 = Variable("W_{W_{coeff2}}", "Pa", "Wing Weight Coefficent 2", value=45.24)
CDA0 = Variable("(CDA0)", "m^2", "fuselage drag area", value=0.031)
W_0 = Variable("W_0", "N", "aircraft weight excluding wing", value=4940.0)

# Free Variables
D = Variable("D", "N", "total drag force")
A = Variable("A", "-", "aspect ratio")
S = Variable("S", "m^2", "total wing area")
V = Variable("V", "m/s", "cruising speed")
W = Variable("W", "N", "total aircraft weight")
Re = Variable("Re", "-", "Reynold's number")
C_D = Variable("C_D", "-", "Drag coefficient of wing")
C_L = Variable("C_L", "-", "Lift coefficent of wing")
C_f = Variable("C_f", "-", "skin friction coefficient")
W_w = Variable("W_w", "N", "wing weight")

constraints = []

# Drag model
C_D_fuse = CDA0 / S
C_D_wpar = k * C_f * S_wetratio
C_D_ind = C_L**2 / (pi * A * e)
constraints += [C_D >= C_D_fuse + C_D_wpar + C_D_ind]

# Wing weight model
W_w_strc = W_W_coeff1 * (N_ult * A**1.5 * (W_0 * W * S) ** 0.5) / tau
W_w_surf = W_W_coeff2 * S
constraints += [W_w >= W_w_surf + W_w_strc]

# and the rest of the models
constraints += [
    D >= 0.5 * rho * S * C_D * V**2,
    Re <= (rho / mu) * V * (S / A) ** 0.5,
    C_f >= 0.074 / Re**0.2,
    W <= 0.5 * rho * S * C_L * V**2,
    W <= 0.5 * rho * S * C_Lmax * V_min**2,
    W >= W_0 + W_w,
]

print("SINGLE\n======")
m = Model(D, constraints)
sol = m.solve(verbosity=0)
print(m.table(sol))
# save solution to a file and retrieve it
sol.save("solution.pkl")
print("\n" + sol.diff(sol))

print("SWEEP\n=====")
sweeps = {
    V_min: [20.0, 20, 25, 25],
    V: [45.0, 55, 45, 55],
}
sweepsol = m.sweep(sweeps, verbosity=0)
print(m.table(sweepsol))
with open("solution.pkl", "rb") as fil:
    sol_loaded = pickle.load(fil)
print(sweepsol.diff(sol_loaded))
