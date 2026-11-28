"example of an SP that turns into a GP"

import numpy as np

from gpkit import Model, SignomialsEnabled, Variable, units
from gpkit.constraints.tight import Tight


def SimPleAC():
    "Creates SimpleAC model"
    # Env. constants
    g = Variable("g", "m/s^2", "gravitational acceleration", value=9.81)
    mu = Variable("\\mu", "kg/m/s", "viscosity of air", value=1.775e-5)
    rho = Variable("\\rho", "kg/m^3", "density of air", value=1.23)
    rho_f = Variable("\\rho_f", "kg/m^3", "density of fuel", value=817)

    # Non-dimensional constants
    C_Lmax = Variable("C_{L,max}", "-", "max CL with flaps down", value=1.6)
    e = Variable("e", "-", "Oswald efficiency factor", value=0.92)
    k = Variable("k", "-", "form factor", value=1.17)
    N_ult = Variable("N_{ult}", "-", "ultimate load factor", value=3.3)
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", "-", "wetted area ratio", value=2.075)
    tau = Variable("\\tau", "-", "airfoil thickness to chord ratio", value=0.12)
    W_W_coeff1 = Variable(
        "W_{W_{coeff1}}", "1/m", "wing weight coefficent 1", value=2e-5
    )  # 12e-5 originally
    W_W_coeff2 = Variable("W_{W_{coeff2}}", "Pa", "wing weight coefficent 2", value=60)

    # Dimensional constants
    Range = Variable("Range", "km", "aircraft range", value=3000)
    TSFC = Variable("TSFC", "1/hr", "thrust specific fuel consumption", value=0.6)
    V_min = Variable("V_{min}", "m/s", "takeoff speed", value=25)
    W_0 = Variable("W_0", "N", "aircraft weight excluding wing", value=6250)

    # Free Variables
    LoD = Variable("L/D", "-", "lift-to-drag ratio")
    D = Variable("D", "N", "total drag force")
    V = Variable("V", "m/s", "cruising speed")
    W = Variable("W", "N", "total aircraft weight")
    Re = Variable("Re", "-", "Reynold's number")
    CDA0 = Variable("(CDA0)", "m^2", "fuselage drag area")  # 0.035 originally
    C_D = Variable("C_D", "-", "drag coefficient")
    C_L = Variable("C_L", "-", "lift coefficient of wing")
    C_f = Variable("C_f", "-", "skin friction coefficient")
    W_f = Variable("W_f", "N", "fuel weight")
    V_f = Variable("V_f", "m^3", "fuel volume")
    V_f_avail = Variable("V_{f_{avail}}", "m^3", "fuel volume available")
    T_flight = Variable("T_{flight}", "hr", "flight time")

    # Free variables (fixed for performance eval.)
    A = Variable("A", "-", "aspect ratio")
    S = Variable("S", "m^2", "total wing area")
    W_w = Variable("W_w", "N", "wing weight")
    W_w_strc = Variable("W_w_strc", "N", "wing structural weight")
    W_w_surf = Variable("W_w_surf", "N", "wing skin weight")
    V_f_wing = Variable("V_f_wing", "m^3", "fuel volume in the wing")
    V_f_fuse = Variable("V_f_fuse", "m^3", "fuel volume in the fuselage")

    objective = W_f

    constraints = []

    # Weight and lift model
    constraints += [
        W >= W_0 + W_w + W_f,
        W_0 + W_w + 0.5 * W_f <= 0.5 * rho * S * C_L * V**2,
        W <= 0.5 * rho * S * C_Lmax * V_min**2,
        T_flight >= Range / V,
        LoD == C_L / C_D,
    ]

    # Thrust and drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L**2 / (np.pi * A * e)
    constraints += [
        W_f >= TSFC * T_flight * D,
        D >= 0.5 * rho * S * C_D * V**2,
        C_D >= C_D_fuse + C_D_wpar + C_D_ind,
        V_f_fuse <= 10 * units("m") * CDA0,
        Re <= (rho / mu) * V * (S / A) ** 0.5,
        C_f >= 0.074 / Re**0.2,
    ]

    # Fuel volume model
    with SignomialsEnabled():
        constraints += [
            V_f == W_f / g / rho_f,
            # linear with b and tau, quadratic with chord
            V_f_wing**2 <= 0.0009 * S**3 / A * tau**2,
            V_f_avail <= V_f_wing + V_f_fuse,  # [SP]
            Tight([V_f_avail >= V_f]),
        ]

    # Wing weight model
    constraints += [
        W_w_surf >= W_W_coeff2 * S,
        W_w_strc**2
        >= W_W_coeff1**2
        / tau**2
        * N_ult**2
        * A**3
        * (V_f_fuse * g * rho_f + W_0)
        * W
        * S,
        W_w >= W_w_surf + W_w_strc,
    ]

    m = Model(objective, constraints)

    return m


sa = SimPleAC()
sa.substitutions.update({"V_f_fuse": 0.5})
sol = sa.sweep({"V_f_wing": np.linspace(0.1, 0.5, 3)}, verbosity=0)
print(sa.table(sol))
