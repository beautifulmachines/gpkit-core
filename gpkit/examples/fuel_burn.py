"""Multi-point aircraft fuel burn minimization.

Three-flight-condition GP model for a fixed-wing aircraft that simultaneously
optimizes wing geometry, structural sizing, propulsion, and fuel burn across
cruise, zero-fuel-weight, and sprint conditions.

Demonstrates multi-point analysis through Vectorize: aerodynamics, drag model,
propulsive efficiency, Breguet range equation, and wing structural model are
all expressed as GP constraints and solved in a single flat optimization.

Extracted from docs/source/ipynb/Fuel/Fuel.ipynb (nbformat-4, legacy notebook).
Ported from gpkit.shortcuts (Var/Vec aliases) to current gpkit API.
Interactive widget code stripped; core model retained.
"""

import math

import numpy as np

from gpkit import Model, Var, Variable, Vectorize

# ---- Breguet range approximation order ----
_BREGUET_ORDER = 4


class FuelBurn(Model):
    """Minimize total fuel weight for a fixed-wing aircraft.

    Three flight conditions (indices 0, 1, 2):
      0 — cruise departure (with fuel)
      1 — cruise arrival  (zero-fuel weight)
      2 — sprint          (speed requirement)
    """

    # ---- Constants ----
    N_lift = Var("-", "wing loading multiplier", value=6.0)
    pi = Var("-", "half of the circle constant", value=np.pi)
    sigma_max = Var("Pa", "allowable stress, 6061-T6", value=250e6)
    sigma_maxshear = Var("Pa", "allowable shear stress", value=167e6)
    g = Var("m/s^2", "gravitational constant", value=9.8)
    w = Var("-", "wing-box width/chord", value=0.5)
    r_h = Var("-", "wing strut taper parameter", value=0.75)
    f_wadd = Var("-", "wing added weight fraction", value=2)
    W_fixed = Var("N", "fixed weight", value=14.7e3)
    C_Lmax = Var("-", "maximum C_L, flaps down", value=1.5)
    rho = Var("kg/m^3", "air density, 3000m", value=0.91)
    rho_sl = Var("kg/m^3", "air density, sea level", value=1.23)
    rho_alum = Var("kg/m^3", "density of aluminum", value=2700)
    mu = Var("kg/m/s", "dynamic viscosity, 3000m", value=1.69e-5)
    e = Var("-", "wing spanwise efficiency", value=0.95)
    A_prop = Var("m^2", "propeller disk area", value=0.785)
    eta_eng = Var("-", "engine efficiency", value=0.35)
    eta_v = Var("-", "propeller viscous efficiency", value=0.85)
    h_fuel = Var("J/kg", "fuel heating value", value=42e6)
    V_sprint_reqt = Var("m/s", "sprint speed requirement", value=150)
    W_pay = Var("N", "payload weight", value=500 * 9.81)
    R_min = Var("m", "minimum airplane range", value=5e6)
    V_stallmax = Var("m/s", "stall speed limit", value=40)
    CDA0 = Var("m^2", "fuselage zero-lift drag area", value=0.05)
    # Empirical engine weight fit: W_eng [N] = 0.0372 * P_max^0.8083 [W^0.8083]
    W_eng_coeff = Var("N/W^0.8083", "engine weight fit coefficient", value=0.0372)
    # Unit normalization constants for wing structural model
    # (dimensionless normalized bending moment: divide by 1 N)
    W_ref = Var("N", "structural normalization weight", value=1)
    # (dimensionless spar stiffness: divide by 1 Pa*m^6)
    EI_ref = Var("Pa*m^6", "structural normalization stiffness", value=1)
    # (dimensionless shear: divide by 1 m^-4)
    k_shear = Var("m^-4", "structural normalization shear factor", value=1)

    # ---- Scalar design variables ----
    S = Var("m^2", "wing area")
    R = Var("m", "airplane range")
    A = Var("-", "aspect ratio")
    I_cap = Var("m^4", "spar cap area moment of inertia per unit chord")
    M_rbar = Var("-", "normalized root bending moment")
    P_max = Var("W", "maximum shaft power")
    V_stall = Var("m/s", "stall speed")
    nu = Var("-", "wing taper/shape parameter")
    p = Var("-", "wing taper ratio parameter")
    q = Var("-", "wing taper ratio parameter 2")
    tau = Var("-", "airfoil thickness/chord ratio")
    t_cap = Var("-", "spar cap thickness/chord ratio")
    t_web = Var("-", "shear web thickness/chord ratio")
    W_cap = Var("N", "spar cap weight")
    W_zfw = Var("N", "zero fuel weight")
    W_eng = Var("N", "engine weight")
    W_mto = Var("N", "maximum takeoff weight")
    W_tw = Var("N", "tare weight (fixed + payload + engine)")
    W_web = Var("N", "shear web weight")
    W_wing = Var("N", "wing weight")

    def setup(self):
        # Unpack frequently used constants
        rho, mu = self.rho, self.mu
        rho_sl = self.rho_sl
        rho_alum, g = self.rho_alum, self.g
        e, pi, A_prop = self.e, self.pi, self.A_prop
        eta_eng, eta_v, h_fuel = self.eta_eng, self.eta_v, self.h_fuel
        N_lift, sigma_max = self.N_lift, self.sigma_max
        sigma_maxshear = self.sigma_maxshear
        f_wadd, C_Lmax = self.f_wadd, self.C_Lmax
        w_wb, r_h, W_fixed = self.w, self.r_h, self.W_fixed
        W_pay, R_min = self.W_pay, self.R_min
        V_stallmax, V_sprint_reqt = self.V_stallmax, self.V_sprint_reqt

        # Scalar design variables
        S, R, A = self.S, self.R, self.A
        I_cap, M_rbar, P_max = self.I_cap, self.M_rbar, self.P_max
        V_stall = self.V_stall
        nu, p, q, tau = self.nu, self.p, self.q, self.tau
        t_cap, t_web = self.t_cap, self.t_web
        W_cap, W_zfw, W_eng = self.W_cap, self.W_zfw, self.W_eng
        W_mto, W_tw, W_web, W_wing = self.W_mto, self.W_tw, self.W_web, self.W_wing

        # ---- Per-flight-condition vectors (3 conditions) ----
        with Vectorize(3):
            V = Variable("V", "m/s", "flight speed")
            C_L = Variable("C_L", "-", "wing lift coefficient")
            C_D = Variable("C_D", "-", "wing drag coefficient")
            C_Dp = Variable("C_{D_p}", "-", "profile drag coefficient")
            T = Variable("T", "N", "thrust force")
            Re = Variable("Re", "-", "Reynolds number")
            W = Variable("W", "N", "aircraft weight")
            eta_i = Variable(r"\eta_i", "-", "ideal propulsive efficiency")
            eta_prop = Variable(r"\eta_{prop}", "-", "propeller efficiency")
            eta_0 = Variable(r"\eta_0", "-", "overall propulsive efficiency")

        # ---- Per-cruise-segment vectors (2 segments) ----
        with Vectorize(2):
            W_fuel = Variable("W_fuel", "N", "fuel weight per segment")
            z_bre = Variable("z_bre", "-", "Breguet range parameter")

        # Breguet: 4th-order Taylor approx for e^z - 1 ≈ sum z^k/k!
        z_bre_sum = sum(
            z_bre**k / math.factorial(k) for k in range(1, _BREGUET_ORDER + 1)
        )

        self.cost = W_fuel.sum()
        return [
            # Steady level flight (all 3 conditions)
            W == 0.5 * rho * C_L * S * V**2,
            T >= 0.5 * rho * C_D * S * V**2,
            Re == (rho / mu) * V * (S / A) ** 0.5,
            # Landing / stall constraint
            W_mto <= 0.5 * rho_sl * V_stall**2 * C_Lmax * S,
            V_stall <= V_stallmax,
            # Sprint condition (condition 2)
            P_max >= T[2] * V[2] / eta_0[2],
            V[2] >= V_sprint_reqt,
            # Drag model (Hoburg & Abbeel wing polar fit)
            C_D >= self.CDA0 / S + C_Dp + C_L**2 / (pi * e * A),
            1
            >= (
                2.56 * C_L**5.88 / (Re**1.54 * tau**3.32 * C_Dp**2.62)
                + 3.8e-9 * tau**6.23 / (C_L**0.92 * Re**1.38 * C_Dp**9.57)
                + 2.2e-3 * Re**0.14 * tau**0.033 / (C_L**0.01 * C_Dp**0.73)
                + 1.19e4 * C_L**9.78 * tau**1.76 / (Re * C_Dp**0.91)
                + 6.14e-6 * C_L**6.53 / (Re**0.99 * tau**0.52 * C_Dp**5.19)
            ),
            # Propulsive efficiency chain
            eta_0 <= eta_eng * eta_prop,
            eta_prop <= eta_i * eta_v,
            4 * eta_i + T * eta_i**2 / (0.5 * rho * V**2 * A_prop) <= 4,
            # Breguet range (cruise segments 0 and 1 only)
            R >= R_min,
            z_bre >= g * R * T[:2] / (h_fuel * eta_0[:2] * W[:2]),
            W_fuel / W[:2] >= z_bre_sum,
            # Weight build-up
            W_tw >= W_fixed + W_pay + W_eng,
            W_zfw >= W_tw + W_wing,
            W_eng >= self.W_eng_coeff * P_max**0.8083,
            W_wing / f_wadd >= W_cap + W_web,
            W[0] >= W_zfw + W_fuel[1],
            W[1] >= W_zfw,
            W_mto >= W[0] + W_fuel[0],
            W[2] == W[0],
            # Wing structural model (Hoburg & Abbeel)
            2 * q >= 1 + p,
            p >= 2.2,
            tau <= 0.25,
            M_rbar >= W_tw * A * p / (24 * self.W_ref),
            0.92**2 / 2 * w_wb * tau**2 * t_cap
            >= I_cap * self.k_shear + 0.92 * w_wb * tau * t_cap**2,
            8
            >= N_lift * M_rbar * A * q**2 * tau * self.EI_ref / (S * I_cap * sigma_max),
            12 >= A * W_tw * N_lift * q**2 / (tau * S * t_web * sigma_maxshear),
            nu**3.94 >= 0.86 * p**-2.38 + 0.14 * p**0.56,
            W_cap >= 8 * rho_alum * g * w_wb * t_cap * S**1.5 * nu / (3 * A**0.5),
            W_web >= 8 * rho_alum * g * r_h * tau * t_web * S**1.5 * nu / (3 * A**0.5),
        ]


m = FuelBurn()
sol = m.solve(verbosity=0)

if __name__ == "__main__":
    print(sol.table())
