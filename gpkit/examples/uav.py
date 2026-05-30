"""Modular fixed-wing UAV for out-and-back mission.

Demonstrates gpkit's component + performance-model pattern: each physical
subsystem is a Model subclass with its own variables and an optional perf()
factory that returns a performance model for a given operating condition.
A single flat GP is still solved under the hood.

Reference: W. Hoburg PhD thesis, Chapter 6 (2013)
https://people.eecs.berkeley.edu/~pabbeel/papers/2013_Hoburg-phdthesis.pdf
"""

import math

from gpkit import Model, Var, Variable, pi

_BREGUET_ORDER = 4

# Physical constants shared across components — not properties of any subsystem
g = Variable("g", "m/s^2", "gravitational constant", value=9.8)


class Wing(Model):
    """Wing structure: spar geometry, sizing constraints, and wing weight.

    Receives W_tilde from Aircraft for structural sizing; the spar
    bending and shear constraints depend on the load the wing must carry.
    """

    S = Var("m^2", "wing area")
    A = Var("-", "aspect ratio")
    tau = Var("-", "airfoil thickness/chord ratio")
    I_cap = Var("m^4", "spar cap area moment of inertia per unit chord")
    M_rbar = Var("-", "normalized root bending moment")
    nu = Var("-", "wing taper/shape parameter")
    p = Var("-", "wing taper ratio parameter")
    q = Var("-", "wing taper ratio parameter 2")
    t_cap = Var("-", "spar cap thickness/chord ratio")
    t_web = Var("-", "shear web thickness/chord ratio")
    W_cap = Var("N", "spar cap weight")
    W_web = Var("N", "shear web weight")
    W = Var("N", "wing weight")
    # Wing structural constants
    N_lift = Var("-", "wing loading multiplier", value=6.0)
    sigma_max = Var("MPa", "allowable stress, 6061-T6", value=250)
    sigma_max_shear = Var("MPa", "allowable shear stress", value=167)
    rho_alum = Var("kg/m^3", "density of aluminum", value=2700)
    w = Var("-", "wing-box width/chord", value=0.5)
    r_h = Var("-", "wing strut taper parameter", value=0.75)
    f_wadd = Var("-", "wing added weight fraction", value=2)
    C_Lmax = Var("-", "maximum C_L, flaps down", value=1.5)
    e = Var("-", "wing spanwise efficiency", value=0.95)
    # Unit normalization constants for the structural model
    W_ref = Var("N", "structural normalization weight", value=1)
    EI_ref = Var("Pa*m^6", "structural normalization stiffness", value=1)
    k_shear = Var("m^-4", "structural normalization shear factor", value=1)

    def setup(self, W_tilde):
        S, A, tau = self.S, self.A, self.tau
        I_cap, M_rbar = self.I_cap, self.M_rbar
        nu, p, q = self.nu, self.p, self.q
        t_cap, t_web = self.t_cap, self.t_web
        W_cap, W_web = self.W_cap, self.W_web
        N_lift, sigma_max = self.N_lift, self.sigma_max
        sigma_max_shear, rho_alum = self.sigma_max_shear, self.rho_alum
        w, r_h = self.w, self.r_h
        return {
            "Geometry": [
                2 * q >= 1 + p,
                p >= 1.9,
                tau <= 0.15,
                nu**3.94 >= 0.86 * p**-2.38 + 0.14 * p**0.56,
            ],
            "Root bending stress": [
                M_rbar >= W_tilde * A * p / (24 * self.W_ref),
                (
                    0.92**2 / 2 * w * tau**2 * t_cap
                    >= I_cap * self.k_shear + 0.92 * w * tau * t_cap**2
                ),
                8
                >= N_lift
                * M_rbar
                * A
                * q**2
                * tau
                * self.EI_ref
                / (S * I_cap * sigma_max),
                12 >= A * W_tilde * N_lift * q**2 / (tau * S * t_web * sigma_max_shear),
            ],
            "Weight rollup": [
                W_cap >= 8 * rho_alum * g * w * t_cap * S**1.5 * nu / (3 * A**0.5),
                W_web
                >= 8 * rho_alum * g * r_h * tau * t_web * S**1.5 * nu / (3 * A**0.5),
                self.W / self.f_wadd >= W_cap + W_web,
            ],
        }

    def perf(self, state):
        "Return a WingAero performance model for the given state."
        return WingAero(self, state)


class Engine(Model):
    """Engine: shaft power, weight fit, and propulsion constants."""

    P_max = Var("W", "maximum shaft power")
    W = Var("N", "engine weight")
    eta_eng = Var("-", "engine efficiency", value=0.35)
    eta_v = Var("-", "propeller viscous efficiency", value=0.85)
    A_prop = Var("m^2", "propeller disk area", value=0.785)
    W_eng_coeff = Var("N/W^0.803", "engine weight fit coefficient", value=9.8 * 0.0038)
    h_fuel = Var("MJ/kg", "fuel heating value", value=42)

    def setup(self):
        return [
            self.W >= self.W_eng_coeff * self.P_max**0.803,
        ]

    def perf(self, state, T):
        "Return a PropulsionPerf model for the given state and thrust."
        return PropulsionPerf(self, state, T)


class Aircraft(Model):
    """System-level vehicle: weight buildup and fuselage drag."""

    W_zfw = Var("N", "zero fuel weight")
    W_tilde = Var("N", "non-wing weight (fixed + payload + engine)")
    W_mto = Var("N", "maximum takeoff weight")
    W_fixed = Var("kN", "fixed weight", value=14.7)
    W_pay = Var("N", "payload weight", value=500 * 9.8)
    CDA0 = Var("m^2", "fuselage zero-lift drag area", value=0.05)

    def setup(self):
        self.wing = Wing(self.W_tilde)
        self.engine = Engine()
        return [
            self.wing,
            self.engine,
            self.W_tilde >= self.W_fixed + self.W_pay + self.engine.W,
            self.W_zfw >= self.W_tilde + self.wing.W,
        ]

    def perf(self, state):
        "Return an AircraftPerf model for the given state."
        return AircraftPerf(self, state)


class FlightState(Model):
    """Flight conditions: velocity, air density, viscosity."""

    V = Var("m/s", "flight speed")
    rho = Var("kg/m^3", "air density", value=0.91)
    mu = Var("kg/m/s", "dynamic viscosity", value=1.69e-5)

    def setup(self):
        pass


class WingAero(Model):
    """Wing aerodynamics: profile drag and Reynolds number at a flight state."""

    C_L = Var("-", "wing lift coefficient")
    C_Dp = Var("-", "profile drag coefficient")
    Re = Var("-", "Reynolds number")

    def setup(self, wing, state):
        C_L, C_Dp, Re = self.C_L, self.C_Dp, self.Re
        S, A, tau = wing.S, wing.A, wing.tau
        V, rho, mu = state.V, state.rho, state.mu
        return [
            Re == (rho / mu) * V * (S / A) ** 0.5,
            # Hoburg & Abbeel wing polar fit
            1
            >= (
                2.556 * C_L**5.881 / (Re**1.541 * tau**3.319 * C_Dp**2.617)
                + 3.823e-9 * tau**6.229 / (C_L**0.914 * Re**1.379 * C_Dp**9.57)
                + 2.151e-3 * Re**0.1441 * tau**0.03325 / (C_L**0.0079 * C_Dp**0.7337)
                + 1.19e4 * C_L**9.783 * tau**1.764 / (Re**0.998 * C_Dp**0.909)
                + 6.143e-6 * C_L**6.535 / (Re**0.995 * tau**0.521 * C_Dp**5.192)
            ),
        ]


class PropulsionPerf(Model):
    """Propulsive efficiency chain at a flight state.

    Receives T (thrust) from AircraftPerf for the actuator disk equation.
    """

    eta_i = Var("-", "ideal propulsive efficiency")
    eta_prop = Var("-", "propeller efficiency")
    eta_0 = Var("-", "overall propulsive efficiency")

    def setup(self, engine, state, T):
        eta_i, eta_prop, eta_0 = self.eta_i, self.eta_prop, self.eta_0
        V, rho = state.V, state.rho
        return [
            eta_0 <= engine.eta_eng * eta_prop,
            eta_prop <= eta_i * engine.eta_v,
            4 * eta_i + T * eta_i**2 / (0.5 * rho * V**2 * engine.A_prop) <= 4,
        ]


class AircraftPerf(Model):
    """Full aircraft performance: lift balance, total drag, thrust.

    Includes the flight state so that V, rho, mu appear in this model's
    unique_varkeys — useful for condition table comparisons.
    """

    T = Var("N", "thrust force")
    W = Var("N", "aircraft weight")
    C_D = Var("-", "total drag coefficient")
    C_Di = Var("-", "induced drag coefficient")

    def setup(self, aircraft, state):
        self.wing_aero = aircraft.wing.perf(state)
        self.prop_perf = aircraft.engine.perf(state, self.T)
        C_L = self.wing_aero.C_L
        C_Dp = self.wing_aero.C_Dp
        S = aircraft.wing.S
        A = aircraft.wing.A
        e = aircraft.wing.e
        rho, V = state.rho, state.V
        return [
            state,
            self.wing_aero,
            self.prop_perf,
            self.W == 0.5 * rho * C_L * S * V**2,
            self.T >= 0.5 * rho * self.C_D * S * V**2,
            self.C_Di >= C_L**2 / (pi * e * A),
            self.C_D >= aircraft.CDA0 / S + C_Dp + self.C_Di,
        ]


class MissionLeg(Model):
    """One cruise leg: Breguet range equation at a single flight state.

    W is the *final* (end-of-leg) weight, following the standard Breguet
    formulation where W_fuel/W = e^z - 1 ≈ z + z²/2 + …
    """

    W_fuel = Var("N", "fuel weight for this leg")
    z_bre = Var("-", "Breguet range parameter")

    def setup(self, aircraft, R):
        # R is Mission.R — the same VarKey shared by outbound and return legs
        self.state = FlightState()
        self.perf = aircraft.perf(self.state)
        T, W = self.perf.T, self.perf.W
        eta_0 = self.perf.prop_perf.eta_0
        z = self.z_bre
        z_sum = sum(z**k / math.factorial(k) for k in range(1, _BREGUET_ORDER + 1))
        return [
            self.perf,  # AircraftPerf already includes self.state
            z >= g * R * T / (aircraft.engine.h_fuel * eta_0 * W),
            self.W_fuel / W >= z_sum,
        ]


class Outbound(MissionLeg):
    """Outbound cruise leg."""


class Return(MissionLeg):
    """Return cruise leg."""


class Mission(Model):
    """Out-and-back mission: two cruise legs, sprint condition, weight chain.

    outbound_leg.perf.W is the turnaround weight — the final weight of the
    outbound leg and initial weight of the return leg.  The sprint condition
    is sized at the turnaround weight.

    For a multi-segment cruise, replace the two named legs with:
        with Vectorize(N): self.segments = MissionSegment(aircraft, self.R)
    """

    R = Var("m", "mission leg range")
    R_min = Var("km", "minimum range requirement", value=5e3)

    def setup(self, aircraft):
        self.outbound_leg = Outbound(aircraft, self.R)
        self.return_leg = Return(aircraft, self.R)
        self.sprint = SprintCondition(aircraft)
        W_out = self.outbound_leg.perf.W  # turnaround weight (end of outbound)
        W_ret = self.return_leg.perf.W  # landing weight (end of return)
        return [
            self.outbound_leg,
            self.return_leg,
            self.sprint,
            self.R >= self.R_min,
            W_out >= aircraft.W_zfw + self.return_leg.W_fuel,
            W_ret >= aircraft.W_zfw,
            aircraft.W_mto >= W_out + self.outbound_leg.W_fuel,
            self.sprint.perf.W == W_out,
        ]


class SprintCondition(Model):
    """Sprint condition: engine sizing at the required sprint speed."""

    V_reqt = Var("m/s", "sprint speed requirement", value=150)

    def setup(self, aircraft):
        self.state = FlightState()
        self.perf = aircraft.perf(self.state)
        return [
            self.perf,  # AircraftPerf already includes self.state
            aircraft.engine.P_max
            >= self.perf.T * self.state.V / self.perf.prop_perf.eta_0,
            self.state.V >= self.V_reqt,
        ]


class LandingCondition(Model):
    """Landing: stall speed constraint at MTOW and sea-level density."""

    V_stall = Var("m/s", "stall speed")
    V_stallmax = Var("m/s", "stall speed limit", value=38)
    rho_sl = Var("kg/m^3", "air density, sea level", value=1.23)

    def setup(self, aircraft):
        return [
            (
                aircraft.W_mto
                <= 0.5
                * self.rho_sl
                * self.V_stall**2
                * aircraft.wing.C_Lmax
                * aircraft.wing.S
            ),
            self.V_stall <= self.V_stallmax,
        ]


class UAV(Model):
    """Fixed-wing UAV: minimize total fuel for an out-and-back mission.

    Submodels are accessed via attributes (e.g. self.aircraft.wing,
    self.mission.outbound_leg) rather than string-key indexing.  String-key
    access via model["name"] remains available for programmatic traversal.
    """

    def setup(self):
        self.aircraft = Aircraft()
        self.mission = Mission(self.aircraft)
        self.landing = LandingCondition(self.aircraft)
        self.cost = self.mission.outbound_leg.W_fuel + self.mission.return_leg.W_fuel
        return [
            self.aircraft,
            self.mission,
            self.landing,
        ]


if __name__ == "__main__":
    m = UAV()
    sol = m.solve(verbosity=0)
    print(sol.table())
    print()
    print(
        sol.table(
            tables=["condition_table"],
            condition_table={
                "Outbound": m.mission.outbound_leg,
                "Return": m.mission.return_leg,
                "Sprint": m.mission.sprint,
            },
        )
    )
