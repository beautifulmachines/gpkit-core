"""Pumped fluid pipeline: one pipe sized by flow, pressure and bending.

A classic sizing tradeoff outside aerospace. A wider bore costs more steel but
less pumping energy; the wall is set by internal pressure plus the bending of a
fluid-filled span between supports.

Demonstrates:
  * one physical component exercised at several operating conditions, via
    Vectorize -- variables built inside the block are per-condition, those
    outside are shared by every condition
  * duty weighting, so conditions carry their share of lifetime energy rather
    than counting equally
  * two ways to treat a quantity that varies across conditions: share one
    variable when the system really has one value (``p_sys``), or take an
    envelope when the peak sizes hardware (``Pump.P_rated``)
  * cost as a rollup: each component owns its own contribution, so ``LCC``
    breaks down by component rather than arriving as one opaque expression
"""

from typing import ClassVar

from numpy import pi

from gpkit import Model, Var, units
from gpkit.util.globals import Vectorize

g = 9.81 * units("m/s^2")


class Fluid(Model):
    """The pumped substance. Defaults describe water at about 20 C."""

    assumptions: ClassVar = [
        "incompressible, Newtonian, constant properties along the run",
    ]

    rho = Var("kg/m^3", "density", value=1000)
    mu = Var("Pa*s", "dynamic viscosity", value=1.0e-3)


class Pipe(Model):
    """Bore, wall thickness and mass of the pipe itself."""

    assumptions: ClassVar = [
        "thin wall: t << D, so mass and section properties use the bore",
        "constant section along the whole run",
    ]

    D = Var("m", "inner diameter")
    t = Var("m", "wall thickness")
    A = Var("m^2", "flow area")
    W = Var("kg", "mass")

    C = Var("-", "installed cost")

    rho_m = Var("kg/m^3", "wall material density", value=7800)
    sigma = Var("MPa", "allowable stress", value=200)
    t_min = Var("mm", "minimum gauge", value=2)
    c_mat = Var("1/kg", "installed material cost", value=6.0)

    def setup(self, L):
        self.L = L
        return {
            "Geometry": [
                self.A == pi / 4 * self.D**2,
                self.t >= self.t_min,
            ],
            "Mass": [
                self.W >= self.rho_m * pi * self.D * self.t * L,
            ],
            "Cost": [
                self.C >= self.c_mat * self.W,
            ],
        }


class Pump(Model):
    """The pump: rating, efficiency and capital cost."""

    assumptions: ClassVar = [
        (
            "one efficiency across every condition; a real pump peaks at its"
            " best-efficiency point and falls off either side"
        ),
    ]

    P_rated = Var("kW", "rating, set by the peak condition")
    C = Var("-", "capital cost")

    eta = Var("-", "efficiency", value=0.7)
    c_unit = Var("1/kW", "capital cost per rated kW", value=900.0)

    def setup(self):
        return {
            "Cost": [
                self.C >= self.c_unit * self.P_rated,
            ],
        }


class PipeFlow(Model):
    """Pressure drop and pumping power at one flow condition."""

    assumptions: ClassVar = [
        "steady, fully developed turbulent flow in a hydraulically smooth pipe",
        (
            "friction factor from the power-law fit f = 0.184 Re^-0.2,"
            " valid for roughly 2e4 < Re < 1e6"
        ),
        "minor losses from bends and fittings are not modelled",
    ]

    Q = Var("m^3/s", "flow demand")
    V = Var("m/s", "bulk velocity")
    Re = Var("-", "Reynolds number")
    f = Var("-", "Darcy friction factor")
    dp = Var("Pa", "pressure drop")
    P = Var("W", "pumping power")

    hours = Var("hr", "lifetime operating hours at this condition")

    def setup(self, pipe, pump, fluid):
        rho, mu = fluid.rho, fluid.mu
        return {
            "Continuity": [
                self.Q <= self.V * pipe.A,
            ],
            "Friction": [
                self.Re <= rho * self.V * pipe.D / mu,
                self.f >= 0.184 * self.Re**-0.2,
            ],
            "Pressure drop": [
                self.dp >= self.f * (pipe.L / pipe.D) * rho * self.V**2 / 2,
                self.P >= self.dp * self.Q / pump.eta,
            ],
        }


class SpanLoading(Model):
    """Bending of a fluid-filled pipe between supports."""

    assumptions: ClassVar = [
        "simply supported at each support, uniformly loaded: M = w*L^2/8",
        "the pipe runs full, so the fluid weight is carried by the wall",
    ]

    M = Var("N*m", "midspan bending moment")
    L_span = Var("m", "support spacing", value=12)

    def setup(self, pipe, fluid):
        w_pipe = pipe.rho_m * pi * pipe.D * pipe.t * g
        w_fluid = fluid.rho * pipe.A * g
        return [
            self.M >= (w_pipe + w_fluid) * self.L_span**2 / 8,
        ]


class Pipeline(Model):
    """Size one pipe and its pump for a duty cycle of flow conditions.

    Minimises installed material cost plus pump capital and lifetime energy.
    The bore trades steel against pumping energy; the wall carries hoop stress
    from the pressure rating together with midspan bending.
    """

    assumptions: ClassVar = [
        "hoop and bending stress are added directly, without an interaction rule",
        (
            "costs are undiscounted lifetime totals; installation labour,"
            " maintenance and decommissioning are excluded"
        ),
    ]
    references: ClassVar = [
        (
            "F. M. White, Fluid Mechanics, 7th ed., ch. 6"
            " (Darcy-Weisbach and smooth-pipe friction correlations)"
        ),
    ]

    L = Var("m", "pipe run length", value=1000)
    p_sys = Var("bar", "system pressure rating", value=16)
    LCC = Var("-", "life-cycle cost: pipe + pump + lifetime energy")
    C_nrg = Var("-", "lifetime energy cost")

    c_nrg = Var("1/(kW*hr)", "energy price", value=0.10)

    def setup(self, n_conditions=3):
        self.fluid = Fluid()
        self.pipe = Pipe(self.L)
        self.pump = Pump()
        with Vectorize(n_conditions):
            self.flow = PipeFlow(self.pipe, self.pump, self.fluid)
        self.span = SpanLoading(self.pipe, self.fluid)

        pipe, pump, flow, span = self.pipe, self.pump, self.flow, self.span
        self.cost = self.LCC
        sigma_hoop = self.p_sys * pipe.D / (2 * pipe.t)
        sigma_bend = 4 * span.M / (pi * pipe.D**2 * pipe.t)

        return [
            self.fluid,
            pipe,
            pump,
            flow,
            span,
            pipe.sigma >= sigma_hoop + sigma_bend,
            # the peak condition sizes the pump; every condition burns energy
            pump.P_rated >= flow.P,
            self.C_nrg >= self.c_nrg * (flow.P * flow.hours).sum(),
            self.LCC >= pipe.C + pump.C + self.C_nrg,
        ]

    @classmethod
    def default(cls):
        """Three conditions: a long-running nominal, a mid, and a brief peak."""
        m = cls()
        m.substitutions[m.flow.Q] = [0.10, 0.18, 0.30]
        m.substitutions[m.flow.hours] = [70000, 25000, 5000]
        return m


if __name__ == "__main__":
    m = Pipeline.default()
    sol = m.solve(verbosity=0)
    print(sol.table())
