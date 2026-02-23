"""Modular aircraft concept"""

import numpy as np

from gpkit import Model, Var, Vectorize
from gpkit.interactive.references import referencesplot


class AircraftP(Model):
    """Aircraft flight physics: weight <= lift, fuel burn"""

    Wfuel = Var("lbf", "fuel weight")
    Wburn = Var("lbf", "segment fuel burn")

    def setup(self, aircraft, state):
        self.aircraft = aircraft
        self.state = state

        self.wing_aero = aircraft.wing.perf(state)
        self.perf_models = [self.wing_aero]

        W = aircraft.W
        S = aircraft.wing.S
        V = state.V
        rho = state.rho
        D = self.wing_aero.D
        CL = self.wing_aero.CL

        return (
            self.Wburn >= 0.1 * D,
            W + self.Wfuel <= 0.5 * rho * CL * S * V**2,
            {"performance": self.perf_models},
        )


class Aircraft(Model):
    """The vehicle model"""

    W = Var("lbf", "weight")

    def setup(self):
        self.fuse = Fuselage()
        self.wing = Wing()
        self.components = [self.fuse, self.wing]

        return [self.W >= sum(c.W for c in self.components), self.components]

    def perf(self, state):
        return AircraftP(self, state)


class FlightState(Model):
    """Context for evaluating flight physics"""

    V = Var("knots", "true airspeed", value=40)
    mu = Var("N*s/m^2", "dynamic viscosity", value=1.628e-5)
    rho = Var("kg/m^3", "air density", value=0.74)

    def setup(self):
        pass


class FlightSegment(Model):
    """Combines a context (flight state) and a component (the aircraft)"""

    def setup(self, aircraft):
        self.aircraft = aircraft

        self.flightstate = FlightState()
        self.aircraftp = aircraft.perf(self.flightstate)

        self.Wburn = self.aircraftp.Wburn
        self.Wfuel = self.aircraftp.Wfuel

        return {"aircraft performance": self.aircraftp, "flightstate": self.flightstate}


class Mission(Model):
    """A sequence of flight segments"""

    def setup(self, aircraft):
        self.aircraft = aircraft

        with Vectorize(4):  # four flight segments
            self.fs = FlightSegment(aircraft)

        Wburn = self.fs.aircraftp.Wburn
        Wfuel = self.fs.aircraftp.Wfuel
        self.takeoff_fuel = Wfuel[0]

        return {
            "fuel constraints": [
                Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1],
                Wfuel[-1] >= Wburn[-1],
            ],
            "flight segment": self.fs,
        }


class WingAero(Model):
    """Wing aerodynamics"""

    CD = Var("-", "drag coefficient")
    CL = Var("-", "lift coefficient")
    e = Var("-", "Oswald efficiency", value=0.9)
    Re = Var("-", "Reynold's number")
    D = Var("lbf", "drag force")

    def setup(self, wing, state):
        self.wing = wing
        self.state = state

        c = wing.c
        A = wing.A
        S = wing.S
        rho = state.rho
        V = state.V
        mu = state.mu

        return [
            self.D >= 0.5 * rho * V**2 * self.CD * S,
            self.Re == rho * V * c / mu,
            self.CD >= 0.074 / self.Re**0.2 + self.CL**2 / np.pi / A / self.e,
        ]


class Wing(Model):
    """Aircraft wing model"""

    W = Var("lbf", "weight")
    S = Var("ft^2", "surface area")
    rho = Var("lbf/ft^2", "areal density", value=1)
    A = Var("-", "aspect ratio", value=27)
    c = Var("ft", "mean chord")

    def setup(self):
        return [self.c == (self.S / self.A) ** 0.5, self.W >= self.S * self.rho]

    def perf(self, state):
        return WingAero(self, state)


class Fuselage(Model):
    """The thing that carries the fuel, engine, and payload

    A full model is left as an exercise for the reader.
    """

    W = Var("lbf", "weight", value=100)

    def setup(self):
        pass


AC = Aircraft()
MISSION = Mission(AC)
M = Model(MISSION.takeoff_fuel, [MISSION, AC])
print(M)
sol = M.solve(verbosity=0)
sol.save("solution.pkl")

vars_of_interest = set(AC.varkeys)
# note that there's two ways to access submodels
assert MISSION["flight segment"]["aircraft performance"] is MISSION.fs.aircraftp
vars_of_interest.update(MISSION.fs.aircraftp.unique_varkeys)
vars_of_interest.add(M["D"])
print(sol.summary())
print()
print(sol.table(tables=["slack constraints"], empty="(none)"))

M.append(MISSION.fs.aircraftp.Wburn >= 0.2 * MISSION.fs.aircraftp.wing_aero.D)
sol2 = M.solve(verbosity=0)
print(sol2.diff(sol))

try:
    from gpkit.interactive.sankey import Sankey

    variablesankey = Sankey(sol2, M).diagram(AC.wing.A)
    sankey = Sankey(sol2, M).diagram(width=1200, height=400, maxlinks=30)
    # the line below shows an interactive graph if run in jupyter notebook
    sankey  # pylint: disable=pointless-statement
except (ImportError, ModuleNotFoundError):
    print("Making Sankey diagrams requires the ipysankeywidget package")

referencesplot(M, openimmediately=False)
