"""Blade Element Momentum Theory (BEMT) model for helicopter hover.

Optimizes rotor design for minimum induced power given a fixed vehicle weight.
Discretizes the rotor disk into N radial bins and applies GP-compatible BEMT
constraints relating induced velocity, thrust coefficient, and power
coefficient in each bin.

Demonstrates vectorized GP modeling using Vectorize and Variable inside a
class-based Model with Var descriptors for scalar quantities.

Extracted from docs/source/ipynb/BEMT.ipynb (nbformat-3, legacy notebook).
Ported to current gpkit API.
"""

import numpy as np

from gpkit import Model, Var, Variable, Vectorize


class BEMTHover(Model):
    """Minimum induced power for a hovering rotor via BEMT.

    Parameters
    ----------
    N : int
        Number of radial discretization bins (default 5).
    W_vehicle : float
        Vehicle weight in Newtons (default 1e4 N, roughly 1000 kg).
    """

    # ---- Fixed parameters ----
    rho = Var("kg/m^3", "air density", value=1.23)
    R_max = Var("m", "maximum rotor radius", value=8)
    R_min = Var("m", "minimum rotor radius", value=0.1)
    Omega_max = Var("rpm", "maximum rotor RPM", value=280)
    Omega_min = Var("rpm", "minimum rotor RPM", value=1)
    dr_min = Var("-", "minimum bin width (non-dimensional)", value=1e-3)

    # ---- Rotor-level design variables ----
    A = Var("m^2", "disk area")
    Omega = Var("rpm", "rotor RPM")
    R = Var("m", "rotor radius")
    P = Var("W", "total induced power")

    def setup(self, N=5, W_vehicle=1e4):
        rho = self.rho
        A, Omega, R, P = self.A, self.Omega, self.R, self.P

        # Per-bin variables (N-element vectors created inside Vectorize context)
        with Vectorize(N):
            # Thrust per bin: fixed to equal share of vehicle weight
            xi = Variable(r"\xi", W_vehicle / N, "N", "thrust per bin (fixed)")
            r = Variable("r", "-", "non-dimensional radius")
            dr = Variable(r"\Delta r", "-", "non-dimensional radius step")
            Vi = Variable("V_i", "m/s", "induced velocity")
            dCT = Variable("dC_T", "-", "incremental thrust coefficient")
            dCP = Variable("dC_P", "-", "incremental power coefficient")
            dP = Variable("dP", "W", "incremental power")

        self.cost = P
        return [
            # Rotor geometry
            A == np.pi * R**2,
            R <= self.R_max,
            R >= self.R_min,
            Omega <= self.Omega_max,
            Omega >= self.Omega_min,
            # Radial discretization: bins tile [0, 1]; each bin has positive width
            dr >= self.dr_min,
            r[0] == dr[0] / 2,
            *[r[j] >= dr[:j].sum() for j in range(1, N)],
            *[r[j] >= r[j - 1] + 0.5 * dr[j - 1] + 0.5 * dr[j] for j in range(1, N)],
            r[-1] <= 1,
            1 >= dr.sum(),
            # BEMT: thrust and power coefficient relations
            xi == rho * A * (Omega * R) ** 2 * dCT,
            0.25 == Vi**2 * r * dr / (dCT * (Omega * R) ** 2),
            0.25 == Vi**3 * r * dr / (dCP * (Omega * R) ** 3),
            dP == rho * A * (Omega * R) ** 3 * dCP,
            P >= dP.sum(),
        ]


m = BEMTHover(N=5)
sol = m.solve(verbosity=0)

if __name__ == "__main__":
    print(sol.table())
