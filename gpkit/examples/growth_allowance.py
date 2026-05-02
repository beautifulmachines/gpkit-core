"""Mass budget with growth allowances at two levels.

Each Spar declares a 20% allowance on top of its physics-based estimate; the
Wing applies an additional 10% subsystem-level allowance on top of the spars'
totals. The rendered budget table shows CBE + GA = Total at every row, and
the GA column accumulates recursively up the tree.

The shared theta variable (``from gpkit.nomials.growth import theta``)
defaults to 1.0 (auto-substituted), so models that don't care about it
behave exactly as if no scaling were applied. Setting
``model.substitutions[theta().key] = 0.5`` would halve every allowance
simultaneously; freeing theta and adding it to the cost lets the optimizer
choose how much margin to carry.
"""

from gpkit import Model, Variable
from gpkit.budgets import build_budget


class Spar(Model):
    """Spar mass: physics-based estimate plus a 20% growth allowance."""

    m: Variable

    def setup(self):
        rho = Variable("rho", 2700, "kg/m^3", "spar density")
        t = Variable("t", 0.005, "m", "spar wall thickness")
        area = Variable("A", 1.0, "m^2", "spar wetted area")
        self.m = Variable("m", "kg", "spar mass", growth=0.20)
        self.cost = self.m
        return self.m.grown_from(rho * t * area)


class Wing(Model):
    """Wing mass: budgeted across two spars with 10% subsystem allowance."""

    spar1: Spar
    spar2: Spar
    m: Variable

    def setup(self):
        self.spar1 = Spar()
        self.spar2 = Spar()
        self.m = Variable("m", "kg", "wing mass", growth=0.10)
        self.cost = self.m
        return [
            self.m.grown_from(self.spar1.m + self.spar2.m),
            self.spar1,
            self.spar2,
        ]


wing = Wing()
sol = wing.solve(verbosity=0)
print(build_budget(sol, wing, wing.m).text())
