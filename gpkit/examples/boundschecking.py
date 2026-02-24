"Verifies that bounds are caught through monomials"

from gpkit import Model, Var
from gpkit.exceptions import UnboundedGP, UnknownInfeasible


class BoundsChecking(Model):
    "Implements a crazy set of unbounded variables."

    Ap = Var("-", "d")
    D = Var("-", "e")
    F = Var("-", "s")
    mi = Var("-", "c")
    mf = Var("-", "r")
    T = Var("-", "i")
    nu = Var("-", "p")
    Fs = Var("-", "t", value=0.9)
    mb = Var("-", "i", value=0.4)
    rf = Var("-", "o", value=0.01)
    V = Var("-", "n", value=300)

    def setup(self):
        self.cost = self.F
        return [
            self.F >= self.D + self.T,
            self.D == self.rf * self.V**2 * self.Ap,
            self.Ap == self.nu,
            self.T == self.mf * self.V,
            self.mf >= self.mi + self.mb,
            self.mf == self.rf * self.V,
            self.Fs <= self.mi,
        ]


m = BoundsChecking()
print(m.str_without(["lineage"]))
try:
    m.solve()
    gp = m.gp()
except UnboundedGP:
    gp = m.gp(checkbounds=False)
missingbounds = gp.check_bounds()

try:
    sol = gp.solve(verbosity=0)  # Errors on mosek_cli
except UnknownInfeasible:  # pragma: no cover
    pass

assert (m.D.key, "lower") in missingbounds
assert (m.nu.key, "lower") in missingbounds
assert (m.Ap.key, "lower") in missingbounds
