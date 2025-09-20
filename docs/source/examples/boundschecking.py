"Verifies that bounds are caught through monomials"

from gpkit import Model, parse_variables
from gpkit.exceptions import UnboundedGP, UnknownInfeasible


# pylint: disable=no-member
class BoundsChecking(Model):
    """Implements a crazy set of unbounded variables.

    Variables
    ---------
    Ap          [-]  d
    D           [-]  e
    F           [-]  s
    mi          [-]  c
    mf          [-]  r
    T           [-]  i
    nu          [-]  p
    Fs    0.9   [-]  t
    mb    0.4   [-]  i
    rf    0.01  [-]  o
    V   300     [-]  n

    Upper Unbounded
    ---------------
    F

    Lower Unbounded
    ---------------
    D

    """

    @parse_variables(__doc__, globals())
    def setup(self):
        self.cost = F
        return [
            F >= D + T,
            D == rf * V**2 * Ap,
            Ap == nu,
            T == mf * V,
            mf >= mi + mb,
            mf == rf * V,
            Fs <= mi,
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
