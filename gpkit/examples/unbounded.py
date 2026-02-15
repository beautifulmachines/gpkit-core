"Demonstrate a trivial unbounded variable"

from gpkit import Model, Variable
from gpkit.constraints.bounded import Bounded

x = Variable("x")

constraints = [x >= 1]

m = Model(1 / x, constraints)  # MOSEK returns DUAL_INFEAS_CER on .solve()
m = Model(1 / x, Bounded(constraints))
# by default, prints bounds warning during solve
sol = m.solve(verbosity=0)
print(sol.summary())
# but they can also be accessed from the solution:
assert (
    sol.meta["boundedness"]["value near upper bound of 1e+30"]
    == sol.meta["boundedness"]["sensitive to upper bound of 1e+30"]
)
