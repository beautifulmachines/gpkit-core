"Example Loose ConstraintSet usage"

from gpkit import Model, Variable
from gpkit.constraints.loose import Loose

Loose.reltol = 1e-4  # set the global tolerance of Loose
x = Variable("x")
x_min = Variable("x_{min}", 1)
m = Model(x, [Loose([x >= 2], senstol=1e-4), x >= x_min])  # set the specific tolerance
m.solve(verbosity=0)  # prints warning
