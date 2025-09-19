"Relaxation examples"

from gpkit import Model, Variable
from gpkit.breakdowns import Breakdowns
from gpkit.constraints.relax import (
    ConstantsRelaxed,
    ConstraintsRelaxed,
    ConstraintsRelaxedEqually,
)

x = Variable("x")
x_min = Variable("x_min", 2)
x_max = Variable("x_max", 1)
m = Model(x, [x <= x_max, x >= x_min])
print("Original model")
print("==============")
print(m)
print("")
# m.solve()  # raises a RuntimeWarning!

print("With constraints relaxed equally")
print("================================")


allrelaxed = ConstraintsRelaxedEqually(m)
mr1 = Model(allrelaxed.relaxvar, allrelaxed)
print(mr1)
print(mr1.solve(verbosity=0).table())  # solves with an x of 1.414

Breakdowns(mr1.solution.to_solution_array()).trace("cost")
print("")

print("With constraints relaxed individually")
print("=====================================")


constraintsrelaxed = ConstraintsRelaxed(m)
mr2 = Model(
    constraintsrelaxed.relaxvars.prod() * m.cost**0.01,
    # add a bit of the original cost in
    constraintsrelaxed,
)
print(mr2)
print(mr2.solve(verbosity=0).table())  # solves with an x of 1.0
print("")

print("With constants relaxed individually")
print("===================================")


constantsrelaxed = ConstantsRelaxed(m)
mr3 = Model(
    constantsrelaxed.relaxvars.prod() * m.cost**0.01,
    # add a bit of the original cost in
    constantsrelaxed,
)
print(mr3)
print(mr3.solve(verbosity=0).table())  # brings x_min down to 1.0
print("")
