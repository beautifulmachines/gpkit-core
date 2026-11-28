"A simple primal infeasible example"

from gpkit import Model, Variable

x = Variable("x")
y = Variable("y")

m = Model(x * y, [x >= 1, y >= 2, x * y >= 0.5, x * y <= 1.5])

# raises UnknownInfeasible on cvxopt, PrimalInfeasible on mosek
# m.solve()
