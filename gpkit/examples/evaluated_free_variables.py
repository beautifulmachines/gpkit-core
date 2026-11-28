"Example post-solve computed variable"

from gpkit import Model, Variable

x = Variable("x")
x2 = Variable("x^2")
m = Model(x, [x >= 2])
m.computed[x2] = lambda sol: sol[x] ** 2
sol = m.solve(verbosity=0)
assert abs(sol[x2] - 4) <= 1e-4
