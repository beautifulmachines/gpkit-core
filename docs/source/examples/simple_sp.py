"""Adapted from t_SP in tests/t_geometric_program.py"""

from gpkit import Model, SignomialsEnabled, Variable

# Decision variables
x = Variable("x")
y = Variable("y")

# must enable signomials for subtraction
with SignomialsEnabled():
    constraints = [x >= 1 - y, y <= 0.1]

# create and solve the SP
m = Model(x, constraints)
sol = m.localsolve(verbosity=0)
print(sol.summary())
assert abs(sol[x] - 0.9) < 1e-6

# full interim solutions are available
print("x values of each GP solve (note convergence)")
print(", ".join(f"{sol.primal[x]:.5f}" for sol in m.program.results))

# use x0 to give the solution, reducing number of GPs needed
m.localsolve(verbosity=0, x0={x: 0.9, y: 0.1})
assert len(m.program.results) == 2
