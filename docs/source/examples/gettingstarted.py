"The getting started page"

from gpkit import Model, Variable, VectorVariable
from gpkit.nomials import Monomial, Posynomial, PosynomialInequality

# Declare a variable, x
x = Variable("x")

# Declare a variable, y, with units of meters
y = Variable("y", "m")

# Declare a variable, z, with units of meters, and a description
z = Variable("z", "m", "A variable called z with units of meters")

rho = Variable("rho", 1.225, "kg/m^3", "Density of air at sea level")
pi = Variable("pi", 3.14159, "-")

# Declare a 3-element vector variable "x" with units of "m"
x = VectorVariable(3, "x", "m", "Cube corner coordinates")
x_min = VectorVariable(3, "x", [1, 2, 3], "m", "Cube corner minimum")

# create a Monomial term xy^2/z
x = Variable("x")
y = Variable("y")
z = Variable("z")
m = x * y**2 / z
assert isinstance(m, Monomial)

# create a Posynomial expression x + xy^2
x = Variable("x")
y = Variable("y")
p = x + x * y**2
assert isinstance(p, Posynomial)

# consider a block with dimensions x, y, z less than 1
# constrain surface area less than 1.0 m^2
x = Variable("x", "m")
y = Variable("y", "m")
z = Variable("z", "m")
S = Variable("S", 1.0, "m^2")
c = 2 * x * y + 2 * x * z + 2 * y * z <= S
assert isinstance(c, PosynomialInequality)

# Formulate a Model
x = Variable("x")
y = Variable("y")
z = Variable("z")
S = 200
objective = 1 / (x * y * z)
constraints = [2 * x * y + 2 * x * z + 2 * y * z <= S, x >= 2 * y]
m = Model(objective, constraints)

# Solve the Model
sol = m.solve(verbosity=0)

# Printing Results 1
print(sol.table())

# Printing Results 2
print(f"The optimal value is {sol.cost:.4g}.")

# Example variable sensitivity usage
x = Variable("x")
x_min = Variable("x_{min}", 2)
sol = Model(x, [x_min <= x]).solve(verbosity=0)
sens_x_min = sol.sens.variables[x_min]

x = Variable("x")
x_squared_min = Variable("x^2_{min}", 2)
sol = Model(x, [x_squared_min <= x**2]).solve(verbosity=0)
sens_x_min = sol.sens.variables[x_squared_min]
