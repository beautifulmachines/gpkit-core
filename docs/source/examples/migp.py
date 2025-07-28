"Example choice variable usage"

import numpy as np

from gpkit import Model, Variable

x = Variable("x", choices=range(1, 4))
num = Variable("numerator")

m = Model(x + num / x)
sol = m.solve(verbosity=0, sweep={num: np.linspace(0.5, 7, 11)})

print(sol.table())
