"Example choice variable usage"

import numpy as np

from gpkit import Model, Variable

x = Variable("x", choices=range(1, 4))
num = Variable("numerator")

m = Model(x + num / x)
sol = m.sweep({num: np.linspace(0.5, 7, 11)}, verbosity=0)

print(sol.table())
