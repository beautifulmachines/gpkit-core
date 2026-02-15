"Show autosweep_1d functionality"

import pickle

import numpy as np

import gpkit
from gpkit import Model, Variable, units
from gpkit.tools.autosweep import autosweep_1d
from gpkit.util.small_scripts import mag

A = Variable("A", "m**2")
w = Variable("w", "m")

m1 = Model(A**2, [A >= w**2 + units.m**2])
tol1 = 1e-3
bst1 = autosweep_1d(m1, tol1, w, [1, 10], verbosity=0)
# pylint: disable=no-member
print(f"Solved after {bst1.nsols} passes, cost logtol +/-{bst1.tol:.3g}")
# autosweep solution accessing
w_vals = np.linspace(1, 10, 10)
sol1 = bst1.sample_at(w_vals)
print(f"values of w: {w_vals}")
a_els = " ".join(f" {n:.1f}" for n in sol1("A").magnitude)
print(f"values of A: [{a_els}] {sol1('A').units}")
cost_estimate = sol1["cost"]
cost_lb, cost_ub = sol1.cost_lb(), sol1.cost_ub()
print(f"cost lower bound:\n{cost_lb}\n")
print(f"cost estimate:\n{cost_estimate}\n")
print(f"cost upper bound:\n{cost_ub}\n")
# you can evaluate arbitrary posynomials
np.testing.assert_allclose(mag(2 * sol1(A)), mag(sol1(2 * A)))
assert (sol1["cost"] == sol1(A**2)).all()
# the cost estimate is the logspace mean of its upper and lower bounds
np.testing.assert_allclose(
    (np.log(mag(cost_lb)) + np.log(mag(cost_ub))) / 2, np.log(mag(cost_estimate))
)
# save autosweep to a file and retrieve it
bst1.save("autosweep.pkl")
with open("autosweep.pkl", "rb") as fil:
    bst1_loaded = pickle.load(fil)

# this problem is two intersecting lines in logspace
m2 = Model(A**2, [A >= (w / 3) ** 2, A >= (w / 3) ** 0.5 * units.m**1.5])
tol2 = {"mosek_cli": 1e-6, "mosek_conif": 1e-6, "cvxopt": 1e-7}[
    gpkit.settings["default_solver"]
]
# test Model method
sol2 = m2.autosweep({w: [1, 10]}, tol2, verbosity=0)
bst2 = sol2.bst
print(f"Solved after {bst2.nsols} passes, cost logtol +/-{bst2.tol:.3g}")
print("Table of solutions used in the autosweep:\n")
print(bst2.solarray.table())
