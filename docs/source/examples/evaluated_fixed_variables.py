"Example pre-solve evaluated fixed variable"

from gpkit import Model, Variable, units

# code from t_GPSubs.test_calcconst in tests/t_sub.py
x = Variable("x", "hours")
t_day = Variable("t_{day}", 12, "hours")
t_night = Variable("t_{night}", lambda c: 1 * units.day - c.quantity(t_day), "hours")

# note that t_night has a function as its value
m = Model(x, [x >= t_day, x >= t_night])
sol = m.solve(verbosity=0)
# assert sol["variables"][t_night] == 12
# floating point roundoff errors running with pytest

# call substitutions
m.substitutions.update({t_day: ("sweep", [8, 12, 16])})
sol = m.solve(verbosity=0)
# assert (sol["variables"][t_night] == [16, 12, 8]).all()
# floating point roundoff errors running with pytest
