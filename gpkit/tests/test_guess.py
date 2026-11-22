"""Tests for guessed term shares on signomial constraints"""

from itertools import pairwise

import numpy as np
import pytest

from gpkit import Model, SignomialEquality, SignomialsEnabled, Variable, guess, pi
from gpkit.constraints.set import keyed_constraints
from gpkit.exceptions import Infeasible, InvalidGPConstraint
from gpkit.nomials import SignomialInequality


def assert_same_monomial(mon, expected):
    "Same exponents, and coefficients equal to within floating point"
    assert mon.exp == expected.exp
    assert mon.c == pytest.approx(expected.c)


def kg(mass):
    "A mass quantity, as a number of kilograms"
    return mass.to("kg").magnitude


class TestLowerBound:
    def test_guessed_shares(self):
        x, y = Variable("x"), Variable("y")
        bound = guess({x: 1, 2 * y: 3}).lower_bound()
        assert_same_monomial(bound, (x / 0.25) ** 0.25 * (2 * y / 0.75) ** 0.75)

    def test_shares_at_a_point(self):
        "At a point where the terms split 1:3, the bound is the same"
        x, y = Variable("x"), Variable("y")
        assert_same_monomial(
            (x + 2 * y).shares_at({x: 1, y: 1.5}).lower_bound(),
            guess({x: 1, 2 * y: 3}).lower_bound(),
        )

    def test_shares_at_partial_point(self):
        "Shares are numbers whenever the substituted terms share an exponent"
        x, y = Variable("x"), Variable("y")
        p = x + 2 * x * y
        assert_same_monomial(
            p.shares_at({y: 1.5}).lower_bound(),
            guess({x: 1, 2 * x * y: 3}).lower_bound(),
        )

    def test_shares_at_is_none_unless_terms_collapse(self):
        x, y = Variable("x"), Variable("y")
        assert (x + 2 * y).shares_at({x: 1}) is None
        with pytest.raises(ValueError, match="monomial"):
            (x + 2 * y).mono_lower_bound({x: 1})

    def test_tiny_share_stays_finite(self):
        "(c/w)**w overflows for a tiny w unless computed in log space"
        x, y = Variable("x"), Variable("y")
        bound = guess({x: 1, 1e300 * y: 1e-310}).lower_bound()
        assert np.isfinite(bound.c)

    def test_mono_lower_bound_is_bound_at_shares(self):
        x, y = Variable("x"), Variable("y")
        p = x + 2 * y
        assert_same_monomial(
            p.mono_lower_bound({x: 1, y: 1.5}),
            p.shares_at({x: 1, y: 1.5}).lower_bound(),
        )

    def test_zero_share_drops_term(self):
        x, y = Variable("x"), Variable("y")
        assert_same_monomial(guess({x: 1, 2 * y: 0}).lower_bound(), x)


class TestGuess:
    def test_builds_signomial_inequality(self):
        "Writing a guess is the opt-in; no SignomialsEnabled needed"
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        assert isinstance(guess({x: 1, y: 1}) >= z, SignomialInequality)
        assert isinstance(z <= guess({x: 1, y: 1}), SignomialInequality)

    def test_weights_normalize(self):
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        gpc = (guess({x: 2, y: 2}) >= z).as_gpconstr({})
        assert_same_monomial(gpc.right, (x / 0.5) ** 0.5 * (y / 0.5) ** 0.5)

    def test_terms_display_as_written(self):
        x, y = Variable("x"), Variable("y")
        assert str(guess({x: 1, pi * y: 0}).posynomial) == "x + π⋅y"

    def test_arithmetic_raises(self):
        x, y = Variable("x"), Variable("y")
        with pytest.raises(TypeError):
            guess({x: 1, y: 1}) + x

    @pytest.mark.parametrize("weights", [(1, -1), (0, 0)])
    def test_bad_weights_raise(self, weights):
        x, y = Variable("x"), Variable("y")
        with pytest.raises(ValueError, match="weight"):
            guess(dict(zip([x, y], weights)))

    def test_term_on_both_sides_raises(self):
        "x + y >= 2y + z is really x >= y + z"
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        with pytest.raises(ValueError, match="both sides"):
            _ = guess({x: 1, y: 1}) >= 2 * y + z

    def test_keys_combining_into_one_term_raise(self):
        x = Variable("x")
        with pytest.raises(ValueError, match="term"):
            guess({x: 1, 2 * x: 1})

    def test_non_monomial_key_raises(self):
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        with pytest.raises(ValueError, match="must be a monomial"):
            guess({x + y: 1, z: 1})

    def test_equality_takes_no_guess(self):
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        with pytest.raises(TypeError, match="inequality"):
            SignomialEquality(z, guess({x: 1, y: 1}))


def pressure_vessel(*, guessed=True):
    """Cylindrical composite vessel with hemispherical end caps and a liner.

    The caps carry half the cylinder's hoop stress, so they are half as thick.
    Guessing a zero share for the cap term counts them at full thickness.
    """
    r = Variable("r", "m", "radius")
    L = Variable("L", "m", "overall length, caps included")
    t = Variable("t", "m", "cylinder wall thickness")
    m = Variable("m", "kg", "mass")
    P = Variable("P", 30e6, "Pa", "pressure")
    V = Variable("V", 0.05, "m^3", "internal volume")
    sf = Variable("sf", 2.25, "-", "safety factor")
    sigma = Variable("sigma", 1.5e9, "Pa", "composite strength")
    rho_c = Variable("rho_c", 1550, "kg/m^3", "composite density")
    rho_l = Variable("rho_l", 2700, "kg/m^3", "liner density")
    t_l = Variable("t_l", 1e-3, "m", "liner thickness")

    cap = 2 * np.pi * rho_c * r**2 * t
    cyl = 2 * np.pi * rho_c * r * t * L
    liner = 2 * np.pi * rho_l * r * L * t_l
    if guessed:
        mass = guess({m: 1, cap: 0}) >= cyl + liner
    else:
        mass = m >= cyl + liner  # caps at full thickness: the conservative GP
    constraints = [
        V + (2 / 3) * np.pi * r**3 <= np.pi * r**2 * L,
        t >= P * r * sf / sigma,
        mass,
        L >= 2 * r,
    ]
    return Model(m, constraints), mass, {"r": r, "L": L, "t": t, "m": m}


CONSERVATIVE_KG = 9.88180  # caps counted at full thickness
CONVERGED_KG = 7.00338  # a sphere: L = 2r


class TestPressureVessel:
    def test_gp_is_conservative(self, solver):
        m, mass, _ = pressure_vessel()
        sol = m.local_gp().solve(solver=solver, verbosity=0)
        assert kg(sol.cost) == pytest.approx(CONSERVATIVE_KG, rel=1e-4)
        ref, _, _ = pressure_vessel(guessed=False)
        refsol = ref.solve(solver=solver, verbosity=0)
        assert kg(sol.cost) == pytest.approx(kg(refsol.cost), rel=1e-6)
        (key,) = [k for k, c in keyed_constraints(m) if c is mass]
        assert key in sol.sens.constraints_by_key

    def test_solve_still_raises(self, solver):
        "The conservative GP is a different problem; solve() doesn't stand in"
        m, _, _ = pressure_vessel()
        with pytest.raises(InvalidGPConstraint):
            m.solve(solver=solver, verbosity=0)

    def test_gp_still_raises(self):
        "gp() is for models that are GPs; local_gp() is the approximation"
        m, _, _ = pressure_vessel()
        with pytest.raises(InvalidGPConstraint):
            m.gp()

    def test_localsolve_refines_without_slack(self, solver):
        m, _, v = pressure_vessel()
        sol = m.localsolve(solver=solver, verbosity=0)
        assert kg(sol.cost) == pytest.approx(CONVERGED_KG, rel=1e-4)
        assert float(sol[v["L"]] / sol[v["r"]]) == pytest.approx(2, rel=1e-4)
        costs = [kg(res.cost) for res in m.program.results]
        assert costs[0] == pytest.approx(CONSERVATIVE_KG, rel=1e-4)
        assert all(b <= a * (1 + 1e-6) for a, b in pairwise(costs))

    def test_x0_sets_first_shares(self, solver):
        "Values for m, r and t collapse m + cap to a monomial, so x0 sets its shares"
        m, _, v = pressure_vessel()
        conservative = m.local_gp().solve(solver=solver, verbosity=0)
        x0 = {v[name]: conservative[v[name]] for name in ("m", "r", "t")}
        m.localsolve(solver=solver, verbosity=0, x0=x0)
        first = kg(m.program.results[0].cost)
        assert first < CONSERVATIVE_KG * (1 - 1e-3)  # GP 1, not the guess's GP 0

    def test_partial_x0_leaves_guess(self, solver):
        "Values for r and t alone leave m + const, two terms: the guess stands"
        m, _, v = pressure_vessel()
        conservative = m.local_gp().solve(solver=solver, verbosity=0)
        x0 = {v[name]: conservative[v[name]] for name in ("r", "t")}
        m.localsolve(solver=solver, verbosity=0, x0=x0)
        first = kg(m.program.results[0].cost)
        assert first == pytest.approx(CONSERVATIVE_KG, rel=1e-4)

    def test_local_gp_names_unguessed_constraint(self):
        x, y, z = Variable("x"), Variable("y"), Variable("w_bare")
        with SignomialsEnabled():
            bare = x + z >= 1
        m = Model(x * y * z, [guess({x: 1, y: 1}) >= 1, bare, x <= 0.9, z <= 0.9])
        with pytest.raises(InvalidGPConstraint, match="w_bare"):
            m.local_gp()


class TestInfeasibleConservativeGP:
    "The SP is feasible, but its guess drops the term it needs"

    @staticmethod
    def model():
        a, b = Variable("a"), Variable("b")
        return Model(1 / (a * b), [guess({a: 1, b: 0}) >= 1, a <= 0.5, b <= 0.75])

    def test_local_gp(self, solver):
        with pytest.raises(Infeasible):
            self.model().local_gp().solve(solver=solver, verbosity=0)

    def test_localsolve(self, solver):
        with pytest.raises(Infeasible):
            self.model().localsolve(solver=solver, verbosity=0)
