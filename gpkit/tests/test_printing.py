"""Text tables: the sections no example output pins.

The model has a unique optimum with round numbers, so a table's text is
stable across solvers and can be compared exactly.
"""

import numpy as np
import pytest

from gpkit import Model, Variable, VectorVariable


class Leg(Model):
    "a submodel holding a free and a fixed vector"

    def setup(self):
        self.tau = Variable("tau", 2, "s", "leg time")
        self.v = VectorVariable(3, "v", "m/s", "speed")
        self.d = VectorVariable(3, "d", [1, 2, 3], "m", "distance")
        return [self.v >= self.d / self.tau]


class Trip(Model):
    "top level, with one constraint that never binds"

    def setup(self):
        self.leg = Leg()
        self.c = Variable("c", 2, "s", "slack bound")
        self.w = Variable("w", 1, "s^2/m", "speed weight")
        self.x = Variable("x", "s", "free time")
        self.cost = self.x + self.w * self.leg.v.sum()
        return [self.leg, self.x >= self.leg.tau, self.x >= self.c / 10]


@pytest.fixture(name="trip")
def fixture_trip():
    return Trip()


@pytest.fixture(name="farther")
def fixture_farther(trip):
    "the trip solved, then with its last leg longer"
    baseline = trip.solve(verbosity=0)
    trip.substitutions[trip.leg.d] = np.array([1, 2, 4])
    return trip.solve(verbosity=0), baseline


class TestSolutionSections:
    def test_free_and_fixed_variables(self, trip):
        sol = trip.solve(verbosity=0)
        text = trip.table(sol, tables=["cost", "freevariables", "constants"])
        assert text == (
            "Optimal Cost\n"
            "------------\n"
            "Trip.x + Trip.w⋅Trip.Leg.v[:].sum() : 5 [s]\n"
            "\n"
            "Free Variables\n"
            "--------------\n"
            "Trip\n"
            "x : 2 [s] free time\n"
            "\n"
            "Trip.Leg\n"
            "v[:] : [ 0.5  1    1.5 ] [m/s] speed\n"
            "\n"
            "Fixed Variables\n"
            "---------------\n"
            "Trip\n"
            "w : 1 [s²/m] (+0.6) speed weight\n"
            "c : 2 [s]    (~0)   slack bound\n"
            "\n"
            "Trip.Leg\n"
            "tau : 2 [s] (-0.2) leg time\n"
            "d[:] : [ 1     2     3    ] [m] distance\n"
            "  sens ( +0.1  +0.2  +0.3 )"
        )

    def test_an_element_without_a_value_shows_as_nan(self):
        "An element no constraint mentions is never solved for."
        v = VectorVariable(3, "v", "m")
        L = Variable("L", 1, "m")
        sol = Model(v[0] + v[2], [v[0] >= L, v[2] >= 2 * L]).solve(verbosity=0)
        assert sol.table(tables=["freevariables"]).split("\n")[2:] == [
            "v[:] : [ 1    nan  2   ] [m]"
        ]

    def test_slack_constraints_are_the_ones_that_never_bind(self, trip):
        sol = trip.solve(verbosity=0)
        text = trip.table(sol, tables=["slack constraints"])
        lines = text.split("\n")
        assert lines[:3] == ["Insensitive Constraints (below 1e-05)", "-" * 37, "Trip"]
        # the sensitivity is solver noise, so only the constraint is compared
        assert [line.split(" : ")[1] for line in lines[3:]] == ["x ≥ c/10"]


class TestSweepSections:
    def test_a_sweep_stacks_each_element(self, trip):
        sweep = trip.sweep({trip.leg.tau: [2, 4]}, verbosity=0)
        tables = ["sweeps", "cost", "freevariables", "constants"]
        text = trip.table(sweep, tables=tables)
        assert text == (
            "Swept Variables\n"
            "---------------\n"
            "Trip.Leg\n"
            "tau : [ 2  4 ] [s] leg time\n"
            "\n"
            "Optimal Cost\n"
            "------------\n"
            "Trip.x + Trip.w⋅Trip.Leg.v[:].sum() : [ 5    5.5 ] [s]\n"
            "\n"
            "Free Variables\n"
            "--------------\n"
            "Trip\n"
            "x : [ 2  4 ] [s] free time\n"
            "\n"
            "Trip.Leg\n"
            "v[0] : [ 0.5   0.25 ] [m/s] speed\n"
            "v[1] : [ 1     0.5  ] [m/s] speed\n"
            "v[2] : [ 1.5   0.75 ] [m/s] speed\n"
            "\n"
            "Fixed Variables\n"
            "---------------\n"
            "Trip\n"
            " w : [ 1     1      ] [s²/m] speed weight\n"
            "sens ( +0.6  +0.273 )\n"
            " c : [ 2   2  ] [s] slack bound\n"
            "sens ( ~0  ~0 )\n"
            "\n"
            "Trip.Leg\n"
            " tau : [ 2     4      ] [s] leg time\n"
            "  sens ( -0.2  +0.455 )\n"
            "d[2] : [ 3     3      ] [m] distance\n"
            "  sens ( +0.3  +0.136 )\n"
            "d[1] : [ 2     2       ] [m] distance\n"
            "  sens ( +0.2  +0.0909 )\n"
            "d[0] : [ 1     1       ] [m] distance\n"
            "  sens ( +0.1  +0.0455 )"
        )


class TestModelDiff:
    "A diff holds no model, so only the model can name the cost by its expression."

    def test_a_solution_diff_names_the_cost_cost(self, farther):
        new, baseline = farther
        assert "cost : +10%" in new.diff(baseline)

    def test_the_models_diff_names_the_cost_by_its_expression(self, trip, farther):
        new, baseline = farther
        assert "Trip.x + Trip.w⋅Trip.Leg.v[:].sum() : +10%" in trip.diff(new, baseline)

    def test_the_models_diff_takes_a_sweep(self, trip):
        baseline = trip.solve(verbosity=0)
        sweep = trip.sweep({trip.leg.tau: [2, 4]}, verbosity=0)
        assert "Trip.x + Trip.w⋅Trip.Leg.v[:].sum() :" in trip.diff(sweep, baseline)


class TestDiffVectors:
    """A diff pairs each row with the baseline's row for the same quantity.

    Unchanged values move by solver noise, so these compare rows, not text.
    """

    def test_a_vector_constant_is_compared_elementwise(self, farther):
        new, baseline = farther
        rows = new.diff(baseline, tables=["constants"]).split("\n")
        (row,) = [r for r in rows if r.startswith("d[:] :")]
        assert "+33.3%" in row

    def test_a_vector_sensitivity_is_compared_elementwise(self, farther):
        new, baseline = farther
        rows = new.diff(baseline, tables=["sensitivities"]).split("\n")
        assert [r for r in rows if r.startswith("d[:] :")]

    def test_a_redeclared_unit_is_one_row_in_each_unit(self):
        "The diff matches feet to metres, so the table pairs them too."
        x_m, x_ft = Variable("x", "m"), Variable("x", "ft")
        baseline = Model(x_m, [x_m >= Variable("L", 1, "m")]).solve(verbosity=0)
        scenario = Model(x_ft, [x_ft >= Variable("L", 2, "m")]).solve(verbosity=0)
        rows = scenario.diff(baseline, tables=["freevariables"]).split("\n")
        (row,) = [r for r in rows if r.startswith("x :")]
        assert "+100%" in row and "ft vs" in row and "m)" in row

    def test_a_sweep_and_its_baseline_show_vectors_alike(self, trip):
        baseline = trip.solve(verbosity=0)
        sweep = trip.sweep({trip.leg.tau: [2, 4]}, verbosity=0)
        tables = ["freevariables", "constants", "sensitivities"]
        text = sweep.diff(baseline, tables=tables)
        assert "(new)" not in text
        assert "(only in baseline)" not in text
        for name in ("v[0]", "d[2]"):
            assert f"{name} :" in text
