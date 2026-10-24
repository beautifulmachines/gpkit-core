"""Text tables: the sections no example output pins.

The model has a unique optimum with round numbers, so a table's text is
stable across solvers and can be compared exactly.
"""

import json

import numpy as np
import pytest

from gpkit import Model, NomialArray, Variable, VectorVariable
from gpkit.printing import Display


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
            " tau : 2 [s] (-0.2) leg time\n"
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

    def test_a_vector_constraint_is_one_row(self, trip):
        "Its sensitivities are per element, gathered as a vector variable's values are."
        sol = trip.solve(verbosity=0)
        lines = trip.table(sol, tables=["tightest constraints"]).split("\n")
        assert lines[-2:] == ["Trip.Leg", "[ +0.1  +0.2  +0.3 ] : v[:] ≥ d[:]/tau"]

    def test_one_never_written_as_a_vector_is_its_elements(self):
        "With no vector form to print, each element is a row, as TOML writes it."
        q, x, y = VectorVariable(2, "q"), Variable("x"), Variable("y")
        m = Model(q.sum(), [q >= 2 * NomialArray([x**0.5, y**0.5]), x >= 1, y >= 1])
        text = m.table(m.solve(verbosity=0), tables=["tightest constraints"])
        assert text.split("\n")[2:4] == [
            "+0.5  : q[0] ≥ 2⋅x^0.5",
            "+0.5  : q[1] ≥ 2⋅y^0.5",
        ]

    def test_scalar_and_vector_rows_align_separately(self):
        "A vector's sensitivities are far wider; scalar rows keep their own column."
        x, y = VectorVariable(3, "x"), Variable("y")
        m = Model(y, [y >= x.sum(), x >= 1])
        lines = m.table(m.solve(verbosity=0), tables=["tightest constraints"])
        assert lines.split("\n")[2:] == [
            "+1 : y ≥ x[:].sum()",
            "[ +0.333  +0.333  +0.333 ] : x[:] ≥ 1",
        ]

    def test_a_vanishing_sensitivity_reads_near_zero(self):
        "As in the fixed variables' column, and without widening its neighbours."
        x = VectorVariable(3, "x")
        m = Model(x[0] + x[2], [x >= 1, x[1] <= 2])  # nothing presses on x[1]
        lines = m.table(m.solve(verbosity=0), tables=["tightest constraints"])
        assert lines.split("\n")[-1] == "[ +0.5  ~0    +0.5 ] : x[:] ≥ 1"

    def test_slack_constraints_are_the_ones_that_never_bind(self, trip):
        sol = trip.solve(verbosity=0)
        text = trip.table(sol, tables=["slack constraints"])
        lines = text.split("\n")
        assert lines[:3] == ["Insensitive Constraints (below 1e-05)", "-" * 37, "Trip"]
        # the sensitivity is solver noise, so only the constraint is compared
        assert [line.split(" : ")[1] for line in lines[3:]] == ["x ≥ c/10"]


class TestDisplay:
    "A row's name, units, label and heading come from a plain variable entry."

    def test_rows_read_the_model_irs_variable_entries(self, trip):
        entries = json.loads(json.dumps(trip.to_ir()["variables"]))
        display = Display(entries)
        element = trip.leg.v[0].key
        for ref, name in [(element.ref, "v[0]"), (element.veckey.ref, "v[:]")]:
            assert display.name(ref) == name
            assert display.units(ref) == "m/s"
            assert display.label(ref) == "speed"
            assert display.heading(ref) == "Trip.Leg"
        assert display.units(trip.w.key.ref) == "s²/m"


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


class TestMarkdown:
    "fmt='md' is the same rows as the text, one pipe table per model."

    def test_free_and_fixed_variables(self, trip):
        sol = trip.solve(verbosity=0)
        tables = ["cost", "freevariables", "constants"]
        assert trip.table(sol, tables=tables, fmt="md") == (
            "### Optimal Cost\n"
            "\n"
            "| Cost | Value | Units |\n"
            "| ---- | ----- | ----- |\n"
            "| Trip.x + Trip.w⋅Trip.Leg.v\\[:\\].sum() | 5 | s |\n"
            "\n"
            "### Free Variables\n"
            "\n"
            "**Trip**\n"
            "\n"
            "| Variable | Value | Units | Label |\n"
            "| -------- | ----- | ----- | ----- |\n"
            "| x | 2 | s | free time |\n"
            "\n"
            "**Trip.Leg**\n"
            "\n"
            "| Variable | Value | Units | Label |\n"
            "| -------- | ----- | ----- | ----- |\n"
            "| v\\[:\\] | \\[ 0.5  1  1.5 \\] | m/s | speed |\n"
            "\n"
            "### Fixed Variables\n"
            "\n"
            "**Trip**\n"
            "\n"
            "| Variable | Value | Units | Sensitivity | Label |\n"
            "| -------- | ----- | ----- | ----------- | ----- |\n"
            "| w | 1 | s²/m | +0.6 | speed weight |\n"
            "| c | 2 | s | \\~0 | slack bound |\n"
            "\n"
            "**Trip.Leg**\n"
            "\n"
            "| Variable | Value | Units | Sensitivity | Label |\n"
            "| -------- | ----- | ----- | ----------- | ----- |\n"
            "| d\\[:\\] | \\[ 1  2  3 \\] | m | \\[ +0.1  +0.2  +0.3 \\] | distance |\n"
            "| tau | 2 | s | -0.2 | leg time |"
        )

    def test_constraints(self, trip):
        sol = trip.solve(verbosity=0)
        assert trip.table(sol, tables=["tightest constraints"], fmt="md") == (
            "### Most Sensitive Constraints\n"
            "\n"
            "**Trip**\n"
            "\n"
            "| Sensitivity | Constraint |\n"
            "| ----------- | ---------- |\n"
            "| +0.4 | x ≥ tau |\n"
            "\n"
            "**Trip.Leg**\n"
            "\n"
            "| Sensitivity | Constraint |\n"
            "| ----------- | ---------- |\n"
            "| \\[ +0.1  +0.2  +0.3 \\] | v\\[:\\] ≥ d\\[:\\]/tau |"
        )

    def test_a_sweep_stacks_each_element(self, trip):
        sweep = trip.sweep({trip.leg.tau: [2, 4]}, verbosity=0)
        assert trip.table(sweep, tables=["sweeps", "freevariables"], fmt="md") == (
            "### Swept Variables\n"
            "\n"
            "**Trip.Leg**\n"
            "\n"
            "| Variable | Value | Units | Label |\n"
            "| -------- | ----- | ----- | ----- |\n"
            "| tau | \\[ 2  4 \\] | s | leg time |\n"
            "\n"
            "### Free Variables\n"
            "\n"
            "**Trip**\n"
            "\n"
            "| Variable | Value | Units | Label |\n"
            "| -------- | ----- | ----- | ----- |\n"
            "| x | \\[ 2  4 \\] | s | free time |\n"
            "\n"
            "**Trip.Leg**\n"
            "\n"
            "| Variable | Value | Units | Label |\n"
            "| -------- | ----- | ----- | ----- |\n"
            "| v\\[0\\] | \\[ 0.5  0.25 \\] | m/s | speed |\n"
            "| v\\[1\\] | \\[ 1  0.5 \\] | m/s | speed |\n"
            "| v\\[2\\] | \\[ 1.5  0.75 \\] | m/s | speed |"
        )

    def test_a_diff(self, trip, farther):
        new, baseline = farther
        assert trip.diff(new, baseline, fmt="md") == (
            "### Cost Change\n"
            "\n"
            "| Cost | Change | Values |\n"
            "| ---- | ------ | ------ |\n"
            "| Trip.x + Trip.w⋅Trip.Leg.v\\[:\\].sum() | +10\\% | (5.5s vs 5s) |\n"
            "\n"
            "### Free Variable Changes\n"
            "\n"
            "**Trip.Leg**\n"
            "\n"
            "| Variable | Change | Label |\n"
            "| -------- | ------ | ----- |\n"
            "| v\\[:\\] | \\[ +0\\%  +0\\%  +33.3\\% \\] | speed |"
        )

    def test_the_models_table_is_its_sections_only(self, trip):
        "Breakdowns are text art, kept only until budgets replace them."
        sol = trip.solve(verbosity=0)
        assert trip.table(sol, fmt="md").startswith("### Optimal Cost\n")

    def test_an_unknown_format_is_an_error(self, trip):
        sol = trip.solve(verbosity=0)
        with pytest.raises(ValueError, match="html"):
            sol.table(fmt="html")


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


class TestDiffTolerance:
    """The table moves what sol.almost_equal says moved, and nothing else.

    A solver returns an unchanged value to within rounding, not exactly.
    """

    def test_an_unchanged_variable_has_no_row(self, farther):
        new, baseline = farther
        rows = new.diff(baseline).split("\n")
        assert not [r for r in rows if r.startswith("x :")]

    def test_an_unchanged_element_reads_zero(self, farther):
        new, baseline = farther
        (row,) = [r for r in new.diff(baseline).split("\n") if r.startswith("v[:]")]
        assert row.split()[2:6] == ["[", "+0%", "+0%", "+33.3%"]

    def test_tol_decides_what_moved(self, farther):
        new, baseline = farther
        assert "v[:]" not in new.diff(baseline, tol=0.5)


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
