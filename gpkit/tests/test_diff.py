"""Tests for diff_solutions: comparing two solution IR documents."""

import json

import pytest

from gpkit import Model, Variable
from gpkit.ir import diff_solutions


def _sol_ir(cost=1.0, cost_units="m", primal=None, constants=None, sens=None):
    """A minimal Solution.to_ir() document.

    Entries are given as {ref: (value, units)}, so a test can state the one
    thing it is about.
    """

    def entries(d):
        out = {}
        for ref, (value, units) in (d or {}).items():
            out[ref] = {"value": value} | ({"units": units} if units else {})
        return out

    ir = {
        "gpkit_ir_version": "1.0",
        "cost": {"value": cost} | ({"units": cost_units} if cost_units else {}),
        "primal": entries(primal),
        "sensitivities": {
            "variables": dict(sens or {}),
            "constraints": {},
            "models": {},
        },
        "meta": {"status": "optimal", "soltime": 0.0, "warnings": {}},
    }
    if constants:
        ir["constants"] = entries(constants)
    return ir


class TestUnchanged:
    def test_a_solution_against_itself_is_unchanged(self):
        ir = _sol_ir(primal={"x|m": (2.0, "m")}, sens={"x|m": 0.5})
        diff = diff_solutions(ir, ir)
        assert diff["changed"] is False
        assert diff["cost"]["rel"] == pytest.approx(0)
        assert diff["primal"]["matched"]["x|m"] == pytest.approx(0)
        assert "only_in_baseline" not in diff["primal"]
        assert "only_in_scenario" not in diff["primal"]

    def test_a_move_inside_tol_is_not_a_change(self):
        a = _sol_ir(primal={"x|m": (2.0, "m")})
        b = _sol_ir(primal={"x|m": (2.0 * (1 + 1e-9), "m")})
        assert diff_solutions(a, b, tol=1e-6)["changed"] is False
        assert diff_solutions(a, b, tol=1e-12)["changed"] is True


class TestValues:
    def test_a_value_that_moved_reports_a_ratio(self):
        a = _sol_ir(primal={"x|m": (2.0, "m")})
        b = _sol_ir(primal={"x|m": (2.2, "m")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["matched"]["x|m"] == pytest.approx(0.1)
        assert diff["changed"] is True

    def test_cost_reports_a_ratio(self):
        diff = diff_solutions(_sol_ir(cost=100.0), _sol_ir(cost=125.0))
        assert diff["cost"]["rel"] == pytest.approx(0.25)

    def test_growing_away_from_zero_has_an_unbounded_ratio(self):
        "Reported by name, since JSON cannot carry the infinity it really is."
        a = _sol_ir(primal={"x|m": (0.0, "m")})
        b = _sol_ir(primal={"x|m": (1.0, "m")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["from_zero"] == ["x|m"]
        assert "x|m" not in diff["primal"].get("matched", {})
        assert diff["changed"] is True

    def test_shrinking_to_zero_is_an_ordinary_minus_one(self):
        "Only leaving zero is unbounded; arriving at it is -100%."
        a = _sol_ir(primal={"x|m": (4.0, "m")})
        b = _sol_ir(primal={"x|m": (0.0, "m")})
        assert diff_solutions(a, b)["primal"]["matched"]["x|m"] == pytest.approx(-1.0)

    def test_zero_to_zero_is_no_change(self):
        a = _sol_ir(primal={"x|m": (0.0, "m")})
        assert diff_solutions(a, a)["changed"] is False

    def test_constants_are_compared_too(self):
        "Constants are what a scenario actually edits."
        a = _sol_ir(constants={"c|m": (2.0, "m")})
        b = _sol_ir(constants={"c|m": (3.0, "m")})
        assert diff_solutions(a, b)["constants"]["matched"]["c|m"] == pytest.approx(0.5)


class TestIndexSets:
    def test_a_key_only_in_the_baseline(self):
        a = _sol_ir(primal={"x|m": (1.0, "m"), "gone|m": (1.0, "m")})
        b = _sol_ir(primal={"x|m": (1.0, "m")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["only_in_baseline"] == ["gone|m"]
        assert "only_in_scenario" not in diff["primal"]
        assert diff["changed"] is True

    def test_a_key_only_in_the_scenario(self):
        "This case used to raise TypeError inside the renderer's filter."
        a = _sol_ir(primal={"x|m": (1.0, "m")})
        b = _sol_ir(primal={"x|m": (1.0, "m"), "fresh|m": (1.0, "m")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["only_in_scenario"] == ["fresh|m"]
        assert diff["changed"] is True


class TestSensitivities:
    def test_sensitivities_compare_additively(self):
        "They are log derivatives: a difference, not a ratio."
        a = _sol_ir(sens={"x|m": 0.2})
        b = _sol_ir(sens={"x|m": 0.5})
        matched = diff_solutions(a, b)["sensitivities"]["variables"]["matched"]
        assert matched["x|m"] == pytest.approx(0.3)

    def test_a_sign_flip_is_reported_as_the_full_swing(self):
        a = _sol_ir(sens={"x|m": -0.4})
        b = _sol_ir(sens={"x|m": 0.4})
        matched = diff_solutions(a, b)["sensitivities"]["variables"]["matched"]
        assert matched["x|m"] == pytest.approx(0.8)

    def test_a_tiny_sensitivity_doubling_is_not_a_change(self):
        "0.01 -> 0.02 doubles, and is immaterial; additive comparison says so."
        a = _sol_ir(sens={"x|m": 0.01})
        b = _sol_ir(sens={"x|m": 0.02})
        assert diff_solutions(a, b, tol=0.05)["changed"] is False


class TestUnitsAreNotIdentity:
    """A model in feet and a model in metres are the same model.

    Units are a declaration choice, so matching on the literal ref would
    manufacture a difference that does not physically exist.
    """

    def test_the_same_value_declared_in_another_unit_is_unchanged(self):
        a = _sol_ir(cost=1.0, primal={"x|ft": (3.280839895, "ft")})
        b = _sol_ir(cost=1.0, primal={"x|m": (1.0, "m")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["matched"]["x|ft"] == pytest.approx(0, abs=1e-9)
        assert "only_in_baseline" not in diff["primal"]
        assert "only_in_scenario" not in diff["primal"]
        assert diff["changed"] is False

    def test_a_redeclared_unit_records_the_scenario_ref(self):
        "Keyed by the baseline ref, so the scenario's ref has to be stated."
        a = _sol_ir(primal={"x|ft": (3.280839895, "ft")})
        b = _sol_ir(primal={"x|m": (1.0, "m")})
        assert diff_solutions(a, b)["primal"]["units_redeclared"] == {"x|ft": "x|m"}

    def test_a_real_move_still_shows_through_a_unit_change(self):
        a = _sol_ir(primal={"x|ft": (3.280839895, "ft")})  # 1 m
        b = _sol_ir(primal={"x|m": (1.1, "m")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["matched"]["x|ft"] == pytest.approx(0.1, abs=1e-9)

    def test_a_changed_dimension_is_reported_as_such(self):
        "Better than removed + added, which is what a ref match would say."
        a = _sol_ir(primal={"x|N": (1.0, "N")})
        b = _sol_ir(primal={"x|s": (1.0, "s")})
        diff = diff_solutions(a, b)
        assert diff["primal"]["dimension_changed"] == {"x|N": ["N", "s"]}
        assert "x|N" not in diff["primal"].get("matched", {})
        assert diff["changed"] is True

    def test_sensitivities_need_no_conversion(self):
        "Log derivatives are scale-invariant, so a unit change leaves them alone."
        a = _sol_ir(sens={"x|ft": 0.5})
        b = _sol_ir(sens={"x|m": 0.5})
        diff = diff_solutions(a, b)
        assert diff["sensitivities"]["variables"]["matched"]["x|ft"] == pytest.approx(0)

    def test_two_keys_collapsing_to_one_keep_their_units(self):
        "Same name, two units in one document: stripping would collide them."
        a = _sol_ir(primal={"x|m": (1.0, "m"), "x|ft": (1.0, "ft")})
        diff = diff_solutions(a, a)
        assert set(diff["primal"]["matched"]) == {"x|m", "x|ft"}

    def test_dimensionless_entries_match(self):
        a = _sol_ir(cost_units="", primal={"x": (1.0, None)})
        b = _sol_ir(cost_units="", primal={"x": (2.0, None)})
        assert diff_solutions(a, b)["primal"]["matched"]["x"] == pytest.approx(1.0)


def test_the_diff_is_json_serializable():
    a = _sol_ir(primal={"x|m": (1.0, "m")}, sens={"x|m": 0.5})
    b = _sol_ir(primal={"y|m": (2.0, "m")}, sens={"y|m": 0.1})
    diff = diff_solutions(a, b)
    assert json.loads(json.dumps(diff)) == diff


def test_changed_agrees_with_almost_equal():
    "almost_equal is this same comparison collapsed to a boolean."
    x = Variable("x_cmp")
    c = Variable("c_cmp", 2.0)
    m = Model(x, [x >= c])
    sol = m.solve(verbosity=0)
    m.substitutions[c] = 2.5
    other = m.solve(verbosity=0)
    assert sol.almost_equal(sol)
    assert not sol.almost_equal(other)
    assert diff_solutions(sol.to_ir(), sol.to_ir())["changed"] is False
    assert diff_solutions(sol.to_ir(), other.to_ir())["changed"] is True
