"""Tests for gpkit.budgets — variable budget computation and display."""

import pytest

from gpkit import Model, Variable, units
from gpkit.budgets import Budget, build_budget, find_budget_constraints

# ---------------------------------------------------------------------------
# Test models
# Each submodel exposes self.m (the mass Variable, with kg units) and uses
# properly-unitized lower bounds so GP constraints are dimensionally consistent.
# ---------------------------------------------------------------------------


class Spar(Model):
    """Spar submodel fixture: single mass variable with a lower bound."""

    m: Variable

    def setup(self):
        m_min = Variable("m_min", 10, "kg", "spar minimum mass")
        self.m = Variable("m", "kg", "spar mass")
        self.cost = self.m
        return [self.m >= m_min]


class Skin(Model):
    """Skin submodel fixture: single mass variable with a lower bound."""

    m: Variable

    def setup(self):
        m_min = Variable("m_min", 5, "kg", "skin minimum mass")
        self.m = Variable("m", "kg", "skin mass")
        self.cost = self.m
        return [self.m >= m_min]


class Wing(Model):
    """Wing submodel fixture: mass budgeted across spar and skin."""

    spar: Spar
    skin: Skin
    m: Variable

    def setup(self):
        self.spar = Spar()
        self.skin = Skin()
        self.m = Variable("m", "kg", "wing mass")
        self.cost = self.m
        return [self.m >= self.spar.m + self.skin.m, self.spar, self.skin]


class Aircraft(Model):
    """Top-level aircraft fixture: total mass budgeted by wing mass."""

    wing: Wing
    m: Variable

    def setup(self):
        self.wing = Wing()
        self.m = Variable("m", "kg", "total mass")
        self.cost = self.m
        return [self.m >= self.wing.m, self.wing]


class AircraftWithMargin(Model):
    """Budget constraint includes a self-referential margin term."""

    wing: Wing
    m: Variable

    def setup(self):
        self.wing = Wing()
        f_margin = Variable("f_margin", 0.1, label="mass margin fraction")
        self.m = Variable("m", "kg", "total mass")
        self.cost = self.m
        # m >= wing.m + f_margin * m  →  m*(1-0.1) >= wing.m  at optimum
        return [self.m >= self.wing.m + f_margin * self.m, self.wing]


class SlackModel(Model):
    """Budget constraint that is NOT tight at the optimum.

    m_total is fixed at 100 kg; m_wing optimizes to its lower bound of 60 kg.
    The budget constraint m_total >= m_wing is slack (100 > 60).
    """

    m_wing: Variable
    m_total: Variable

    def setup(self):
        m_wing_min = Variable("m_wing_min", 60, "kg", "wing mass lower bound")
        self.m_wing = Variable("m_wing", "kg", "wing mass")
        self.m_total = Variable("m_total", 100, "kg", "total mass (fixed)")
        self.cost = self.m_wing
        return [
            self.m_total >= self.m_wing,  # budget constraint (slack: 100 > 60)
            self.m_wing >= m_wing_min,  # wing: forces m_wing = 60 kg
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def solve(model):
    """Solve *model* silently and return (solution, model)."""
    sol = model.solve(verbosity=0)
    return sol, model


# ---------------------------------------------------------------------------
# Tests: find_budget_constraints
# ---------------------------------------------------------------------------


class TestFindBudgetConstraints:
    """Tests for find_budget_constraints()."""

    def test_wing_budget(self):
        model = Aircraft()
        sol, _ = solve(model)
        matches = find_budget_constraints(model, model.wing.m.key, sol)
        # Wing has one budget constraint: wing.m >= spar.m + skin.m
        assert len(matches) >= 1
        _, lt, _ = matches[0]
        # lt should have two terms (spar.m and skin.m)
        assert len(lt.hmap) == 2

    def test_top_level(self):
        model = Aircraft()
        sol, _ = solve(model)
        matches = find_budget_constraints(model, model.m.key, sol)
        assert len(matches) >= 1

    def test_leaf_budget_has_constant_term(self):
        model = Aircraft()
        sol, _ = solve(model)
        # Spar mass has budget constraint m >= m_min, where m_min is a constant
        matches = find_budget_constraints(model, model.wing.spar.m.key, sol)
        assert len(matches) >= 1
        _, lt, _ = matches[0]
        # m_min is a substituted constant, so lt has exactly one constant term
        assert len(lt.hmap) == 1


# ---------------------------------------------------------------------------
# Tests: build_budget — basic decomposition
# ---------------------------------------------------------------------------


class TestBuildBudgetBasic:
    """Tests for build_budget() basic decomposition."""

    def test_returns_budget(self):
        model = Aircraft()
        sol, _ = solve(model)
        assert isinstance(build_budget(sol, model, model.m), Budget)

    def test_total_correct(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert abs(b.total - float(sol[model.m].magnitude)) < 1e-6

    def test_units(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert b.units == "kg"

    def test_children_sum_to_total(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        child_sum = sum(n.value for n in b.children)
        assert abs(child_sum - b.total) / b.total < 1e-4

    def test_fractions_sum_to_one(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        frac_sum = sum(n.fraction for n in b.children)
        assert abs(frac_sum - 1.0) < 1e-4

    def test_recursion_into_wing(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # One child (wing.m), which itself has two children (spar.m, skin.m)
        assert len(b.children) == 1
        wing_node = b.children[0]
        assert len(wing_node.children) == 2

    def test_display_units(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m, display_units="g")
        assert b.units == "g"
        assert b.total > 1000  # 15 kg → 15000 g


# ---------------------------------------------------------------------------
# Tests: build_budget — margin / self-referential term
# ---------------------------------------------------------------------------


class TestBuildBudgetMargin:
    """Tests for build_budget() with self-referential margin terms."""

    def test_margin_term_present(self):
        model = AircraftWithMargin()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        labels = [n.label for n in b.children]
        assert any("[margin]" in lbl for lbl in labels)

    def test_margin_term_value(self):
        model = AircraftWithMargin()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        margin_nodes = [n for n in b.children if "[margin]" in n.label]
        assert len(margin_nodes) == 1
        # f_margin = 0.1, wing.m = 15 kg → m_total ≈ 16.67 kg, margin ≈ 1.67 kg
        assert margin_nodes[0].value > 0

    def test_children_sum_with_margin(self):
        model = AircraftWithMargin()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        child_sum = sum(n.value for n in b.children)
        assert abs(child_sum - b.total) / b.total < 1e-4


# ---------------------------------------------------------------------------
# Tests: slack detection
# ---------------------------------------------------------------------------


class TestBudgetSlack:
    """Tests for slack detection when a budget constraint is not tight."""

    def test_slack_node_added(self):
        model = SlackModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m_total)
        labels = [n.label for n in b.children]
        assert "[slack]" in labels

    def test_slack_node_value(self):
        model = SlackModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m_total)
        slack_node = next(n for n in b.children if n.label == "[slack]")
        # m_total=100 kg, m_wing=60 kg → slack = 40 kg
        assert slack_node.value == pytest.approx(40.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: Solution.budget() method
# ---------------------------------------------------------------------------


class TestSolutionBudgetMethod:
    """Tests for the Solution.budget() convenience method."""

    def test_method_returns_budget(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = sol.budget(model.m)
        assert isinstance(b, Budget)

    def test_method_no_model_raises(self):
        model = Aircraft()
        sol, _ = solve(model)
        del sol.meta["model"]
        with pytest.raises(ValueError, match="No model in solution"):
            sol.budget(model.m)


# ---------------------------------------------------------------------------
# Tests: rendering
# ---------------------------------------------------------------------------


class TestBudgetRendering:
    """Tests for Budget.text(), .markdown(), .to_dict(), and __repr__."""

    def test_text_contains_units(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert "[kg]" in b.text()

    def test_text_contains_percent(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert "%" in b.text()

    def test_markdown_has_table(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        md = b.markdown()
        assert "| --- |" in md
        assert "[kg]" in md

    def test_to_dict(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        d = b.to_dict()
        assert d["units"] == "kg"
        assert isinstance(d["children"], list)
        assert d["total"] == pytest.approx(b.total)

    def test_repr_is_text(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert repr(b) == b.text()


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestBudgetErrors:
    """Tests for error handling in build_budget()."""

    def test_type_error_on_string(self):
        model = Aircraft()
        sol, _ = solve(model)
        with pytest.raises(TypeError):
            build_budget(sol, model, "m")

    def test_spar_budget_leaf(self):
        model = Aircraft()
        sol, _ = solve(model)
        # Spar.m's budget terminates at a constant — no named-variable children
        b = build_budget(sol, model, model.wing.spar.m)
        named_children = [n for n in b.children if n.vk is not None]
        assert len(named_children) == 0


# ---------------------------------------------------------------------------
# Tests: mixed units (issue #161)
# ---------------------------------------------------------------------------


class Cylinder(Model):
    """Cylinder model with variables spanning different unit dimensions."""

    def setup(self):
        self.m = Variable("m", "kg", "mass")
        V = Variable("V", "m^3", "volume")
        rho = Variable("rho", 7800, "kg/m^3", "density")
        L = Variable("L", 1, "m", "length")
        A = Variable("A", "m^2", "cross-sectional area")
        self.cost = self.m
        return [
            self.m >= rho * V,
            V >= L * A,
            A >= 0.01 * units("m^2"),
        ]


class TestBuildBudgetMixedUnits:
    """Tests for budget() when sub-variables have different units than top variable."""

    def test_no_crash(self):
        # Should not raise pint.DimensionalityError (issue #161)
        model = Cylinder()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert isinstance(b, Budget)

    def test_total_correct(self):
        model = Cylinder()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # rho=7800 kg/m^3, L=1 m, A=0.01 m^2 → m = 78 kg
        assert abs(b.total - 78.0) < 1e-4

    def test_children_have_finite_values(self):
        model = Cylinder()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert all(c.value == c.value for c in b.children)  # no NaN

    def test_solution_budget_method(self):
        model = Cylinder()
        sol = model.solve(verbosity=0)
        b = sol.budget(model.m)
        assert isinstance(b, Budget)

    def test_label_includes_constants(self):
        # m >= rho*V: child label should show "rho·V", not just "V"
        model = Cylinder()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert any("rho" in c.label for c in b.children)

    def test_no_cross_unit_recursion(self):
        # rho·V contributes to m, but V (m^3) != m (kg): should not recurse into V
        model = Cylinder()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        rho_v_node = b.children[0]
        assert rho_v_node.children == []
