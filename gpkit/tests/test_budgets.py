"""Tests for gpkit.budgets — variable budget computation and display."""

# pylint: disable=attribute-defined-outside-init

import math

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

    def test_child_labels_include_model_context(self):
        # Child labels must include lineage so "m" is not ambiguous
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        wing_node = b.children[0]
        assert "Wing" in wing_node.label
        spar_node = next(n for n in wing_node.children if abs(n.value - 10) < 1e-3)
        assert "Spar" in spar_node.label

    def test_child_labels_drop_parent_lineage(self):
        # Under Aircraft, the wing node should be "Wing.m" not "Aircraft.Wing.m"
        # Under Wing, the spar node should be "Spar.m" not "Aircraft.Wing.Spar.m"
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        wing_node = b.children[0]
        assert wing_node.label == "Wing.m"
        spar_node = next(n for n in wing_node.children if abs(n.value - 10) < 1e-3)
        assert spar_node.label == "Spar.m"


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
# Tests: growth allowance integration
# ---------------------------------------------------------------------------


class GrowthSpar(Model):
    """Leaf component with a 20% growth allowance."""

    m: Variable

    def setup(self):
        e = Variable("e", 50, "kg", "spar nominal estimate")
        self.m = Variable("m", "kg", "spar mass", growth=0.20)
        self.cost = self.m
        return self.m.grown_from(e)


class GrowthSkin(Model):
    """Leaf component without a growth allowance."""

    m: Variable

    def setup(self):
        e = Variable("e", 30, "kg", "skin nominal estimate")
        self.m = Variable("m", "kg", "skin mass")
        self.cost = self.m
        return [self.m >= e]


class GrowthWing(Model):
    """Wing with 10% subsystem growth budgeted across two spars."""

    spar1: GrowthSpar
    spar2: GrowthSpar
    m: Variable

    def setup(self):
        self.spar1 = GrowthSpar()
        self.spar2 = GrowthSpar()
        self.m = Variable("m", "kg", "wing mass", growth=0.10)
        self.cost = self.m
        return [
            self.m.grown_from(self.spar1.m + self.spar2.m),
            self.spar1,
            self.spar2,
        ]


class GrowthMixedWing(Model):
    """Wing with 10% growth, mixing a growth-enabled spar and a plain skin."""

    spar: GrowthSpar
    skin: GrowthSkin
    m: Variable

    def setup(self):
        self.spar = GrowthSpar()
        self.skin = GrowthSkin()
        self.m = Variable("m", "kg", "wing mass", growth=0.10)
        self.cost = self.m
        return [
            self.m.grown_from(self.spar.m + self.skin.m),
            self.spar,
            self.skin,
        ]


class TestBudgetWithGrowth:
    """Budget walker peels off allowance terms and computes Nominal/Growth columns."""

    def test_allowance_term_not_rendered_as_child(self):
        model = GrowthSpar()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        labels = [n.label for n in b.children]
        assert not any("growth" in (lbl or "") for lbl in labels)

    def test_leaf_cbe_plus_ga_equals_total(self):
        model = GrowthSpar()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # m = 1.20 * 50 = 60; cbe = 50 (from leaf expr); ga = 10
        assert b.total == pytest.approx(60.0, rel=1e-4)
        assert b.cbe_total == pytest.approx(50.0, rel=1e-4)
        assert b.ga_total == pytest.approx(10.0, rel=1e-4)
        assert b.cbe_total + b.ga_total == pytest.approx(b.total, rel=1e-4)

    def test_two_level_recursive_ga_accumulation(self):
        model = GrowthWing()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # spar_total = 60 each; wing_cbe constraint = 120; wing.m = 132
        # leaf cbe sum (recursive) = 50 + 50 = 100
        # ga (recursive) = 132 - 100 = 32 (Wing.m_growth=12 + 2*Spar.m_growth=20)
        assert b.total == pytest.approx(132.0, rel=1e-4)
        assert b.cbe_total == pytest.approx(100.0, rel=1e-4)
        assert b.ga_total == pytest.approx(32.0, rel=1e-4)

    def test_invariant_at_every_node(self):
        model = GrowthWing()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        for node in _walk_all_nodes(b.children):
            assert node.cbe_value + node.ga_value == pytest.approx(node.value, rel=1e-4)

    def test_mixed_growth_and_plain_children(self):
        model = GrowthMixedWing()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # Spar: cbe=50, ga=10, total=60
        # Skin: cbe=30, ga=0, total=30 (no growth declared)
        # Wing.m_cbe = spar.m + skin.m = 90; Wing.m_growth = 9; Wing.m = 99
        # Recursive cbe at wing = 50 + 30 = 80; ga = 99 - 80 = 19
        assert b.total == pytest.approx(99.0, rel=1e-4)
        assert b.cbe_total == pytest.approx(80.0, rel=1e-4)
        assert b.ga_total == pytest.approx(19.0, rel=1e-4)
        # Skin's row should have ga=0
        skin_node = next(
            n
            for n in b.children
            if n.vk and n.vk.name == "m" and "skin" in n.vk.lineagestr().lower()
        )
        assert skin_node.ga_value == pytest.approx(0.0, abs=1e-4)


class TestBudgetRenderingWithGrowth:
    """Text/markdown render adds Nominal and Growth columns when growth is present."""

    def test_text_includes_nominal_and_growth_columns(self):
        model = GrowthWing()
        sol, _ = solve(model)
        out = build_budget(sol, model, model.m).text()
        assert "Nominal" in out
        assert "Growth" in out

    def test_markdown_includes_nominal_and_growth_columns(self):
        model = GrowthWing()
        sol, _ = solve(model)
        out = build_budget(sol, model, model.m).markdown()
        assert "Nominal" in out
        assert "Growth" in out

    def test_text_unchanged_for_no_growth_budget(self):
        model = Aircraft()
        sol, _ = solve(model)
        out = build_budget(sol, model, model.m).text()
        assert "Nominal" not in out
        assert "Growth" not in out

    def test_to_dict_always_has_cbe_ga(self):
        model = GrowthSpar()
        sol, _ = solve(model)
        d = build_budget(sol, model, model.m).to_dict()
        assert "cbe_total" in d
        assert "ga_total" in d
        assert d["cbe_total"] + d["ga_total"] == pytest.approx(d["total"], rel=1e-4)


class TestBudgetGrowthCellBlanking:
    """GA cell renders blank for rows with no growth in their subtree."""

    def test_leaf_row_in_growth_subtree_has_blank_ga(self):
        model = GrowthSpar()
        sol, _ = solve(model)
        out = build_budget(sol, model, model.m).text()
        # Row for the leaf 'e' should NOT contain a numeric Growth value
        # ("0", "0.0", "1e-08", etc). Find the e-row and check its GA col.
        e_line = next(
            line for line in out.splitlines() if line.lstrip().startswith("e ")
        )
        # Three numeric columns: Nominal, Growth, Total. Growth should be blank.
        # Split-from-the-right since the Total/Units/Frac suffix is fixed.
        tokens = e_line.split()
        # Tokens: ['e', nominal, total, '[kg]', pct]  (no growth token)
        assert tokens == ["e", "50", "50", "[kg]", "83.3%"]

    def test_non_growth_submodel_has_blank_ga_despite_noise(self):
        model = GrowthMixedWing()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # Find the GrowthSkin.m node (no growth declared)
        skin_node = next(n for n in b.children if n.vk and "Skin" in n.label)
        assert skin_node.has_growth is False
        # Even if numerical noise made ga_value tiny non-zero, render should
        # be blank. Check the rendered line.
        out = b.text()
        skin_line = next(line for line in out.splitlines() if "GrowthSkin.m" in line)
        # No numeric token between Nominal=30 and Total=30
        tokens = skin_line.split()
        assert tokens == ["GrowthSkin.m", "30", "30", "[kg]", "30.3%"]

    def test_growth_enabled_row_still_renders_numeric_ga(self):
        model = GrowthSpar()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert b.has_growth is True
        # The top-row entry in the rendered table should carry the numeric GA
        out = b.text()
        # The first row beneath the divider line is the top-level "Spar.m"
        # row; it should contain "10" (the growth allowance value).
        top_row = next(
            line
            for line in out.splitlines()
            if "GrowthSpar" in line and "Budget" not in line
        )
        tokens = top_row.split()
        # Tokens: ['GrowthSpar.m', nominal=50, growth=10, total=60, units, frac]
        assert "10" in tokens

    def test_has_growth_propagates_in_to_dict(self):
        model = GrowthSpar()
        sol, _ = solve(model)
        d = build_budget(sol, model, model.m).to_dict()
        assert d["has_growth"] is True
        # Find the leaf 'e' child — it should have has_growth False
        e_child = next(c for c in d["children"] if c["label"] == "e")
        assert e_child["has_growth"] is False


def _walk_all_nodes(nodes):
    "Yield every BudgetNode in a tree (depth-first)."
    for n in nodes:
        yield n
        yield from _walk_all_nodes(n.children)


# ---------------------------------------------------------------------------
# Tests: unit-mismatch coefficient bug
# ---------------------------------------------------------------------------


class MixedUnitMass(Model):
    """Budget constraint where variables have compatible but different mass units.

    m (kg) >= m_pay (kg) + m_struct (g) — physically correct but GP coefficient
    for m_struct will be 0.001 (g/kg), NOT 1.0.  The budget must show the
    physical value (m_struct converted to kg), not 0.001 * m_struct_in_g.
    """

    def setup(self):
        self.m = Variable("m", "kg", "total mass")
        self.m_struct = Variable("m_struct", "g", "structural mass (in grams)")
        m_pay = Variable("m_pay", 100, "kg", "payload")
        f = Variable("f", 0.03, "-", "structural fraction")
        self.cost = self.m
        return [
            self.m >= m_pay + self.m_struct,
            self.m_struct >= f * self.m,
        ]


class TestBudgetUnitMismatchCoeff:
    """GP hmap coefficient encodes unit conversion; budget must use physical value."""

    def test_value_is_physical(self):
        model = MixedUnitMass()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        struct_node = next(
            n for n in b.children if n.vk is not None and "m_struct" in n.label
        )
        # m_struct is declared in grams; value should be in g, not the hmap coefficient
        assert struct_node.units == "g"
        expected_g = float(sol[model.m_struct].to("g").magnitude)
        assert abs(struct_node.value - expected_g) / expected_g < 1e-4

    def test_label_has_no_spurious_coeff(self):
        model = MixedUnitMass()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        struct_node = next(
            n for n in b.children if n.vk is not None and "m_struct" in n.label
        )
        # label should not start with a numeric coefficient like "0.001·m_struct"
        assert not struct_node.label[0].isdigit()


# ---------------------------------------------------------------------------
# Tests: mixed units (issue #161)
# ---------------------------------------------------------------------------


class Cylinder(Model):
    """Cylinder model with variables spanning different unit dimensions."""

    def setup(self):  # pylint: disable=attribute-defined-outside-init
        self.m = Variable("m", "kg", "mass")
        vol = Variable("V", "m^3", "volume")
        rho = Variable("rho", 7800, "kg/m^3", "density")
        length = Variable("L", 1, "m", "length")
        area = Variable("A", "m^2", "cross-sectional area")
        self.cost = self.m
        return [
            self.m >= rho * vol,
            vol >= length * area,
            area >= 0.01 * units("m^2"),
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


# ---------------------------------------------------------------------------
# Tests: mixed-unit coefficient (issue #162)
# ---------------------------------------------------------------------------


class MixedUnitCoeffModel(Model):
    """Constraint has terms in different units; hmap carries conversion factor."""

    def setup(self):  # pylint: disable=attribute-defined-outside-init
        self.m = Variable("m", "kg", "total mass")
        m_a = Variable("m_a", 1, "lbs", "component A (1 lb ≈ 0.4536 kg)")
        m_b = Variable("m_b", 1, "kg", "component B (1 kg)")
        self.cost = self.m
        return [self.m >= m_a + m_b]


class TestMixedUnitCoeff:
    """Values must be correct when unit conversion is baked into hmap coefficient."""

    def test_children_sum_to_total(self):
        model = MixedUnitCoeffModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        child_sum = sum(n.value for n in b.children)
        assert abs(child_sum - b.total) / b.total < 1e-4

    def test_component_values_correct(self):
        """m_a ≈ 0.4536 kg, m_b = 1.0 kg."""
        model = MixedUnitCoeffModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        m_b_node = next(n for n in b.children if "m_b" in n.label)
        assert m_b_node.value == pytest.approx(1.0, rel=1e-4)
        m_a_node = next(n for n in b.children if "m_a" in n.label)
        assert m_a_node.value == pytest.approx(0.45359237, rel=1e-4)

    def test_labels_show_physical_coeff(self):
        """Pure unit-conversion terms should NOT show a numeric coefficient."""
        model = MixedUnitCoeffModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # m_a and m_b each appear with coefficient 1 in the original expression —
        # neither label should start with a digit (no spurious "0.4536·m_a")
        for node in b.children:
            assert not node.label[
                0
            ].isdigit(), f"spurious coeff in label: {node.label!r}"


# ---------------------------------------------------------------------------
# Tests: physical coefficient preserved in label (issue #163)
# ---------------------------------------------------------------------------


class PhysCoeffModel(Model):
    """Budget where one term has a genuine physical scale factor (0.25)."""

    def setup(self):  # pylint: disable=attribute-defined-outside-init
        self.m = Variable("m", "kg", "total mass")
        m_a = Variable("m_a", 4, "kg", "component A")
        m_b = Variable("m_b", 1, "kg", "component B")
        self.cost = self.m
        # 0.25 is a real physical factor, not a unit artifact
        return [self.m >= 0.25 * m_a + m_b]


class TestPhysCoeff:
    """Physical coefficients (not unit conversions) must appear in the label."""

    def test_coeff_shown_in_label(self):
        model = PhysCoeffModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        # 0.25*m_a contributes 0.25*4 = 1 kg; label should mention 0.25
        m_a_node = next(n for n in b.children if "m_a" in n.label)
        assert "0.25" in m_a_node.label

    def test_value_correct(self):
        model = PhysCoeffModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        m_a_node = next(n for n in b.children if "m_a" in n.label)
        assert m_a_node.value == pytest.approx(1.0, rel=1e-4)  # 0.25 * 4 kg

    def test_children_sum_to_total(self):
        model = PhysCoeffModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        child_sum = sum(n.value for n in b.children)
        assert abs(child_sum - b.total) / b.total < 1e-4


# ---------------------------------------------------------------------------
# Tests: per-row units column
# ---------------------------------------------------------------------------


class TestBudgetNodeUnits:
    """Each BudgetNode carries its own units; mixed-unit budgets show per-row units."""

    def test_uniform_units_all_kg(self):
        """Aircraft budget: all nodes in kg, units column shows kg everywhere."""
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        wing_node = b.children[0]
        assert wing_node.units == "kg"
        for child in wing_node.children:
            assert child.units == "kg"

    def test_mixed_unit_node_shows_native_units(self):
        """m_struct (declared in g) should have units='g' and value in grams."""
        model = MixedUnitMass()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        struct_node = next(
            n for n in b.children if n.vk is not None and "m_struct" in n.label
        )
        assert struct_node.units == "g"
        # Value in grams should be ~1000x the kg value
        kg_val = float(sol[model.m_struct].to("kg").magnitude)
        assert struct_node.value == pytest.approx(kg_val * 1000, rel=1e-3)

    def test_fraction_consistent_across_units(self):
        """Fractions sum to 1 regardless of per-row units."""
        model = MixedUnitMass()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        frac_sum = sum(n.fraction for n in b.children)
        assert abs(frac_sum - 1.0) < 1e-4

    def test_text_units_column(self):
        """text() output contains a units column with per-row units."""
        model = MixedUnitMass()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        text = b.text()
        assert "[g]" in text
        assert "[kg]" in text

    def test_markdown_units_column(self):
        """markdown() contains a Units column header and per-row units."""
        model = MixedUnitMass()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        md = b.markdown()
        assert "| Units |" in md
        assert "[g]" in md

    def test_to_dict_includes_units(self):
        """to_dict() includes a 'units' key on each child node."""
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        child_dict = b.to_dict()["children"][0]
        assert "units" in child_dict
        assert child_dict["units"] == "kg"

    def test_slack_node_units(self):
        """Slack node units match the budget level units."""
        model = SlackModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m_total)
        slack_node = next(n for n in b.children if n.label == "[slack]")
        assert slack_node.units == "kg"


# ---------------------------------------------------------------------------
# Tests: depth parameter
# ---------------------------------------------------------------------------


class TestBudgetDepth:
    """Tests for build_budget() depth parameter."""

    def test_depth_zero_no_children(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m, depth=0)
        assert not b.children

    def test_depth_one_no_grandchildren(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m, depth=1)
        assert len(b.children) == 1
        wing_node = b.children[0]
        assert not wing_node.children

    def test_depth_two_has_grandchildren(self):
        model = Aircraft()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m, depth=2)
        assert len(b.children) == 1
        wing_node = b.children[0]
        assert len(wing_node.children) == 2
        for grandchild in wing_node.children:
            assert not grandchild.children

    def test_depth_inf_same_as_default(self):
        model = Aircraft()
        sol, _ = solve(model)
        b_default = build_budget(sol, model, model.m)
        b_inf = build_budget(sol, model, model.m, depth=math.inf)
        assert len(b_default.children) == len(b_inf.children)
        assert len(b_default.children[0].children) == len(b_inf.children[0].children)

    def test_solution_budget_depth_kwarg(self):
        """Solution.budget() accepts and respects depth=."""
        model = Aircraft()
        sol, _ = solve(model)
        b = sol.budget(model.m, depth=1)
        assert len(b.children) == 1
        assert not b.children[0].children


# ---------------------------------------------------------------------------
# Tests: dimensionless top-level budget variable (Bug 1)
# ---------------------------------------------------------------------------


class DimensionlessBudgetModel(Model):
    """Budget where the top-level variable is dimensionless."""

    def setup(self):  # pylint: disable=attribute-defined-outside-init
        self.f = Variable("f", "-", "total fraction")
        f_a = Variable("f_a", "-", "fraction a")
        f_b = Variable("f_b", "-", "fraction b")
        self.cost = self.f
        return [
            self.f >= f_a + f_b,
            f_a >= Variable("f_a_min", 0.3, "-"),
            f_b >= Variable("f_b_min", 0.4, "-"),
        ]


class TestDimensionlessBudget:
    """build_budget() must not crash when the top-level variable has no units."""

    def test_no_crash(self):
        model = DimensionlessBudgetModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.f)
        assert isinstance(b, Budget)

    def test_units_is_dimensionless(self):
        model = DimensionlessBudgetModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.f)
        assert b.units == "dimensionless"

    def test_total_correct(self):
        model = DimensionlessBudgetModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.f)
        assert abs(b.total - 0.7) < 1e-4  # f_a_min=0.3, f_b_min=0.4 → f=0.7


# ---------------------------------------------------------------------------
# Tests: scaled term does not recurse (Bug 2)
# ---------------------------------------------------------------------------


class ScaledTermModel(Model):
    """Budget constraint has a term f_scale * m_sub where f_scale is a constant.

    m_sub has its own budget constraint. The f_scale * m_sub node must NOT
    have children — recursing into m_sub's budget would produce sub-totals
    that don't match the scaled node value.
    """

    def setup(self):  # pylint: disable=attribute-defined-outside-init
        f_scale = Variable("f_scale", 0.5, "-", "scaling factor")
        self.m_sub = Variable("m_sub", "kg", "sub mass")
        m_sub_min = Variable("m_sub_min", 10, "kg")
        self.m = Variable("m", "kg", "total mass")
        self.cost = self.m
        return [
            self.m >= f_scale * self.m_sub,
            self.m_sub >= m_sub_min,
        ]


class TestScaledTermNoRecursion:
    """Scaled terms (f * m_sub) must be leaf nodes, not recursed into."""

    def test_scaled_node_has_no_children(self):
        model = ScaledTermModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        assert len(b.children) == 1
        scaled_node = b.children[0]
        assert scaled_node.children == []

    def test_scaled_node_value_correct(self):
        """Node value is 0.5 * 10 = 5 kg, not 10 kg."""
        model = ScaledTermModel()
        sol, _ = solve(model)
        b = build_budget(sol, model, model.m)
        scaled_node = b.children[0]
        assert scaled_node.value == pytest.approx(5.0, rel=1e-4)
