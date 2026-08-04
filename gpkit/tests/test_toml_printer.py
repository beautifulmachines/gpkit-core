"Tests for the TOML printer (AST → expression strings, Model → TOML)."

import pytest

from gpkit import Model, Variable, VectorVariable, units
from gpkit.ast_nodes import ConstNode, ExprNode, UnitsNode, VarNode
from gpkit.examples.uav import UAV
from gpkit.toml import load_toml
from gpkit.toml._printer import _ref_to_name, ast_to_expr, to_toml
from gpkit.util.globals import NamedVariables
from gpkit.util.small_scripts import mag
from gpkit.varkey import VarKey

# ---------------------------------------------------------------------------
# AST → expression string
# ---------------------------------------------------------------------------


class TestAstToExpr:
    """AST node rendering to plain expression strings."""

    def test_var_node(self):
        vk = VarKey(name="x")
        assert ast_to_expr(VarNode(vk)) == "x"

    def test_var_node_with_units(self):
        vk = VarKey(name="h", units="ft")
        assert ast_to_expr(VarNode(vk)) == "h"

    def test_const_node(self):
        assert ast_to_expr(ConstNode(2.0)) == "2"
        assert ast_to_expr(ConstNode(3.14)) == "3.14"

    def test_raw_number(self):
        assert ast_to_expr(42) == "42"
        assert ast_to_expr(2.5) == "2.5"

    def test_add(self):
        a = ConstNode(1.0)
        b = ConstNode(2.0)
        node = ExprNode("add", (a, b))
        assert ast_to_expr(node) == "1 + 2"

    def test_mul(self):
        x = VarNode(VarKey(name="x"))
        y = VarNode(VarKey(name="y"))
        node = ExprNode("mul", (x, y))
        assert ast_to_expr(node) == "x*y"

    def test_mul_drops_1(self):
        x = VarNode(VarKey(name="x"))
        node = ExprNode("mul", (ConstNode(1.0), x))
        assert ast_to_expr(node) == "x"

    def test_div(self):
        x = VarNode(VarKey(name="x"))
        y = VarNode(VarKey(name="y"))
        node = ExprNode("div", (x, y))
        assert ast_to_expr(node) == "x/y"

    def test_pow(self):
        x = VarNode(VarKey(name="x"))
        node = ExprNode("pow", (x, 2))
        assert ast_to_expr(node) == "x**2"

    def test_neg(self):
        node = ExprNode("neg", (ConstNode(3.0),))
        assert ast_to_expr(node) == "-3"

    def test_parenthesization(self):
        """Mul of add should parenthesize the add."""
        a = ConstNode(1.0)
        b = ConstNode(2.0)
        c = ConstNode(3.0)
        add_node = ExprNode("add", (a, b))
        mul_node = ExprNode("mul", (add_node, c))
        assert ast_to_expr(mul_node) == "(1 + 2)*3"

    def test_ir_dict(self):
        """IR dicts (from to_ir() JSON) should render correctly."""
        ir = {
            "node": "expr",
            "op": "mul",
            "children": [
                {"node": "var", "ref": "h|ft"},
                {"node": "var", "ref": "w|ft"},
            ],
        }
        assert ast_to_expr(ir) == "h*w"

    def test_units_node(self):
        """UnitsNode (e.g. `1 * units("W")`) must round-trip as a real unit
        literal: dropping it (rendering "1") would silently change a
        round-tripped model's solved values whenever it's combined with an
        operand of different unit scale (see test_units_only_constant_in_constraint)."""
        node = UnitsNode(units("W").hmap.units)
        assert ast_to_expr(node) == "units('W')"

    def test_div_by_units_node_preserves_literal(self):
        """x / (1*units.W) must keep the units() literal, not simplify it
        away — the division may carry a real unit-conversion factor."""
        x = VarNode(VarKey(name="x"))
        node = ExprNode("div", (x, UnitsNode(units("W").hmap.units)))
        assert ast_to_expr(node) == "x/units('W')"


# ---------------------------------------------------------------------------
# to_toml: Python Model → TOML string
# ---------------------------------------------------------------------------


class TestToToml:
    """Python Model to TOML string generation."""

    def test_simple_model(self):
        h = Variable("h", "ft")
        w = Variable("w", "ft")
        d = Variable("d", "ft")
        m = Model(1 / (h * w * d), [d / w >= 2])
        toml_str = to_toml(m)

        assert "[vars]" in toml_str
        assert "[model]" in toml_str
        assert 'objective = "max: h*w*d"' in toml_str
        assert "d/w >= 2" in toml_str

    def test_vector_model(self):
        d = VectorVariable(3, "d", "m", "dimensions")
        A = Variable("A", "m^2")
        m = Model(A, [A >= 2 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])])
        toml_str = to_toml(m)

        assert "[vectors.3]" in toml_str
        assert 'd = ["m", "dimensions"]' in toml_str

    def test_write_to_file(self, tmp_path):
        x = Variable("x")
        m = Model(x, [x >= 1])
        path = tmp_path / "test.toml"
        to_toml(m, path=path)

        assert path.exists()
        content = path.read_text()
        assert "[model]" in content

    def test_units_only_constant_in_constraint(self):
        """Regression test: a bare `1 * units(...)` normalization constant
        (e.g. used to non-dimensionalize before a fractional power, as in
        gpkit.examples.uav.Engine) produced a UnitsNode in the AST that
        crashed to_toml() with 'Cannot render AST node: UnitsNode'.

        P is declared in kW while P_ref is exactly 1 W, so the (P/P_ref)
        normalization carries a real x1000 conversion factor before the
        fractional power — dropping it (rendering "1") would silently
        change the solved values on reload instead of just losing display
        units, since P**0.803 has different units *and* magnitude than
        (P/P_ref)**0.803.
        """
        P = Variable("P", "kW", "power")
        P_min = Variable("P_min", 3, "kW")
        P_ref = 1 * units("W")
        W = Variable("W", "N", "weight")
        m = Model(
            W,
            [
                W >= (P / P_ref) ** 0.803 * Variable("W_coeff", 1, "N"),
                P >= P_min,
            ],
        )
        sol1 = m.solve(verbosity=0)

        toml_str = to_toml(m)
        assert "UnitsNode" not in toml_str
        assert "units('W')" in toml_str

        toml_model = load_toml(toml_str)
        sol2 = toml_model.solve(verbosity=0)
        assert mag(sol1["W"]) == pytest.approx(mag(sol2["W"]), rel=1e-5)

    def test_python_model_to_loadable_toml(self):
        """A Python Model produces TOML that loads and solves identically."""
        x = Variable("x", "m", "length")
        y = Variable("y", "m", "width")
        a_min = Variable("A_min", 100, "m^2", "minimum area")

        py_model = Model(x + y, [x * y >= a_min])
        py_sol = py_model.solve(verbosity=0)

        toml_str = to_toml(py_model)
        toml_model = load_toml(toml_str)
        toml_sol = toml_model.solve(verbosity=0)

        assert mag(py_sol["x"]) == pytest.approx(mag(toml_sol["x"]), rel=1e-5)
        assert mag(py_sol["y"]) == pytest.approx(mag(toml_sol["y"]), rel=1e-5)


# ---------------------------------------------------------------------------
# Round-trip tests: TOML → Model → TOML → Model → solve
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip: TOML file → Model → TOML string → Model → solve."""

    def _round_trip(self, toml_path):
        """Load a TOML, solve, generate TOML, load again, solve, compare."""
        m1 = load_toml(toml_path)
        sol1 = m1.solve(verbosity=0)

        toml_str = to_toml(m1)
        m2 = load_toml(toml_str)
        sol2 = m2.solve(verbosity=0)

        return sol1, sol2

    def test_simple_box_round_trip(self):
        sol1, sol2 = self._round_trip("docs/source/examples/toml/simple_box.toml")
        assert mag(sol1["h"]) == pytest.approx(mag(sol2["h"]), rel=1e-5)
        assert mag(sol1["w"]) == pytest.approx(mag(sol2["w"]), rel=1e-5)
        assert mag(sol1["d"]) == pytest.approx(mag(sol2["d"]), rel=1e-5)

    def test_water_tank_round_trip(self):
        sol1, sol2 = self._round_trip("docs/source/examples/toml/water_tank.toml")
        assert mag(sol1["A"]) == pytest.approx(mag(sol2["A"]), rel=1e-5)

    def test_simpleflight_round_trip(self):
        sol1, sol2 = self._round_trip("docs/source/examples/toml/simpleflight.toml")
        assert mag(sol1["D"]) == pytest.approx(mag(sol2["D"]), rel=1e-5)
        assert mag(sol1["W"]) == pytest.approx(mag(sol2["W"]), rel=1e-5)

    def test_uav_round_trip(self):
        """gpkit.examples.uav.UAV: a modular multi-model example whose
        Mission submodel instantiates AircraftPerf (and its own children,
        FlightState/WingAero/PropulsionPerf) three times over — once per
        named flight condition (Outbound, Return, SprintCondition) — and
        whose Aircraft model references child variables like
        self.engine.W across model boundaries. Regression test for the
        combination of every multi-model to_toml/load_toml fix above."""
        m1 = UAV()
        sol1 = m1.solve(verbosity=0)

        toml_str = to_toml(m1)
        m2 = load_toml(toml_str)
        sol2 = m2.solve(verbosity=0)

        assert mag(sol1.cost) == pytest.approx(mag(sol2.cost), rel=1e-3)


# ---------------------------------------------------------------------------
# Multi-instance models: a class instantiated more than once in the tree,
# and constraints that reference a *different* model's variable.
# ---------------------------------------------------------------------------


class TestMultiInstanceModels:
    """to_toml/load_toml when the same Model class is instantiated more
    than once (previously: duplicate [models.X] TOML headers, or silently
    merged variables from different instances) and when a constraint
    references a variable owned by a different model than the one it lives
    in (previously: printed as a bare, ambiguous name)."""

    def _repeated_leg_model(self):
        # Model instance numbering is a global counter keyed by (lineage,
        # class name); reset so each call gets deterministic "Leg"/"Leg1"
        # ids regardless of what other tests constructed earlier.
        NamedVariables.reset_modelnumbers()

        class Leg(Model):
            def setup(self):
                self.x = Variable("x", "m")
                xmin = Variable("xmin", 1, "m")
                return [self.x >= xmin]

        class Top(Model):
            def setup(self):
                self.a = Leg()
                self.b = Leg()
                self.cost = self.a.x + self.b.x
                return [self.a, self.b]

        return Top()

    def test_repeated_class_gets_disambiguated_sections(self):
        toml_str = to_toml(self._repeated_leg_model())
        assert "[models.Leg]" in toml_str
        assert "[models.Leg1]" in toml_str

    def test_repeated_class_round_trip(self):
        m1 = self._repeated_leg_model()
        sol1 = m1.solve(verbosity=0)

        toml_str = to_toml(m1)
        m2 = load_toml(toml_str)
        sol2 = m2.solve(verbosity=0)

        assert mag(sol1.cost) == pytest.approx(mag(sol2.cost), rel=1e-5)

    def _cross_referencing_aircraft_model(self):
        NamedVariables.reset_modelnumbers()

        class Engine(Model):
            def setup(self):
                self.W = Variable("W", "N", "engine weight")
                w_min = Variable("Wmin", 5, "N")
                return [self.W >= w_min]

        class Wing(Model):
            def setup(self):
                self.W = Variable("W", "N", "wing weight")
                w_min = Variable("Wmin", 3, "N")
                return [self.W >= w_min]

        class Aircraft(Model):
            def setup(self):
                self.engine = Engine()
                self.wing = Wing()
                self.W_total = Variable("W_total", "N")
                self.cost = self.W_total
                return [
                    self.engine,
                    self.wing,
                    self.W_total >= self.engine.W + self.wing.W,
                ]

        return Aircraft()

    def test_cross_model_reference_is_qualified(self):
        """Aircraft's constraint references self.engine.W and self.wing.W —
        both named "W", like Aircraft's own W_total is unrelated to either.
        Each must print qualified ("Engine.W", "Wing.W"), not bare "W"."""
        toml_str = to_toml(self._cross_referencing_aircraft_model())
        aircraft_section = toml_str.split("[models.Aircraft]")[1]
        assert "Engine.W" in aircraft_section
        assert "Wing.W" in aircraft_section

    def test_cross_model_reference_round_trip(self):
        m1 = self._cross_referencing_aircraft_model()
        sol1 = m1.solve(verbosity=0)

        toml_str = to_toml(m1)
        m2 = load_toml(toml_str)
        sol2 = m2.solve(verbosity=0)

        assert mag(sol1.cost) == pytest.approx(mag(sol2.cost), rel=1e-5)


# ---------------------------------------------------------------------------
# _ref_to_name: lineage and suffix stripping
# ---------------------------------------------------------------------------


class TestRefToName:
    """_ref_to_name extracts bare variable names from IR ref strings."""

    def test_bare_name(self):
        assert _ref_to_name("S") == "S"

    def test_units_stripped(self):
        assert _ref_to_name("S|ft²") == "S"

    def test_lineage_stripped(self):
        assert _ref_to_name("wing0.S|ft²") == "S"

    def test_deep_lineage(self):
        assert _ref_to_name("Aircraft.Wing.S|ft²") == "S"

    def test_vector_element(self):
        assert _ref_to_name("d[0]#3|ft") == "d[0]"

    def test_lineage_vector_element(self):
        assert _ref_to_name("wing0.d[0]#3|ft") == "d[0]"

    def test_no_suffix(self):
        assert _ref_to_name("x") == "x"
