"Tests for the TOML printer (AST → expression strings, Model → TOML)."

import pytest

from gpkit import Model, Variable, VectorVariable
from gpkit.ast_nodes import ConstNode, ExprNode, VarNode
from gpkit.toml import load_toml
from gpkit.toml._printer import _ref_to_name, ast_to_expr, to_toml
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
        assert _ref_to_name("Aircraft0.Wing0.S|ft²") == "S"

    def test_vector_element(self):
        assert _ref_to_name("d[0]#3|ft") == "d[0]"

    def test_lineage_vector_element(self):
        assert _ref_to_name("wing0.d[0]#3|ft") == "d[0]"

    def test_no_suffix(self):
        assert _ref_to_name("x") == "x"
