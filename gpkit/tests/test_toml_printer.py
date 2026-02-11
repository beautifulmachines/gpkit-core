"Tests for the TOML printer (AST → expression strings, Model → TOML)."

import pytest

from gpkit import Model, Variable, VectorVariable
from gpkit.ast_nodes import ConstNode, ExprNode, VarNode
from gpkit.toml import load_toml
from gpkit.toml._printer import ast_to_expr, to_toml
from gpkit.util.small_scripts import mag

# ---------------------------------------------------------------------------
# AST → expression string
# ---------------------------------------------------------------------------


class TestAstToExpr:
    def test_var_node(self):
        from gpkit.varkey import VarKey

        vk = VarKey(name="x")
        assert ast_to_expr(VarNode(vk)) == "x"

    def test_var_node_with_units(self):
        from gpkit.varkey import VarKey

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
        from gpkit.varkey import VarKey

        x = VarNode(VarKey(name="x"))
        y = VarNode(VarKey(name="y"))
        node = ExprNode("mul", (x, y))
        assert ast_to_expr(node) == "x*y"

    def test_mul_drops_1(self):
        from gpkit.varkey import VarKey

        x = VarNode(VarKey(name="x"))
        node = ExprNode("mul", (ConstNode(1.0), x))
        assert ast_to_expr(node) == "x"

    def test_div(self):
        from gpkit.varkey import VarKey

        x = VarNode(VarKey(name="x"))
        y = VarNode(VarKey(name="y"))
        node = ExprNode("div", (x, y))
        assert ast_to_expr(node) == "x/y"

    def test_pow(self):
        from gpkit.varkey import VarKey

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


# ---------------------------------------------------------------------------
# Round-trip tests: TOML → Model → TOML → Model → solve
# ---------------------------------------------------------------------------


class TestRoundTrip:
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
# Python Model → TOML auto-generation
# ---------------------------------------------------------------------------


class TestAutoGenerate:
    def test_python_model_to_loadable_toml(self):
        """A Python Model produces TOML that loads and solves identically."""
        h = Variable("h", "m", "height")
        w = Variable("w", "m", "width")
        d = Variable("d", "m", "depth")
        A_wall = Variable("A_wall", 200, "m^2", "wall area")
        A_floor = Variable("A_floor", 50, "m^2", "floor area")

        py_model = Model(
            1 / (h * w * d),
            [
                A_wall >= 2 * h * w + 2 * h * d,
                A_floor >= w * d,
                h / w >= 2,
                d / w >= 2,
            ],
        )
        py_sol = py_model.solve(verbosity=0)

        toml_str = to_toml(py_model)
        toml_model = load_toml(toml_str)
        toml_sol = toml_model.solve(verbosity=0)

        assert mag(py_sol["h"]) == pytest.approx(mag(toml_sol["h"]), rel=1e-5)
        assert mag(py_sol["w"]) == pytest.approx(mag(toml_sol["w"]), rel=1e-5)
        assert mag(py_sol["d"]) == pytest.approx(mag(toml_sol["d"]), rel=1e-5)
