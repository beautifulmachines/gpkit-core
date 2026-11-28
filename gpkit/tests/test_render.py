"""Tests for rendering expressions and constraints as text, LaTeX and TOML.

Every form comes from one traversal, so these pin where the forms used to
disagree: numbers, parentheses, and nomials that were never written as one
expression.
"""

from gpkit import Variable, VectorVariable, units
from gpkit.ast_nodes import VarNode, ast_from_ir, to_ast
from gpkit.display.render import LATEX, LATEX_ALIGNED, TEXT, render
from gpkit.toml._printer import constraint_to_expr

x = Variable("x")
y = Variable("y")
m = Variable("m", "m")


class TestEntryPoint:
    "render takes any nomial, array or constraint; str() is render with defaults."

    def test_nomial(self):
        assert render(x + 2 * y, TEXT) == "x + 2⋅y"

    def test_units_suffix_unless_excluded(self):
        assert render(2 * m, TEXT) == "2⋅m [m]"
        assert render(2 * m, TEXT, {"units"}) == "2⋅m"

    def test_constraint(self):
        assert render(x >= 2 * y, LATEX) == r"x \geq 2 y"

    def test_aligned_latex_constraint(self):
        assert render(x >= y, LATEX_ALIGNED) == r"x &\geq y"

    def test_vector(self):
        assert render(VectorVariable(3, "v"), TEXT) == "v[:]"

    def test_str_is_render(self):
        c = x >= 2 * y
        assert str(c) == render(c, TEXT)
        assert str(2 * m) == render(2 * m, TEXT)

    def test_anything_else_is_its_str(self):
        "A custom constraint class knows nothing of rendering; it shows as its str."
        assert render(object, TEXT) == str(object)


class TestVarNode:
    "A variable in a tree is its ref, so a live tree and a document's are the same."

    def test_holds_the_ref(self):
        assert to_ast(x) == VarNode(x.key.ref)

    def test_from_ir_needs_no_registry(self):
        assert ast_from_ir({"node": "var", "ref": "x"}) == VarNode("x")


class TestCancelled:
    "A written form naming a variable the math lacks is not used: x/x has no x."

    expr = 3 * m / m * y

    def test_renders_from_its_terms(self):
        assert render(self.expr, TEXT, {"units"}) == "3⋅y"

    def test_ir_side_has_no_ast(self):
        "Every ref in the model IR's ASTs resolves in its variables."
        assert "ast" not in (x >= self.expr).to_ir()["right"]


def _bare(ref):
    "x and y have no lineage or units, so each ref is the name TOML writes."
    return ref


def _terms(text):
    "The terms of a sum, whatever order they print in."
    return set(text.split(" + "))


class TestNumbers:
    "Text and LaTeX show the same digits; TOML shows the exact value."

    def test_str_and_latex_show_the_same_coefficient(self):
        c = x >= 2.71828 * y
        assert render(c, TEXT) == "x ≥ 2.718⋅y"
        assert render(c, LATEX) == r"x \geq 2.718 y"

    def test_large_coefficient_uses_latex_scientific_notation(self):
        assert render(1234567 * x, LATEX) == r"1.235 \times 10^{6} x"

    def test_toml_coefficient_is_exact(self):
        c = (x >= 2.71828 * y).to_ir()
        assert constraint_to_expr(c, _bare) == "x >= 2.71828*y"


class TestParentheses:
    "A child is grouped exactly when its operator binds looser than its slot."

    def test_power_of_a_power_groups_the_base(self):
        assert render((x**2) ** 0.5, TEXT) == "(x²)^0.5"

    def test_product_in_a_denominator_groups(self):
        assert render(x / (y * m), TEXT, {"units"}) == "x/(y⋅m)"


class TestUnitLiterals:
    def test_latex_unit_literal_is_set_once(self):
        assert render(m / (2 * units.m), LATEX, {"units"}) == r"\frac{m}{2 \mathrm{m}}"


class TestUnwritten:
    "A nomial with no written form renders from its terms, the same in every form."

    derived = (2.71828 * x**0.33333 + y).sub({y: 3})

    def test_str_matches_written_precision(self):
        assert _terms(render(self.derived, TEXT)) == {"2.718⋅x^0.3333", "3"}

    def test_latex_spaces_coefficient_from_variable(self):
        assert _terms(render(self.derived, LATEX)) == {"2.718 x^{0.3333}", "3"}

    def test_toml_renders_it(self):
        c = x >= self.derived
        left, right = constraint_to_expr(c.to_ir(), _bare).split(" >= ")
        assert left == "x"
        assert _terms(right) == {"2.71828*x**0.33333", "3"}


class TestArrayLiterals:
    "A list operand is a numeric array, so the expression around it is unwritten."

    def test_vector_constraint_with_a_list_serializes(self):
        a = VectorVariable(3, "a")
        b = VectorVariable(3, "b")
        assert (a >= b * [1, 2, 3]).to_ir()["right"] == {}


class TestLeaves:
    "A variable or constant was written, so it carries an AST into the IR."

    def test_variable_side_has_ast(self):
        ir = (x >= y).to_ir()
        assert ir["left"]["ast"] == {"node": "var", "ref": x.key.ref}
        assert ir["right"]["ast"] == {"node": "var", "ref": y.key.ref}

    def test_constant_side_has_ast(self):
        ir = (x >= 3).to_ir()
        assert ir["right"]["ast"] == {"node": "const", "value": 3.0}
