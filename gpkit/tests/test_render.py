"""Tests for rendering expressions and constraints as text, LaTeX and TOML.

Every form comes from one traversal, so these pin where the forms used to
disagree: numbers, parentheses, and nomials that were never written as one
expression.
"""

from gpkit import Variable, VectorVariable, units
from gpkit.toml._printer import constraint_to_expr

x = Variable("x")
y = Variable("y")
m = Variable("m", "m")


def _terms(text):
    "The terms of a sum, whatever order they print in."
    return set(text.split(" + "))


class TestNumbers:
    "Text and LaTeX show the same digits; TOML shows the exact value."

    def test_str_and_latex_show_the_same_coefficient(self):
        c = x >= 2.71828 * y
        assert c.str_without({"units"}) == "x ≥ 2.718⋅y"
        assert c.latex({"units"}) == r"x \geq 2.718 y"

    def test_large_coefficient_uses_latex_scientific_notation(self):
        assert (1234567 * x).latex({"units"}) == r"1.235 \times 10^{6} x"

    def test_toml_coefficient_is_exact(self):
        assert constraint_to_expr((x >= 2.71828 * y).to_ir()) == "x >= 2.71828*y"


class TestParentheses:
    "A child is grouped exactly when its operator binds looser than its slot."

    def test_power_of_a_power_groups_the_base(self):
        assert ((x**2) ** 0.5).str_without({"units"}) == "(x²)^0.5"

    def test_product_in_a_denominator_groups(self):
        assert (x / (x * y)).str_without({"units"}) == "x/(x⋅y)"


class TestUnitLiterals:
    def test_latex_unit_literal_is_set_once(self):
        assert (m / (2 * units.m)).latex({"units"}) == r"\frac{m}{2 \mathrm{m}}"


class TestUnwritten:
    "A nomial with no written form renders from its terms, the same in every form."

    derived = (2.71828 * x**0.33333 + y).sub({y: 3})

    def test_str_matches_written_precision(self):
        assert _terms(self.derived.str_without({"units"})) == {"2.718⋅x^0.3333", "3"}

    def test_latex_spaces_coefficient_from_variable(self):
        assert _terms(self.derived.latex({"units"})) == {"2.718 x^{0.3333}", "3"}

    def test_toml_renders_it(self):
        c = x >= self.derived
        left, right = constraint_to_expr(c.to_ir()).split(" >= ")
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
