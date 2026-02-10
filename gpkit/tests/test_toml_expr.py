"Tests for the TOML expression parser."

import pytest

from gpkit import Variable, VectorVariable
from gpkit.constraints.single_equation import SingleEquationConstraint
from gpkit.nomials.math import Monomial, Posynomial
from gpkit.toml._expr import (
    TomlExpressionError,
    eval_expr,
    parse_constraint,
    parse_objective,
)

# ---------------------------------------------------------------------------
# Basic arithmetic (pure numbers, no gpkit objects)
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_integer(self):
        assert eval_expr("42", {}) == 42

    def test_float(self):
        assert eval_expr("3.14", {}) == pytest.approx(3.14)

    def test_scientific_notation(self):
        assert eval_expr("1.78e-5", {}) == pytest.approx(1.78e-5)

    def test_negative_number(self):
        assert eval_expr("-2", {}) == -2

    def test_addition(self):
        assert eval_expr("2 + 3", {}) == 5

    def test_subtraction(self):
        assert eval_expr("10 - 4", {}) == 6

    def test_multiplication(self):
        assert eval_expr("3 * 7", {}) == 21

    def test_division(self):
        assert eval_expr("10 / 4", {}) == pytest.approx(2.5)

    def test_power(self):
        assert eval_expr("2**3", {}) == 8

    def test_precedence(self):
        assert eval_expr("2 + 3 * 4", {}) == 14

    def test_parentheses(self):
        assert eval_expr("(2 + 3) * 4", {}) == 20


# ---------------------------------------------------------------------------
# Variable references and gpkit types
# ---------------------------------------------------------------------------


class TestVariables:
    def test_variable_lookup(self):
        x = Variable("x")
        result = eval_expr("x", {"x": x})
        assert result is x

    def test_variable_times_constant(self):
        x = Variable("x")
        result = eval_expr("2*x", {"x": x})
        assert isinstance(result, Monomial)
        assert result.c == pytest.approx(2.0)
        assert dict(result.exp) == {x.key: 1}

    def test_monomial_product(self):
        h = Variable("h")
        w = Variable("w")
        result = eval_expr("h*w", {"h": h, "w": w})
        assert isinstance(result, Monomial)
        assert result.c == pytest.approx(1.0)
        assert dict(result.exp) == {h.key: 1, w.key: 1}

    def test_posynomial_sum(self):
        x = Variable("x")
        y = Variable("y")
        result = eval_expr("2*x + 3*y", {"x": x, "y": y})
        assert isinstance(result, Posynomial)
        assert len(result.hmap) == 2

    def test_compound_expression(self):
        h = Variable("h")
        w = Variable("w")
        d = Variable("d")
        result = eval_expr("2*h*w + 2*h*d", {"h": h, "w": w, "d": d})
        assert isinstance(result, Posynomial)
        assert len(result.hmap) == 2

    def test_power_variable(self):
        x = Variable("x")
        result = eval_expr("x**2", {"x": x})
        assert isinstance(result, Monomial)
        assert result.c == pytest.approx(1.0)
        assert dict(result.exp) == {x.key: 2}

    def test_fractional_power(self):
        x = Variable("x")
        result = eval_expr("x**0.5", {"x": x})
        assert isinstance(result, Monomial)
        assert dict(result.exp) == {x.key: 0.5}

    def test_reciprocal(self):
        x = Variable("x")
        result = eval_expr("1/x", {"x": x})
        assert isinstance(result, Monomial)
        assert dict(result.exp) == {x.key: -1}

    def test_division_of_variables(self):
        x = Variable("x")
        y = Variable("y")
        result = eval_expr("x/y", {"x": x, "y": y})
        assert isinstance(result, Monomial)
        assert dict(result.exp) == {x.key: 1, y.key: -1}

    def test_unknown_variable_error(self):
        x = Variable("x")
        with pytest.raises(TomlExpressionError, match="Unknown variable 'y'"):
            eval_expr("x + y", {"x": x})

    def test_dimension_as_int(self):
        """Dimensions resolve as plain integers in expressions."""
        assert eval_expr("N - 1", {"N": 6}) == 5

    def test_dimension_times_variable(self):
        dx = Variable("dx")
        result = eval_expr("(N-1)*dx", {"N": 6, "dx": dx})
        assert isinstance(result, Monomial)
        assert result.c == pytest.approx(5.0)
        assert dict(result.exp) == {dx.key: 1}


# ---------------------------------------------------------------------------
# Subscript and slice
# ---------------------------------------------------------------------------


class TestSubscript:
    def test_integer_index(self):
        d = VectorVariable(3, "d")
        result = eval_expr("d[0]", {"d": d})
        assert isinstance(result, Monomial)
        assert result.key.idx == (0,)

    def test_computed_index(self):
        V = VectorVariable(6, "V")
        result = eval_expr("V[N-1]", {"V": V, "N": 6})
        assert isinstance(result, Monomial)
        assert result.key.idx == (5,)

    def test_negative_index(self):
        d = VectorVariable(3, "d")
        result = eval_expr("d[-1]", {"d": d})
        assert isinstance(result, Monomial)
        assert result.key.idx == (2,)

    def test_slice_upper(self):
        V = VectorVariable(6, "V")
        result = eval_expr("V[:-1]", {"V": V})
        assert len(result) == 5
        assert result[0].key.idx == (0,)

    def test_slice_lower(self):
        V = VectorVariable(6, "V")
        result = eval_expr("V[1:]", {"V": V})
        assert len(result) == 5
        assert result[0].key.idx == (1,)

    def test_indexed_product(self):
        d = VectorVariable(3, "d")
        result = eval_expr("d[0]*d[1]*d[2]", {"d": d})
        assert isinstance(result, Monomial)
        assert result.c == pytest.approx(1.0)
        # All three indexed variable keys appear in the exponents
        assert len(result.exp) == 3


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_ge_constraint(self):
        x = Variable("x")
        c = parse_constraint("x >= 1", {"x": x})
        assert isinstance(c, SingleEquationConstraint)
        assert c.oper == ">="

    def test_le_constraint(self):
        x = Variable("x")
        c = parse_constraint("x <= 10", {"x": x})
        assert isinstance(c, SingleEquationConstraint)
        assert c.oper == "<="

    def test_eq_constraint(self):
        x = Variable("x")
        y = Variable("y")
        c = parse_constraint("x == y", {"x": x, "y": y})
        assert isinstance(c, SingleEquationConstraint)
        assert c.oper == "="

    def test_complex_ge(self):
        h = Variable("h", "ft")
        w = Variable("w", "ft")
        d = Variable("d", "ft")
        A = Variable("A_wall", 200, "ft^2")
        ns = {"A_wall": A, "h": h, "w": w, "d": d}
        c = parse_constraint("A_wall >= 2*h*w + 2*h*d", ns)
        assert isinstance(c, SingleEquationConstraint)
        assert c.oper == ">="
        # The right side should be a posynomial with 2 terms
        assert len(c.right.hmap) == 2

    def test_not_a_constraint_error(self):
        x = Variable("x")
        with pytest.raises(TomlExpressionError, match="Expected a constraint"):
            parse_constraint("2*x", {"x": x})

    def test_chained_comparison_error(self):
        x = Variable("x")
        with pytest.raises(TomlExpressionError, match="Chained"):
            parse_constraint("1 <= x <= 10", {"x": x})

    def test_slice_constraint_produces_multiple(self):
        """Slice-based vector constraints expand to element-wise constraints."""
        V = VectorVariable(6, "V")
        c = parse_constraint("V[:-1] >= V[1:]", {"V": V})
        # ArrayConstraint wraps 5 element-wise constraints
        assert len(list(c)) == 5


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


class TestObjective:
    def test_min_passthrough(self):
        x = Variable("x")
        cost = parse_objective("min: x", {"x": x})
        assert cost is x

    def test_min_reciprocal(self):
        h = Variable("h")
        w = Variable("w")
        d = Variable("d")
        ns = {"h": h, "w": w, "d": d}
        cost = parse_objective("min: 1/(h*w*d)", ns)
        assert isinstance(cost, Monomial)
        assert dict(cost.exp) == {h.key: -1, w.key: -1, d.key: -1}

    def test_max_inverts_to_reciprocal(self):
        h = Variable("h")
        w = Variable("w")
        d = Variable("d")
        ns = {"h": h, "w": w, "d": d}
        min_cost = parse_objective("min: h*w*d", ns)
        max_cost = parse_objective("max: h*w*d", ns)
        # min: h*w*d has exponents +1 each
        assert dict(min_cost.exp) == {h.key: 1, w.key: 1, d.key: 1}
        # max: h*w*d → minimize 1/(h*w*d), so exponents are all -1
        assert dict(max_cost.exp) == {h.key: -1, w.key: -1, d.key: -1}

    def test_bad_objective(self):
        with pytest.raises(TomlExpressionError, match="min.*max"):
            parse_objective("h*w*d", {"h": Variable("h")})


# ---------------------------------------------------------------------------
# Safety: reject dangerous constructs
# ---------------------------------------------------------------------------


class TestSafety:
    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('import os')",
            "open('/etc/passwd')",
            "print('hello')",
            "lambda: 1",
            "[x for x in [1,2,3]]",
            "type('A', (), {})()",
            "globals()",
            "locals()",
            "dir()",
            "getattr(x, '__class__')",
            "x.__class__",
            "x.real",
            "{1: 2}",
            "[1, 2, 3]",
            "(x for x in [1])",
            "f'{x}'",
        ],
    )
    def test_rejects_dangerous_expression(self, expr):
        ns = {"x": Variable("x")}
        with pytest.raises((TomlExpressionError, SyntaxError)):
            eval_expr(expr, ns)

    def test_non_numeric_constant_rejected(self):
        with pytest.raises(TomlExpressionError, match="Non-numeric"):
            eval_expr("'hello'", {})

    def test_syntax_error_caught(self):
        with pytest.raises(TomlExpressionError, match="Syntax error"):
            eval_expr("2 +* 3", {})
