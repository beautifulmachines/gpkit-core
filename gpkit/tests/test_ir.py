"""Tests for the IR (Intermediate Representation) infrastructure.

Phase 1: AST node dataclasses and VarKey.var_ref
"""

import sys

import pytest

from gpkit import SignomialsEnabled, Variable, VarKey, VectorVariable
from gpkit.ast_nodes import ConstNode, ExprNode, VarNode, to_ast


class TestASTNodes:
    """Tests for AST node dataclass hierarchy."""

    def test_varnode_from_variable(self):
        x = Variable("x")
        node = to_ast(x)
        assert isinstance(node, VarNode)
        assert node.varkey is x.key

    def test_varnode_str_without(self):
        x = Variable("x")
        node = VarNode(x.key)
        assert node.str_without() == "x"

    def test_constnode(self):
        node = ConstNode(3.14)
        assert node.value == 3.14
        assert node.str_without() == "3.14"

    def test_exprnode_add(self):
        x = Variable("x")
        y = Variable("y")
        result = x + y
        assert isinstance(result.ast, ExprNode)
        assert result.ast.op == "add"
        assert len(result.ast.children) == 2

    def test_exprnode_mul(self):
        x = Variable("x")
        y = Variable("y")
        result = x * y
        assert isinstance(result.ast, ExprNode)
        assert result.ast.op == "mul"

    def test_exprnode_div(self):
        x = Variable("x")
        result = x / 2
        assert isinstance(result.ast, ExprNode)
        assert result.ast.op == "div"

    def test_exprnode_pow(self):
        x = Variable("x")
        result = x**3
        assert isinstance(result.ast, ExprNode)
        assert result.ast.op == "pow"
        assert result.ast.children[1] == 3

    def test_exprnode_neg(self):
        x = Variable("x")
        with SignomialsEnabled():
            result = -x
        assert isinstance(result.ast, ExprNode)
        assert result.ast.op == "neg"
        assert len(result.ast.children) == 1

    def test_signomial_pow_records_correct_exponent(self):
        """Regression: Signomial.__pow__ used to record expo=0 (bug fix)."""
        x = Variable("x")
        p = x**3 + x + 5
        p2 = p**2
        assert isinstance(p2.ast, ExprNode)
        assert p2.ast.op == "pow"
        assert p2.ast.children[1] == 2

    def test_signomial_pow_str(self):
        """Verify string output of Signomial.__pow__ shows correct exponent."""
        x = Variable("x")
        p = x**3 + x + 5
        p2_str = str(p**2)
        assert "^0" not in p2_str
        if sys.platform[:3] != "win":
            assert "\u00b2" in p2_str  # Unicode superscript ²
            assert p2_str == "(x\u00b3 + x + 5)\u00b2"

    def test_to_ast_passthrough_for_numbers(self):
        assert to_ast(42) == 42
        assert to_ast(3.14) == 3.14

    def test_to_ast_returns_existing_ast(self):
        x = Variable("x")
        y = Variable("y")
        s = x + y
        assert to_ast(s) is s.ast

    def test_to_ast_idempotent(self):
        x = Variable("x")
        node = VarNode(x.key)
        assert to_ast(node) is node

    def test_nested_ast(self):
        """Compound expressions produce nested ExprNode trees."""
        x = Variable("x")
        y = Variable("y")
        result = (x + y) * x
        assert isinstance(result.ast, ExprNode)
        assert result.ast.op == "mul"
        left = result.ast.children[0]
        assert isinstance(left, ExprNode)
        assert left.op == "add"

    def test_frozen(self):
        """AST nodes are immutable."""
        node = ExprNode("add", (ConstNode(1), ConstNode(2)))
        with pytest.raises(AttributeError):
            node.op = "mul"


class TestVarRef:
    """Tests for VarKey.var_ref property."""

    def test_plain_variable(self):
        vk = VarKey("x")
        assert vk.var_ref == "x"

    def test_with_lineage(self):
        vk = VarKey("S", lineage=(("Aircraft", 0), ("Wing", 0)))
        assert vk.var_ref == "Aircraft0.Wing0.S"

    def test_with_lineage_nonzero_num(self):
        vk = VarKey("S", lineage=(("Aircraft", 0), ("Wing", 1)))
        assert vk.var_ref == "Aircraft0.Wing1.S"

    def test_indexed(self):
        x = VectorVariable(3, "x")
        # VectorVariable creates element VarKeys with idx
        assert x[0].key.var_ref == "x[0]"
        assert x[2].key.var_ref == "x[2]"

    def test_lineage_and_index(self):
        vk = VarKey("c_l", lineage=(("Wing", 0),), idx=(1,), shape=(3,))
        assert vk.var_ref == "Wing0.c_l[1]"
