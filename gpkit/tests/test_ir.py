"""Tests for the IR (Intermediate Representation) infrastructure."""

# pylint: disable=invalid-name,too-many-lines

import json

import pytest

from gpkit import Model, SignomialsEnabled, Variable, VarKey, VectorVariable
from gpkit.ast_nodes import ConstNode, ExprNode, VarNode, ast_from_ir, to_ast
from gpkit.constraints.array import ArrayConstraint
from gpkit.ir import from_json, to_json
from gpkit.nomials.map import NomialMap
from gpkit.nomials.math import (
    Monomial,
    MonomialEquality,
    Posynomial,
    PosynomialInequality,
    Signomial,
    SignomialInequality,
    SingleSignomialEquality,
    constraint_from_ir,
    nomial_from_ir,
)
from gpkit.units import qty
from gpkit.util.small_classes import EMPTY_HV, HashVector

# ── Shared test model definitions ────────────────────────────────────


class Wing(Model):
    """simple wing with weight proportional to area"""

    def setup(self):
        S = Variable("S", 100, label="wing area")
        W = Variable("W", label="wing weight")
        self.cost = W
        return [W >= S * 0.1]


class Aircraft(Model):
    """Single-wing aircraft for IR nesting tests."""

    def setup(self):
        W = Variable("W", label="total weight")
        wing = Wing()
        self.cost = W
        return [W >= wing.cost * 1.2, wing]


class Sub(Model):
    """Minimal sub-model with one free variable."""

    def setup(self):
        m = Variable("m")
        self.cost = m
        return [m >= 1]


class Widget(Model):
    """Model composing two Sub instances."""

    def setup(self):
        s1 = Sub()
        s2 = Sub()
        self.cost = s1.cost + s2.cost
        return [s1, s2]


class Spar(Model):
    """Simple spar with a thickness variable."""

    def setup(self):
        t = Variable("t", label="spar thickness")
        self.cost = t
        return [t >= 0.01]


class SparredWing(Model):
    """Wing sub-model containing a Spar child."""

    def setup(self):
        S = Variable("S", label="wing area")
        spar = Spar()
        self.cost = S + spar.cost
        return [S >= 1, spar]


class SparredAircraft(Model):
    """Aircraft with two-level nesting (wing + spar)."""

    def setup(self):
        W = Variable("W", label="total weight")
        wing = SparredWing()
        self.cost = W
        return [W >= wing.cost * 1.2, wing]


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


class TestVarKeyIR:
    """Tests for VarKey.to_ir() / VarKey.from_ir() round-trip."""

    def test_plain(self):
        vk = VarKey("x")
        ir = vk.to_ir()
        assert ir["name"] == "x"
        assert "lineage" not in ir
        assert "units" not in ir
        vk2 = VarKey.from_ir(ir)
        assert vk2 == vk

    def test_with_units(self):
        vk = VarKey("S", unitrepr="m^2")
        ir = vk.to_ir()
        assert ir["units"] == "m^2"
        vk2 = VarKey.from_ir(ir)
        assert vk2 == vk
        assert vk2.unitrepr == "m^2"

    def test_with_lineage(self):
        vk = VarKey("S", lineage=(("Aircraft", 0), ("Wing", 0)))
        ir = vk.to_ir()
        assert ir["lineage"] == [["Aircraft", 0], ["Wing", 0]]
        vk2 = VarKey.from_ir(ir)
        assert vk2 == vk
        assert vk2.lineage == (("Aircraft", 0), ("Wing", 0))

    def test_with_idx(self):
        vk = VarKey("c_l", idx=(1,), shape=(3,))
        ir = vk.to_ir()
        assert ir["idx"] == [1]
        assert ir["shape"] == [3]
        vk2 = VarKey.from_ir(ir)
        assert vk2 == vk

    def test_with_label(self):
        vk = VarKey("W", label="total weight")
        ir = vk.to_ir()
        assert ir["label"] == "total weight"
        vk2 = VarKey.from_ir(ir)
        assert vk2.label == "total weight"

    def test_json_serializable(self):
        vk = VarKey("S", lineage=(("Wing", 0),), unitrepr="m^2", label="area")
        ir = vk.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        vk2 = VarKey.from_ir(ir2)
        assert vk2 == vk

    def test_dimensionless_omits_units(self):
        vk = VarKey("x")
        ir = vk.to_ir()
        assert "units" not in ir


class TestASTNodeIR:
    """Tests for AST node to_ir() / ast_from_ir() round-trip."""

    def _var_registry(self, *variables):
        "Build var_registry from Variables."
        return {v.key.var_ref: v.key for v in variables}

    def test_varnode(self):
        x = Variable("x")
        node = VarNode(x.key)
        ir = node.to_ir()
        assert ir == {"node": "var", "ref": "x"}
        registry = self._var_registry(x)
        node2 = ast_from_ir(ir, registry)
        assert isinstance(node2, VarNode)
        assert node2.varkey == x.key

    def test_constnode(self):
        node = ConstNode(3.14)
        ir = node.to_ir()
        assert ir == {"node": "const", "value": 3.14}
        node2 = ast_from_ir(ir, {})
        assert isinstance(node2, ConstNode)
        assert node2.value == 3.14

    def test_exprnode_add(self):
        x = Variable("x")
        y = Variable("y")
        result = x + y
        ir = result.ast.to_ir()
        assert ir["node"] == "expr"
        assert ir["op"] == "add"
        assert len(ir["children"]) == 2
        registry = self._var_registry(x, y)
        ast2 = ast_from_ir(ir, registry)
        assert isinstance(ast2, ExprNode)
        assert ast2.op == "add"

    def test_exprnode_pow_preserves_exponent(self):
        x = Variable("x")
        result = x**3
        ir = result.ast.to_ir()
        assert ir["op"] == "pow"
        # exponent is a raw number, not a const node
        assert ir["children"][1] == 3
        registry = self._var_registry(x)
        ast2 = ast_from_ir(ir, registry)
        assert ast2.children[1] == 3

    def test_nested_ast_roundtrip(self):
        x = Variable("x")
        y = Variable("y")
        result = (x + y) * x
        ir = result.ast.to_ir()
        registry = self._var_registry(x, y)
        ast2 = ast_from_ir(ir, registry)
        assert ast2.op == "mul"
        assert ast2.children[0].op == "add"

    def test_json_roundtrip(self):
        x = Variable("x")
        y = Variable("y")
        result = x * y + x**2
        ir = result.ast.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        registry = self._var_registry(x, y)
        ast2 = ast_from_ir(ir2, registry)
        assert ast2.str_without() == result.ast.str_without()


class TestNomialMapIR:
    """Tests for NomialMap.to_ir() / NomialMap.from_ir() round-trip."""

    def test_monomial(self):
        """Single-term NomialMap (monomial)."""
        x = Variable("x")
        hmap = NomialMap({HashVector({x.key: 2}): 3.0})
        ir = hmap.to_ir()
        assert len(ir["terms"]) == 1
        assert ir["terms"][0]["coeff"] == 3.0
        assert ir["terms"][0]["exps"]["x"] == 2
        registry = {x.key.var_ref: x.key}
        hmap2 = NomialMap.from_ir(ir, registry)
        assert len(hmap2) == 1
        ((exp, coeff),) = hmap2.items()
        assert coeff == 3.0
        assert exp[x.key] == 2

    def test_posynomial(self):
        """Multi-term NomialMap (posynomial)."""
        x = Variable("x")
        y = Variable("y")
        hmap = NomialMap(
            {
                HashVector({x.key: 1}): 2.0,
                HashVector({y.key: 1}): 3.0,
            }
        )
        ir = hmap.to_ir()
        assert len(ir["terms"]) == 2
        registry = {x.key.var_ref: x.key, y.key.var_ref: y.key}
        hmap2 = NomialMap.from_ir(ir, registry)
        assert len(hmap2) == 2

    def test_constant_term(self):
        """Constant term uses EMPTY_HV, no exps key."""
        hmap = NomialMap({EMPTY_HV: 5.0})
        ir = hmap.to_ir()
        assert len(ir["terms"]) == 1
        assert "exps" not in ir["terms"][0]
        assert ir["terms"][0]["coeff"] == 5.0
        hmap2 = NomialMap.from_ir(ir, {})
        assert hmap2[EMPTY_HV] == 5.0

    def test_with_units(self):
        """Units survive round-trip."""
        x = Variable("x", unitrepr="m")
        hmap = NomialMap({HashVector({x.key: 1}): 1.0})
        hmap.units = qty("m")
        ir = hmap.to_ir()
        assert "units" in ir
        registry = {x.key.var_ref: x.key}
        hmap2 = NomialMap.from_ir(ir, registry)
        assert hmap2.units is not None

    def test_json_serializable(self):
        """NomialMap IR is JSON-serializable."""
        x = Variable("x")
        hmap = NomialMap({HashVector({x.key: 1}): 2.5})
        ir = hmap.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        registry = {x.key.var_ref: x.key}
        hmap2 = NomialMap.from_ir(ir2, registry)
        assert len(hmap2) == 1


class TestNomialIR:
    """Tests for Signomial.to_ir() / Signomial.from_ir() round-trip."""

    def _registry(self, nomial):
        "Build var_registry from a nomial's varkeys."
        return {vk.var_ref: vk for vk in nomial.vks}

    def test_monomial_roundtrip(self):
        x = Variable("x")
        m = 2 * x**3
        ir = m.to_ir()
        assert ir["type"] == "Monomial"
        m2 = Signomial.from_ir(ir, self._registry(m))
        assert isinstance(m2, Monomial)
        assert len(m2.hmap) == 1
        ((exp, coeff),) = m2.hmap.items()
        assert coeff == 2.0
        assert exp[x.key] == 3

    def test_posynomial_roundtrip(self):
        x = Variable("x")
        y = Variable("y")
        p = x + 2 * y
        ir = p.to_ir()
        assert ir["type"] == "Posynomial"
        p2 = Signomial.from_ir(ir, self._registry(p))
        assert isinstance(p2, Posynomial)
        assert len(p2.hmap) == 2

    def test_signomial_roundtrip(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            s = x - y
        ir = s.to_ir()
        assert ir["type"] == "Signomial"
        s2 = Signomial.from_ir(ir, self._registry(s))
        assert isinstance(s2, Signomial)
        assert s2.any_nonpositive_cs

    def test_ast_survives_roundtrip(self):
        x = Variable("x")
        y = Variable("y")
        p = x + 2 * y
        ir = p.to_ir()
        assert "ast" in ir
        p2 = Signomial.from_ir(ir, self._registry(p))
        assert p2.ast is not None
        assert p2.ast.str_without() == p.ast.str_without()

    def test_units_survive_roundtrip(self):
        x = Variable("x", unitrepr="m")
        y = Variable("y", unitrepr="m")
        p = x + y
        ir = p.to_ir()
        assert "units" in ir
        p2 = Signomial.from_ir(ir, self._registry(p))
        assert p2.units is not None

    def test_json_roundtrip(self):
        x = Variable("x")
        y = Variable("y")
        p = x + 2 * y
        ir = p.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        p2 = Signomial.from_ir(ir2, self._registry(p))
        assert len(p2.hmap) == len(p.hmap)

    def test_constant_nomial(self):
        """A nomial with only a constant term."""
        m = Signomial(5.0)
        ir = m.to_ir()
        assert ir["type"] == "Monomial"
        m2 = Signomial.from_ir(ir, {})
        assert m2.cs[0] == 5.0


class TestNomialFromIR:
    """Tests for the nomial_from_ir dispatcher function."""

    def _registry(self, nomial):
        "Build var_registry from a nomial's varkeys."
        return {vk.var_ref: vk for vk in nomial.vks}

    def test_monomial(self):
        x = Variable("x")
        m = 2 * x**3
        ir = m.to_ir()
        m2 = nomial_from_ir(ir, self._registry(m))
        assert isinstance(m2, Monomial)
        assert len(m2.hmap) == 1

    def test_posynomial(self):
        x = Variable("x")
        y = Variable("y")
        p = x + 2 * y
        ir = p.to_ir()
        p2 = nomial_from_ir(ir, self._registry(p))
        assert isinstance(p2, Posynomial)
        assert len(p2.hmap) == 2

    def test_signomial(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            s = x - y
        ir = s.to_ir()
        s2 = nomial_from_ir(ir, self._registry(s))
        assert isinstance(s2, Signomial)
        assert s2.any_nonpositive_cs


class TestConstraintIR:
    """Tests for constraint to_ir() / from_ir() round-trips."""

    def _registry(self, constraint):
        "Build var_registry from a constraint's varkeys."
        return {vk.var_ref: vk for vk in constraint.vks}

    def test_posy_inequality_roundtrip(self):
        """PosynomialInequality: x >= y + 1"""
        x = Variable("x")
        y = Variable("y")
        c = x >= y + 1
        assert isinstance(c, PosynomialInequality)

        ir = c.to_ir()
        assert ir["type"] == "PosynomialInequality"
        assert ir["oper"] == ">="
        assert "left" in ir
        assert "right" in ir

        registry = self._registry(c)
        c2 = PosynomialInequality.from_ir(ir, registry)
        assert isinstance(c2, PosynomialInequality)
        assert c2.oper == ">="
        assert len(c2.unsubbed) == len(c.unsubbed)

    def test_posy_inequality_leq(self):
        """PosynomialInequality with <= operator.

        Note: x + y <= 2*x*y triggers Monomial.__ge__ (subclass reflected
        method) so the stored oper is '>=' with swapped left/right.
        """
        x = Variable("x")
        y = Variable("y")
        c = x + y <= 2 * x * y
        assert isinstance(c, PosynomialInequality)

        ir = c.to_ir()
        assert ir["oper"] in ("<=", ">=")
        registry = self._registry(c)
        c2 = PosynomialInequality.from_ir(ir, registry)
        assert c2.oper == c.oper

    def test_monomial_equality_roundtrip(self):
        """MonomialEquality: x == y"""
        x = Variable("x")
        y = Variable("y")
        c = x == y
        assert isinstance(c, MonomialEquality)

        ir = c.to_ir()
        assert ir["type"] == "MonomialEquality"
        assert ir["oper"] == "="

        registry = self._registry(c)
        c2 = MonomialEquality.from_ir(ir, registry)
        assert isinstance(c2, MonomialEquality)
        assert c2.oper == "="
        assert len(c2.unsubbed) == len(c.unsubbed)

    def test_signomial_inequality_roundtrip(self):
        """SignomialInequality: x >= 1 - y (requires SignomialsEnabled).

        Note: Monomial >= Signomial triggers Signomial.__le__ (subclass
        reflected method) so the stored oper may be '<=' with swapped sides.
        """
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            c = x >= 1 - y
        assert isinstance(c, SignomialInequality)

        ir = c.to_ir()
        assert ir["type"] == "SignomialInequality"
        assert ir["oper"] in ("<=", ">=")

        registry = self._registry(c)
        c2 = SignomialInequality.from_ir(ir, registry)
        assert isinstance(c2, SignomialInequality)
        assert c2.oper == c.oper

    def test_single_signomial_equality_roundtrip(self):
        """SingleSignomialEquality round-trip.

        Constructed directly since Posynomial == Signomial doesn't produce
        a constraint via operator overloading.
        """
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        with SignomialsEnabled():
            c = SingleSignomialEquality(x + y, 1 - z)
        assert isinstance(c, SingleSignomialEquality)

        ir = c.to_ir()
        assert ir["type"] == "SingleSignomialEquality"
        assert ir["oper"] == "="

        registry = self._registry(c)
        c2 = SingleSignomialEquality.from_ir(ir, registry)
        assert isinstance(c2, SingleSignomialEquality)
        assert c2.oper == "="

    def test_array_constraint_to_ir(self):
        """ArrayConstraint serializes as list of element constraints."""
        x = VectorVariable(3, "x")
        y = VectorVariable(3, "y")
        c = x >= y
        assert isinstance(c, ArrayConstraint)

        ir_list = c.to_ir()
        assert isinstance(ir_list, list)
        assert len(ir_list) == 3
        for ir_dict in ir_list:
            assert ir_dict["type"] == "PosynomialInequality"
            assert ir_dict["oper"] == ">="

    def test_constraint_from_ir_dispatch(self):
        """constraint_from_ir dispatches to the correct class."""
        x = Variable("x")
        y = Variable("y")

        c = x >= y + 1
        ir = c.to_ir()
        registry = self._registry(c)
        c2 = constraint_from_ir(ir, registry)
        assert isinstance(c2, PosynomialInequality)

    def test_constraint_from_ir_dispatch_meq(self):
        """constraint_from_ir dispatches MonomialEquality."""
        x = Variable("x")
        y = Variable("y")
        c = x == y
        ir = c.to_ir()
        registry = self._registry(c)
        c2 = constraint_from_ir(ir, registry)
        assert isinstance(c2, MonomialEquality)

    def test_constraint_from_ir_dispatch_signomial(self):
        """constraint_from_ir dispatches SignomialInequality."""
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            c = x >= 1 - y
        ir = c.to_ir()
        registry = self._registry(c)
        c2 = constraint_from_ir(ir, registry)
        assert isinstance(c2, SignomialInequality)

    def test_constraint_lineage(self):
        """Lineage metadata survives round-trip."""
        x = Variable("x")
        y = Variable("y")
        c = x >= y + 1
        c.lineage = (("Aircraft", 0), ("Wing", 0))

        ir = c.to_ir()
        assert ir["lineage"] == [["Aircraft", 0], ["Wing", 0]]

        registry = self._registry(c)
        c2 = PosynomialInequality.from_ir(ir, registry)
        assert c2.lineage == (("Aircraft", 0), ("Wing", 0))

    def test_constraint_no_lineage(self):
        """Constraint without lineage omits lineage key."""
        x = Variable("x")
        y = Variable("y")
        c = x >= y + 1
        c.lineage = ()

        ir = c.to_ir()
        assert "lineage" not in ir

    def test_constraint_json_roundtrip(self):
        """Constraint IR is JSON-serializable and survives round-trip."""
        x = Variable("x")
        y = Variable("y")
        c = x >= y + 1

        ir = c.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)

        registry = self._registry(c)
        c2 = constraint_from_ir(ir2, registry)
        assert isinstance(c2, PosynomialInequality)
        assert c2.oper == ">="

    def test_signomial_json_roundtrip(self):
        """SignomialInequality IR survives JSON round-trip."""
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            c = x >= 1 - y
        ir = c.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)

        registry = self._registry(c)
        c2 = constraint_from_ir(ir2, registry)
        assert isinstance(c2, SignomialInequality)

    def test_constraint_from_ir_unknown_type(self):
        """constraint_from_ir raises on unknown type."""
        with pytest.raises(ValueError, match="Unknown constraint type"):
            constraint_from_ir({"type": "BogusConstraint"}, {})


class TestModelIR:
    """Tests for Model.to_ir() / Model.from_ir() round-trips."""

    def test_ir_document_structure(self):
        """IR document has required top-level keys."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        ir = m.to_ir()
        assert ir["gpkit_ir_version"] == "1.0"
        assert "variables" in ir
        assert "cost" in ir
        assert "constraints" in ir
        assert len(ir["variables"]) == 2
        assert len(ir["constraints"]) == 2

    def test_simple_gp_roundtrip(self):
        """Simple GP round-trip: solve both, compare costs."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        sol = m.solve(verbosity=0)

        ir = m.to_ir()
        m2 = Model.from_ir(ir)
        sol2 = m2.solve(verbosity=0)

        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_substitutions_roundtrip(self):
        """Substitutions survive round-trip."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x, [x >= y], substitutions={y: 3})
        ir = m.to_ir()
        assert "substitutions" in ir
        assert ir["substitutions"]["y"] == 3.0

        m2 = Model.from_ir(ir)
        sol = m.solve(verbosity=0)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_no_substitutions(self):
        """Model without substitutions omits substitutions key."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + y, [x * y >= 1])
        ir = m.to_ir()
        assert "substitutions" not in ir

    def test_units_roundtrip(self):
        """Model with pint units round-trips with matching costs."""
        x = Variable("x", unitrepr="m")
        y = Variable("y", unitrepr="m")
        m = Model(x + y, [x * y >= 1 * Variable("u", unitrepr="m^2")])
        m.substitutions[m["u"]] = 1
        sol = m.solve(verbosity=0)

        ir = m.to_ir()
        m2 = Model.from_ir(ir)
        sol2 = m2.solve(verbosity=0)

        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_nested_model_roundtrip(self):
        """Nested model: lineage appears in IR variables."""
        ac = Aircraft()
        ir = ac.to_ir()

        # Verify lineage in variable refs
        assert "Aircraft0.W" in ir["variables"]
        assert "Aircraft0.Wing0.W" in ir["variables"]
        assert "Aircraft0.Wing0.S" in ir["variables"]

        # Verify lineage metadata
        wing_s = ir["variables"]["Aircraft0.Wing0.S"]
        assert wing_s["lineage"] == [["Aircraft", 0], ["Wing", 0]]

        # Round-trip solve
        sol = ac.solve(verbosity=0)
        ac2 = Model.from_ir(ir)
        sol2 = ac2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_reused_submodel(self):
        """Reused sub-model: both instances appear with distinct var_refs."""
        w = Widget()
        ir = w.to_ir()

        # Both Sub instances should be present
        assert "Widget0.Sub0.m" in ir["variables"]
        assert "Widget0.Sub1.m" in ir["variables"]

        # Round-trip solve
        sol = w.solve(verbosity=0)
        w2 = Model.from_ir(ir)
        sol2 = w2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_sp_roundtrip(self):
        """SP round-trip with localsolve."""
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(x, [x >= 1 - y, y <= 0.5])
        sol = m.localsolve(verbosity=0)

        ir = m.to_ir()
        m2 = Model.from_ir(ir)
        sol2 = m2.localsolve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_vector_variable_roundtrip(self):
        """VectorVariable round-trip."""
        x = VectorVariable(3, "x")
        m = Model(x.prod(), [x >= 1])
        sol = m.solve(verbosity=0)

        ir = m.to_ir()
        # Indexed variables should appear
        assert "x[0]" in ir["variables"]
        assert "x[1]" in ir["variables"]
        assert "x[2]" in ir["variables"]

        m2 = Model.from_ir(ir)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_json_serialization(self):
        """json.dumps(model.to_ir()) succeeds."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        ir = m.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        m2 = Model.from_ir(ir2)
        sol = m.solve(verbosity=0)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_to_json_string(self):
        """to_json returns valid JSON string."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        json_str = to_json(m)
        assert isinstance(json_str, str)
        ir = json.loads(json_str)
        assert ir["gpkit_ir_version"] == "1.0"

    def test_to_json_file(self, tmp_path):
        """to_json writes to file when path given."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        filepath = tmp_path / "model.json"
        result = to_json(m, filepath)
        assert result is None
        assert filepath.exists()

        m2 = from_json(filepath)
        sol = m.solve(verbosity=0)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4

    def test_from_json_string(self):
        """from_json accepts a JSON string."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        json_str = to_json(m)
        m2 = from_json(json_str)
        sol = m.solve(verbosity=0)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4


class TestModelTree:
    """Tests for model_tree structural metadata in the IR."""

    def test_flat_model(self):
        """Flat model with no sub-models: model_tree has no children."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        ir = m.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Model"
        assert tree["children"] == []
        assert len(tree["constraint_indices"]) == 2
        assert tree["constraint_indices"] == [0, 1]
        # All variables should appear in the root node
        assert sorted(tree["variables"]) == sorted(ir["variables"].keys())

    def test_one_level_nesting(self):
        """Aircraft > Wing: tree has one child."""
        ac = Aircraft()
        ir = ac.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Aircraft"
        assert tree["instance_id"].startswith("Aircraft")
        assert len(tree["children"]) == 1

        wing = tree["children"][0]
        assert wing["class"] == "Wing"
        assert "Wing" in wing["instance_id"]
        assert wing["children"] == []

        # Aircraft owns its W variable
        ac_w_vars = [v for v in tree["variables"] if v.endswith(".W")]
        assert len(ac_w_vars) == 1
        # Wing owns S and W
        wing_vars = wing["variables"]
        assert any(v.endswith(".S") for v in wing_vars)
        assert any(v.endswith(".W") for v in wing_vars)

    def test_two_level_nesting(self):
        """Aircraft > Wing > Spar: tree has nested children."""
        ac = SparredAircraft()
        ir = ac.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "SparredAircraft"
        assert len(tree["children"]) == 1

        wing = tree["children"][0]
        assert wing["class"] == "SparredWing"
        assert len(wing["children"]) == 1

        spar = wing["children"][0]
        assert spar["class"] == "Spar"
        assert spar["children"] == []
        assert any(v.endswith(".t") for v in spar["variables"])

    def test_reused_submodel(self):
        """Widget with two Sub instances: same class, different instance_id."""
        w = Widget()
        ir = w.to_ir()
        tree = ir["model_tree"]

        assert tree["class"] == "Widget"
        assert len(tree["children"]) == 2

        sub0, sub1 = tree["children"]
        assert sub0["class"] == "Sub"
        assert sub1["class"] == "Sub"
        assert sub0["instance_id"] != sub1["instance_id"]
        # Each Sub owns its own m variable
        assert len(sub0["variables"]) == 1
        assert len(sub1["variables"]) == 1
        assert sub0["variables"][0] != sub1["variables"][0]
        assert sub0["variables"][0].endswith(".m")
        assert sub1["variables"][0].endswith(".m")

    def test_constraint_ownership(self):
        """Constraint indices correctly map to the flat constraint list."""

        class Top(Model):
            """Top-level model for constraint ownership tests."""

            def setup(self):
                y = Variable("y")
                sub = Sub()
                self.cost = y + sub.cost
                return [y >= 2, sub]

        m = Top()
        ir = m.to_ir()
        tree = ir["model_tree"]

        # Top owns constraint 0 (y >= 2), Sub owns constraint 1 (m >= 1)
        assert tree["constraint_indices"] == [0]
        sub_tree = tree["children"][0]
        assert sub_tree["constraint_indices"] == [0 + 1]  # == [1]

        # All constraint indices should cover the full flat list
        all_indices = set(tree["constraint_indices"])
        for child in tree["children"]:
            all_indices.update(child["constraint_indices"])
        assert all_indices == set(range(len(ir["constraints"])))

    def test_variable_ownership(self):
        """Each variable in the IR appears in exactly one tree node."""
        ac = Aircraft()
        ir = ac.to_ir()
        tree = ir["model_tree"]

        # Collect all variables from all tree nodes
        def collect_vars(node):
            result = list(node["variables"])
            for child in node["children"]:
                result.extend(collect_vars(child))
            return result

        tree_vars = collect_vars(tree)
        ir_vars = set(ir["variables"].keys())

        # Every IR variable appears in some tree node
        assert set(tree_vars) == ir_vars
        # No variable appears in more than one node
        assert len(tree_vars) == len(set(tree_vars))

    def test_model_tree_json_serializable(self):
        """model_tree survives JSON round-trip."""
        ac = Aircraft()
        ir = ac.to_ir()
        json_str = json.dumps(ir)
        ir2 = json.loads(json_str)
        assert "model_tree" in ir2
        assert ir2["model_tree"]["class"] == "Aircraft"
        assert len(ir2["model_tree"]["children"]) == 1

    def test_prev_tests_still_pass_with_model_tree(self):
        """model_tree is additive: previous round-trip still works."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + 2 * y, [x * y >= 1, y >= 0.5])
        ir = m.to_ir()
        assert "model_tree" in ir

        # from_ir ignores model_tree and still produces a solvable model
        m2 = Model.from_ir(ir)
        sol = m.solve(verbosity=0)
        sol2 = m2.solve(verbosity=0)
        assert abs(sol.cost - sol2.cost) < 1e-4
