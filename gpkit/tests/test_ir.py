"""Tests for the IR (Intermediate Representation) infrastructure."""

import json

import numpy as np
import pytest

import gpkit
from gpkit import Model, SignomialsEnabled, Variable, VarKey, Vectorize, VectorVariable
from gpkit.ast_nodes import (
    ConstNode,
    ExprNode,
    PiNode,
    UnitsNode,
    VarNode,
    ast_from_ir,
    to_ast,
)
from gpkit.constraints import ArrayConstraint
from gpkit.constraints.bounded import Bounded
from gpkit.constraints.relax import ConstantsRelaxed
from gpkit.constraints.set import build_model_tree
from gpkit.constraints.tight import Tight
from gpkit.nomials.map import NomialMap
from gpkit.nomials.math import (
    MonomialEquality,
    PosynomialInequality,
    Signomial,
    SignomialInequality,
    SingleSignomialEquality,
)
from gpkit.tests.test_margin_objective import SimpleMarginModel
from gpkit.units import qty, units, ureg
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


class MultiComponent(Model):
    """Model with sum() over sub-model variables (triggers Monomial AST child)."""

    def setup(self):
        W = Variable("W", label="total weight")
        sub1 = Sub()
        sub2 = Sub()
        components = [sub1, sub2]
        self.cost = W
        return [W >= sum(c.cost for c in components), sub1, sub2]


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

    def test_to_ast_unit_monomial_becomes_units_node(self):
        """units("lbf") is variable-free with magnitude 1;
        to_ast should return UnitsNode, not raw Monomial."""
        unit_mono = units("lbf")
        node = to_ast(unit_mono)
        assert isinstance(node, UnitsNode)
        assert "lbf" in str(node.units) or "force_pound" in str(node.units)

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


class TestVarKeyIR:
    """Tests for VarKey.to_ir()."""

    def test_plain(self):
        vk = VarKey("x")
        ir = vk.to_ir()
        assert ir["name"] == "x"
        assert "lineage" not in ir
        assert "units" not in ir

    def test_with_units(self):
        "Units are reported as declared -- to_toml() writes this back out."
        vk = VarKey("S", units="m^2")
        assert vk.to_ir()["units"] == "m^2"

    @pytest.mark.parametrize("declared", ["USD", "USD/kg", "N*m", "ohm", "lbf", "m^2"])
    def test_ir_units_can_always_be_read_back(self, declared):
        """A unit the IR writes has to be one a registry can parse.

        Two spellings of one unit are fine, since nothing compares them as
        strings -- but an unparseable one is not. The symbol form fails that:
        USD's symbol is "$", which pint cannot read back.
        """
        vk = VarKey("q", units=declared)
        assert ureg(vk.to_ir()["units"]).units == vk.units.units

    def test_units_given_as_a_quantity_are_still_readable(self):
        "Nothing was declared as a string here, so a spelling has to be derived."
        vk = VarKey("q", units=qty("USD"))
        assert ureg(vk.to_ir()["units"]).units == vk.units.units

    def test_with_lineage(self):
        vk = VarKey("S", lineage=(("Aircraft", 0), ("Wing", 0)))
        assert vk.to_ir()["lineage"] == [["Aircraft", 0], ["Wing", 0]]

    def test_with_idx(self):
        vk = VarKey("c_l", idx=(1,), shape=(3,))
        ir = vk.to_ir()
        assert ir["idx"] == [1]
        assert ir["shape"] == [3]

    def test_with_label(self):
        vk = VarKey("W", label="total weight")
        assert vk.to_ir()["label"] == "total weight"

    def test_json_serializable(self):
        vk = VarKey("S", lineage=(("Wing", 0),), units="m^2", label="area")
        ir = vk.to_ir()
        assert json.loads(json.dumps(ir)) == ir

    def test_dimensionless_omits_units(self):
        vk = VarKey("x")
        ir = vk.to_ir()
        assert "units" not in ir


class TestASTNodeIR:
    """Tests for AST node to_ir() / ast_from_ir() round-trip."""

    def _var_registry(self, *variables):
        "Build var_registry from Variables."
        return {v.key.ref: v.key for v in variables}

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

    def test_pinode_is_constnode(self):
        node = PiNode()
        assert isinstance(node, ConstNode)
        assert node.value == pytest.approx(np.pi)

    def test_pinode_str(self):
        node = PiNode()
        s = node.str_without()
        assert s in ("π", "PI")

    def test_pinode_latex(self):
        node = PiNode()
        assert node.latex() == r"\pi"

    def test_pinode_ir_roundtrip(self):
        node = PiNode()
        ir = node.to_ir()
        assert ir == {"node": "pi"}
        node2 = ast_from_ir(ir, None)
        assert isinstance(node2, PiNode)

    def test_unitsnode_ir_roundtrip(self):
        node = UnitsNode(units("W").hmap.units)
        ir = node.to_ir()
        assert ir == {"node": "units", "units": "1 watt"}
        node2 = ast_from_ir(ir, None)
        assert isinstance(node2, UnitsNode)
        assert node2.units == node.units

    def test_to_ast_gpkit_pi(self):
        result = to_ast(gpkit.pi)
        assert isinstance(result, PiNode)

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
    """Tests for NomialMap.to_ir()."""

    def test_monomial(self):
        """Single-term NomialMap (monomial)."""
        x = Variable("x")
        hmap = NomialMap({HashVector({x.key: 2}): 3.0})
        ir = hmap.to_ir()
        assert len(ir["terms"]) == 1
        assert ir["terms"][0]["coeff"] == 3.0
        assert ir["terms"][0]["exps"]["x"] == 2

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
        assert len(hmap.to_ir()["terms"]) == 2

    def test_constant_term(self):
        """Constant term uses EMPTY_HV, no exps key."""
        hmap = NomialMap({EMPTY_HV: 5.0})
        ir = hmap.to_ir()
        assert len(ir["terms"]) == 1
        assert "exps" not in ir["terms"][0]
        assert ir["terms"][0]["coeff"] == 5.0

    def test_with_units(self):
        x = Variable("x", units="m")
        hmap = NomialMap({HashVector({x.key: 1}): 1.0})
        hmap.units = qty("m")
        assert "units" in hmap.to_ir()

    def test_json_serializable(self):
        """NomialMap IR is JSON-serializable."""
        x = Variable("x")
        ir = NomialMap({HashVector({x.key: 1}): 2.5}).to_ir()
        assert json.loads(json.dumps(ir)) == ir


class TestNomialIR:
    """Tests for Signomial.to_ir()."""

    def test_monomial(self):
        x = Variable("x")
        ir = (2 * x**3).to_ir()
        assert ir["type"] == "Monomial"
        assert ir["terms"] == [{"coeff": 2.0, "exps": {"x": 3}}]

    def test_posynomial(self):
        x = Variable("x")
        y = Variable("y")
        ir = (x + 2 * y).to_ir()
        assert ir["type"] == "Posynomial"
        assert len(ir["terms"]) == 2

    def test_signomial(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            s = x - y
        assert s.to_ir()["type"] == "Signomial"

    def test_ast_is_serialized(self):
        x = Variable("x")
        y = Variable("y")
        assert "ast" in (x + 2 * y).to_ir()

    def test_units_are_serialized(self):
        x = Variable("x", units="m")
        y = Variable("y", units="m")
        assert "units" in (x + y).to_ir()

    def test_json_serializable(self):
        x = Variable("x")
        y = Variable("y")
        ir = (x + 2 * y).to_ir()
        assert json.loads(json.dumps(ir)) == ir

    def test_constant_nomial(self):
        """A nomial with only a constant term."""
        assert Signomial(5.0).to_ir()["type"] == "Monomial"


class TestConstraintIR:
    """Tests for constraint to_ir()."""

    def test_posy_inequality(self):
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

    def test_posy_inequality_leq(self):
        """PosynomialInequality with <= operator.

        Note: x + y <= 2*x*y triggers Monomial.__ge__ (subclass reflected
        method) so the stored oper is '>=' with swapped left/right.
        """
        x = Variable("x")
        y = Variable("y")
        c = x + y <= 2 * x * y
        assert isinstance(c, PosynomialInequality)
        assert c.to_ir()["oper"] in ("<=", ">=")

    def test_monomial_equality(self):
        """MonomialEquality: x == y"""
        x = Variable("x")
        y = Variable("y")
        c = x == y
        assert isinstance(c, MonomialEquality)

        ir = c.to_ir()
        assert ir["type"] == "MonomialEquality"
        assert ir["oper"] == "="

    def test_signomial_inequality(self):
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

    def test_single_signomial_equality(self):
        """SingleSignomialEquality.

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

    def test_slice_in_ast_serializes(self):
        """A slice in the AST (e.g. from arr[:j].sum()) serializes.

        This pattern appears in vectorized integration constraints like
        r[j] >= dr[:j].sum() used in BEMTHover and similar models.
        """
        with Vectorize(4):
            dr = Variable("dr", "-", "bin width")

        x = Variable("x", "-")
        ir = Model(x, [x >= dr[:2].sum()]).to_ir()
        assert json.loads(json.dumps(ir)) == ir

    def test_tuple_with_integer_index_in_ast_serializes(self):
        """2D array indexed with [int, :] produces a tuple child containing a
        raw integer in the IR, which has to survive JSON."""
        a = VectorVariable((2, 3), "a", "-")
        x = Variable("x", "-")
        ir = Model(x, [x >= a[0, :].sum()]).to_ir()
        assert json.loads(json.dumps(ir)) == ir

    def test_constraint_lineage(self):
        x = Variable("x")
        y = Variable("y")
        c = x >= y + 1
        c.lineage = (("Aircraft", 0), ("Wing", 0))
        assert c.to_ir()["lineage"] == [["Aircraft", 0], ["Wing", 0]]

    def test_constraint_no_lineage(self):
        """Constraint without lineage omits lineage key."""
        x = Variable("x")
        y = Variable("y")
        c = x >= y + 1
        c.lineage = ()

        ir = c.to_ir()
        assert "lineage" not in ir

    def test_constraint_json_serializable(self):
        x = Variable("x")
        y = Variable("y")
        ir = (x >= y + 1).to_ir()
        assert json.loads(json.dumps(ir)) == ir

    def test_signomial_json_serializable(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            c = x >= 1 - y
        ir = c.to_ir()
        assert json.loads(json.dumps(ir)) == ir


class TestModelIR:
    """Tests for Model.to_ir()."""

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

    def test_substitutions(self):
        x = Variable("x")
        y = Variable("y")
        m = Model(x, [x >= y], substitutions={y: 3})
        assert m.to_ir()["substitutions"]["y"] == 3.0

    def test_no_substitutions(self):
        """Model without substitutions omits substitutions key."""
        x = Variable("x")
        y = Variable("y")
        m = Model(x + y, [x * y >= 1])
        ir = m.to_ir()
        assert "substitutions" not in ir

    def test_substituted_variable_is_declared(self):
        """A substitution names a variable the document declares (#214).

        A constant no constraint references is still a parameter of the model,
        so dropping its declaration while keeping its value would leave the
        document asserting a value for a variable it never introduces.
        """
        x = Variable("x", "m")
        rho = Variable("rho", "kg/m^3", "declared, referenced by nothing")
        m = Model(x, [x >= Variable("L", 2, "m")], substitutions={rho: 1600})
        ir = m.to_ir()
        assert set(ir["substitutions"]) <= set(ir["variables"])
        assert rho.key.ref in ir["variables"]

    def test_units_pow_fractional_to_ir(self):
        "Regression for #159: units(...)**<float> must not raise IRSerializationError."
        W = Variable("W", "lbf")
        W_eng = Variable("W_eng", "lbf")
        mfac = Variable("mfac", "-")
        m = Model(W, [W / mfac >= 2.572 * W_eng**0.922 * units("lbf") ** 0.078])
        ir = m.to_ir()  # must not raise
        assert "constraints" in ir

    def test_nested_model(self):
        """Nested model: lineage appears in IR variables."""
        ir = Aircraft().to_ir()

        # Verify lineage in variable refs
        assert "Aircraft.W" in ir["variables"]
        assert "Aircraft.Wing.W" in ir["variables"]
        assert "Aircraft.Wing.S" in ir["variables"]

        # Verify lineage metadata
        wing_s = ir["variables"]["Aircraft.Wing.S"]
        assert wing_s["lineage"] == [["Aircraft", 0], ["Wing", 0]]

    def test_reused_submodel(self):
        """Reused sub-model: both instances appear with distinct refs."""
        ir = Widget().to_ir()
        assert "Widget.Sub.m" in ir["variables"]
        assert "Widget.Sub1.m" in ir["variables"]

    def test_vector_variable(self):
        """Vector elements appear individually, ref including #shape."""
        x = VectorVariable(3, "x")
        ir = Model(x.prod(), [x >= 1]).to_ir()
        assert "x[0]#3" in ir["variables"]
        assert "x[1]#3" in ir["variables"]
        assert "x[2]#3" in ir["variables"]

    def test_json_serialization(self):
        """json.dumps(model.to_ir()) succeeds."""
        x = Variable("x")
        y = Variable("y")
        ir = Model(x + 2 * y, [x * y >= 1, y >= 0.5]).to_ir()
        assert json.loads(json.dumps(ir)) == ir


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

    def test_model_tree_uses_children_not_lineage(self):
        """build_model_tree() reflects _children, not just lineage."""

        class _Wing(Model):
            def setup(self):
                S = Variable("S")
                self.cost = S
                return [S >= 10]

        class _Aircraft(Model):
            wing: "_Wing"

            def setup(self):
                W = Variable("W")
                self.wing = _Wing()
                self.cost = W
                return [W >= self.wing.cost * 1.2, self.wing]

        a = _Aircraft()
        tree = build_model_tree(a)
        # The children list from _children has exactly one entry: _Wing
        assert len(tree["children"]) == 1
        assert tree["children"][0]["class"] == "_Wing"
        # Also verify submodels and tree children agree
        assert len(a.submodels) == 1


class TestMultiComponentIR:
    """Tests for models using sum() over sub-model variables."""

    def test_sum_over_submodel_costs(self):
        """sum(c.cost for c in components) produces serializable AST."""
        ir = MultiComponent().to_ir()
        assert ir["constraints"]
        assert json.loads(json.dumps(ir)) == ir


# ── Monomial substitution serialization ───────────────────────────────


def test_to_ir_monomial_substitution():
    """to_ir() must not crash when a substitution value is a Monomial (e.g.
    created by multiplying a scalar by a units() expression).  Regression
    test for the bug where float(Monomial) raised TypeError."""
    C3 = Variable("C3", "km^2/s^2")
    C3min = Variable("C3min", "km^2/s^2")
    m = Model(C3, [C3 >= C3min], {C3min: 9.0 * units("km^2/s^2")})
    ir = m.to_ir()
    assert ir["substitutions"][C3min.key.ref] == pytest.approx(9.0)


# ── Solution IR ───────────────────────────────────────────────────────


class _IRChild(Model):
    "A child, so sensitivities span more than one model."

    def setup(self):
        self.t = Variable("t", "mm", "thickness")
        self.t_min = Variable("t_min", 2.0, "mm", "min thickness")
        return [self.t >= self.t_min]


def _solved():
    "Two variables, two units, one child, one fixed value."

    class _IRTop(Model):
        def setup(self):
            self.D = Variable("D", "m", "diameter")
            self.child = _IRChild()
            self.cost = self.D
            return [self.D >= 1000 * self.child.t, self.child]

    m = _IRTop()
    return m, m.solve(verbosity=0)


class TestSolutionIR:
    """A solution serializes to refs and numbers, and joins the model IR.

    Structure lives in Model.to_ir(); this carries results only, so the two
    documents are a pair and nothing is encoded twice.
    """

    def test_cost_carries_value_and_units(self):
        """A cost expression declares no units, so full names are derived.

        Symbols would read better but need not parse back, and a money objective
        is an ordinary thing to minimize.
        """
        _, sol = _solved()
        assert sol.to_ir()["cost"] == {
            "value": pytest.approx(2.0, rel=1e-6),
            "units": "meter",
        }

    def test_a_money_objective_has_readable_cost_units(self):
        price = Variable("price_ir", "USD")
        floor = Variable("price_min_ir", 40.0, "USD")
        sol = Model(price, [price >= floor]).solve(verbosity=0)
        assert ureg(sol.to_ir()["cost"]["units"]).units == ureg("USD").units

    def test_values_are_in_declared_units(self):
        "t is 2 mm, not 0.002 m -- the declared unit is the one reported."
        m, sol = _solved()
        ir = sol.to_ir()
        assert ir["primal"][m.child.t.key.ref] == {
            "value": pytest.approx(2.0, rel=1e-6),
            "units": "mm",
        }
        assert ir["constants"][m.child.t_min.key.ref] == {
            "value": pytest.approx(2.0),
            "units": "mm",
        }

    def test_dimensionless_omits_units(self):
        x = Variable("x_dimless")
        sol = Model(x, [x >= 1]).solve(verbosity=0)
        assert sol.to_ir()["primal"][x.key.ref] == {"value": pytest.approx(1.0)}
        assert "units" not in sol.to_ir()["cost"]

    def test_units_are_spelled_as_the_model_ir_spells_them(self):
        """One spelling of a unit across both documents, and pint can read it.

        The display form substitutes a middle dot for products ('m⋅N'), which no
        unit registry parses -- so it cannot be what a joinable IR carries.
        """
        torque = Variable("tau_ir", "N*m", "torque")
        f_min = Variable("tau_min_ir", 2.0, "N*m")
        m = Model(torque, [torque >= f_min])
        sol = m.solve(verbosity=0)
        declared = m.to_ir()["variables"][torque.key.ref]["units"]
        assert sol.to_ir()["primal"][torque.key.ref]["units"] == declared
        assert ureg(declared).units == torque.key.units.units

    def test_meta_is_status_soltime_warnings_only(self):
        "Not wholesale: models write into meta (bounded.py adds boundedness)."
        _, sol = _solved()
        assert set(sol.to_ir()["meta"]) == {"status", "soltime", "warnings"}

    def test_sensitivities_are_keyed_by_ref(self):
        m, sol = _solved()
        ir = sol.to_ir()
        model_ir = m.to_ir()
        assert set(ir["sensitivities"]["constraints"]) == {
            c["key"] for c in model_ir["constraints"]
        }
        assert set(ir["sensitivities"]["variables"]) <= set(model_ir["variables"])

    def test_variables_join_the_model_ir(self):
        "Every solution key names a variable the model IR declares."
        m, sol = _solved()
        declared = set(m.to_ir()["variables"])
        ir = sol.to_ir()
        assert set(ir["primal"]) | set(ir["constants"]) <= declared

    def test_is_json_serializable(self):
        _, sol = _solved()
        assert json.loads(json.dumps(sol.to_ir())) == sol.to_ir()

    def test_vectors_are_elements_only(self):
        "Model.to_ir() already carries the parent entry with its shape."
        xv = VectorVariable(3, "xv", "m")
        sol = Model(sum(xv), [xv >= 2 * Variable("u_v", 1, "m")]).solve(verbosity=0)
        refs = set(sol.to_ir()["primal"])
        assert len(refs) == 3
        assert all("[" in r for r in refs)  # elements, never the bare parent

    def test_derived_is_omitted_when_there_is_no_margin(self):
        _, sol = _solved()
        assert "derived" not in sol.to_ir()

    def test_derived_carries_the_margin_and_ref_keyed_sensitivities(self):
        m = SimpleMarginModel()
        sol = m.solve(verbosity=0)
        derived = sol.to_ir()["derived"]
        assert derived["name"] == "mass margin"
        assert derived["units"] == "kg"
        assert set(derived["sensitivities"]) <= set(m.to_ir()["variables"])


class TestWarningSubjectsBecomeRefs:
    "A warning's subject is a live object in process and a ref in the IR."

    def test_a_constraint_subject_becomes_its_constraint_key(self):
        x = Variable("x_w")
        x_min = Variable("x_min_w", 2)
        m = Model(x, [Tight([x >= 1]), x >= x_min])
        sol = m.solve(verbosity=0)
        (warn,) = sol.to_ir()["meta"]["warnings"]["Unexpectedly Loose Constraints"]
        keys = {k.ref for k in sol.sens.constraints_by_key}
        assert warn["subject"] in keys
        assert warn["value"] == pytest.approx(1, abs=1e-3)

    def test_a_variable_subject_becomes_its_varkey_ref(self):
        x = Variable("x_r")
        x_min = Variable("x_min_r", 2)
        x_max = Variable("x_max_r", 1)
        inner = Model(x, [x <= x_max, x >= x_min])
        relaxed = ConstantsRelaxed(inner)
        sol = Model(relaxed.relaxvars.prod(), relaxed).solve(verbosity=0)
        ir = sol.to_ir()
        warns = ir["meta"]["warnings"]["Relaxed Constants"]
        known = set(ir["primal"]) | set(ir["constants"])
        assert warns and all(w["subject"] in known for w in warns)

    def test_a_subjectless_warning_omits_the_field(self):
        x = Variable("x_b")
        sol = Model(1 / x, Bounded([x >= 1])).solve(verbosity=0)
        (warn,) = sol.to_ir()["meta"]["warnings"]["Arbitrarily Bounded Variables"][:1]
        assert "subject" not in warn
        assert "value" not in warn
