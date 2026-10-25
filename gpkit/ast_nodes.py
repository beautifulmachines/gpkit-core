"AST node dataclass hierarchy for expression trees"

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce

import numpy as np

from .exceptions import IRSerializationError
from .units import ureg


@dataclass(frozen=True)
class ASTNode:
    """Base class for all AST nodes. gpkit.display.render renders them."""

    def to_ir(self):
        "Serialize this AST node to an IR dict."
        raise NotImplementedError


@dataclass(frozen=True)
class VarNode(ASTNode):
    """A variable, by its VarKey.ref.

    A tree is syntax only, so a live tree and one read from a document are
    the same; whoever renders it names each ref.
    """

    ref: str

    def to_ir(self):
        return {"node": "var", "ref": self.ref}


@dataclass(frozen=True)
class ConstNode(ASTNode):
    """A numeric constant in the expression tree."""

    value: float

    def to_ir(self):
        return {"node": "const", "value": self.value}


@dataclass(frozen=True)
class PiNode(ConstNode):
    """A ConstNode for the mathematical constant π.

    Inherits from ConstNode so isinstance(node, ConstNode) is True and
    node.value gives np.pi for any code that needs the numeric value.
    Renders as π (text) and \\pi (LaTeX).
    """

    value: float = field(default_factory=lambda: np.pi)

    def to_ir(self):
        return {"node": "pi"}


@dataclass(frozen=True)
class UnitsNode(ASTNode):
    """A pure-units leaf in the expression tree (e.g. units.m, units.lbf).

    Rendered as its units, so ``4 * units.m`` shows what the 4 measures.  TOML
    writes it as a unit literal, never as 1: its units can differ in scale from
    a neighboring variable's (a kW variable over a bare 1 W carries a factor of
    1000), so dropping it would change the solved values on reload.
    """

    units: object  # pint Quantity

    def to_ir(self):
        return {"node": "units", "units": str(self.units)}


@dataclass(frozen=True)
class ExprNode(ASTNode):
    """An operation combining child AST nodes.

    op is one of: "add", "mul", "div", "pow", "neg", "sum", "prod", "index"
    children is a tuple whose shape depends on op:
      add/mul/div: (left, right)
      pow: (base, exponent)  -- exponent may be a numeric value
      neg: (operand,)
      sum/prod: (array_node,)
      index: (array_node, idx_spec)
    """

    op: str
    children: tuple

    def to_ir(self):
        return {
            "node": "expr",
            "op": self.op,
            "children": [ast_to_ir(c) for c in self.children],
        }


def expr(op, children):
    """The ExprNode applying op to its operands' ASTs, or None if one has none.

    An expression is written only if all of it was: an operand assembled
    element by element, whose to_ast() is None, leaves the whole without an AST
    rather than with one holding something no renderer or reader can follow.
    """
    children = tuple(children)
    if any(c is None for c in children):
        return None  # an operand has no AST, so neither does the expression
    return ExprNode(op, children)


def varnodes(node):
    "Every VarNode in a tree."
    if isinstance(node, VarNode):
        yield node
    elif isinstance(node, ExprNode):
        for child in node.children:
            yield from varnodes(child)


def variables_of(obj):
    """{ref: VarKey} for every variable in obj's math, and their vectors.

    obj is a nomial, an array of them, or a VarKey.  A vector's ref is included
    because a written form names a whole vector where the math holds its
    elements.
    """
    if isinstance(obj, np.ndarray):
        vks = [vk for el in obj.flat for vk in getattr(el, "vks", ())]
    else:
        vks = list(getattr(obj, "vks", ()))
    if hasattr(obj, "key"):  # a variable, a vector variable, or a VarKey itself
        vks.append(obj.key)
    keys = {vk.ref: vk for vk in vks}
    keys.update((vk.veckey.ref, vk.veckey) for vk in vks if vk.veckey is not None)
    return keys


def written_ast(obj, variables=None):
    """obj's written form, or None if it has none it can be shown by.

    A written form that names a variable obj's math lacks is not used: in
    3*x/x*y, x cancelled, so nothing could name it and no document lists it.
    """
    tree = to_ast(obj)
    if not isinstance(tree, ASTNode):
        return tree  # None, or a raw number
    variables = variables_of(obj) if variables is None else variables
    if any(node.ref not in variables for node in varnodes(tree)):
        return None
    return tree


def _term_order(term):
    "Sort key for an IR term: its variables' refs and exponents, constants last."
    powers = sorted((ref, x) for ref, x in term.get("exps", {}).items() if x)
    return (not powers, powers)


def _product(factors):
    return reduce(lambda a, b: ExprNode("mul", (a, b)), factors)


def ast_from_terms(terms):
    """The AST of a sum of monomials, for math with no written form.

    terms are in the IR's form, [{"coeff": c, "exps": {ref: x}}], which an
    hmap's to_ir() also gives.  Variables with negative exponents form the
    denominator, so 2*x/y reads as written.

    Terms and factors are sorted by ref, constants last, since an hmap's order
    depends on how it was built.
    """
    monomials = []
    for term in sorted(terms, key=_term_order):
        coeff = term["coeff"]
        exps = sorted(term.get("exps", {}).items())
        num = [
            VarNode(k) if x == 1 else ExprNode("pow", (VarNode(k), x))
            for k, x in exps
            if x > 0
        ]
        den = [
            VarNode(k) if x == -1 else ExprNode("pow", (VarNode(k), -x))
            for k, x in exps
            if x < 0
        ]
        if abs(coeff) != 1 or not num:
            num.insert(0, ConstNode(float(abs(coeff))))
        mono = _product(num)
        for factor in den:  # x/y/z, which LaTeX sets as one fraction
            mono = ExprNode("div", (mono, factor))
        monomials.append(ExprNode("neg", (mono,)) if coeff < 0 else mono)
    if not monomials:
        return ConstNode(0.0)
    return reduce(lambda a, b: ExprNode("add", (a, b)), monomials)


def ast_to_ir(child):
    "Serialize an AST node or raw value to IR; the inverse of ast_from_ir."
    if isinstance(child, ASTNode):
        return child.to_ir()
    if isinstance(child, (int, float)):
        return child
    if isinstance(child, slice):
        return {
            "node": "slice",
            "start": child.start,
            "stop": child.stop,
            "step": child.step,
        }
    if isinstance(child, tuple):
        return {"node": "tuple", "items": [ast_to_ir(el) for el in child]}
    # numpy scalars and similar numeric types
    try:
        return float(child)
    except (TypeError, ValueError) as exc:
        raise IRSerializationError(
            f"Cannot serialize AST child of type {type(child).__name__}: {child!r}"
        ) from exc


def _list_from_ir(items):
    "Deserialize a list of AST children: dicts recurse, raw values pass through."
    return [ast_from_ir(c) if isinstance(c, dict) else c for c in items]


def _ast_from_ir_var(d):
    return VarNode(d["ref"])


def _ast_from_ir_const(d):
    return ConstNode(d["value"])


def _ast_from_ir_pi(_d):
    return PiNode()


def _ast_from_ir_expr(d):
    return ExprNode(d["op"], tuple(_list_from_ir(d["children"])))


def _ast_from_ir_units(d):
    return UnitsNode(ureg.Quantity(d["units"]))


def _ast_from_ir_slice(d):
    return slice(d.get("start"), d.get("stop"), d.get("step"))


def _ast_from_ir_tuple(d):
    return tuple(_list_from_ir(d["items"]))


_AST_NODE_BUILDERS = {
    "var": _ast_from_ir_var,
    "const": _ast_from_ir_const,
    "pi": _ast_from_ir_pi,
    "units": _ast_from_ir_units,
    "expr": _ast_from_ir_expr,
    "slice": _ast_from_ir_slice,
    "tuple": _ast_from_ir_tuple,
}


def ast_from_ir(ir_dict):
    "Reconstruct an AST node from its IR dict, or pass a raw number through."
    if not isinstance(ir_dict, dict):
        return ir_dict  # e.g. an exponent in pow
    builder = _AST_NODE_BUILDERS.get(ir_dict["node"])
    if builder is None:
        raise ValueError(f"Unknown AST IR node type: {ir_dict['node']}")
    return builder(ir_dict)


def to_ast(obj):
    """Convert an operand to an AST node for expression tree construction.

    Used by math.py and array.py when building ASTs for nomial operations.
    Returns VarNode for Variables, the existing .ast for nomials that have one,
    None for a nomial or array that has none -- it was never written as one
    expression -- or the object unchanged (raw numbers, index specs, etc.).
    """
    if isinstance(obj, ASTNode):
        return obj
    # Variable-free nomial (e.g. units("lbf") or 4*units("m^2")):
    # collapse to a leaf node.  This runs before the .ast check so that
    # unit arithmetic like units.m**2 (which builds an intermediate pow
    # AST) is replaced with one flat node carrying the final units.
    if hasattr(obj, "hmap"):
        try:
            ((exp, c),) = obj.hmap.items()
            if not exp:  # empty HashVector = no variables
                c_float = float(c)
                if c_float == 1.0 and obj.hmap.units is not None:
                    return UnitsNode(obj.hmap.units)
                # For valued constants, fall through to the obj.ast check below
                # so the existing AST (which may carry UnitsNode children) is
                # preferred over a bare ConstNode that would lose unit info.
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "ast") and obj.ast is not None:
        return obj.ast
    if hasattr(obj, "hmap"):
        try:
            ((exp, c),) = obj.hmap.items()
            if not exp:
                return ConstNode(float(c))
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "key"):
        return VarNode(obj.key.ref)
    # a nomial or array that got this far was never written as one expression;
    # a list or tuple operand is an array too, as in y * [1, 2, 3]
    unwritten = hasattr(obj, "hmap") or isinstance(obj, (np.ndarray, list, tuple))
    return None if unwritten else obj
