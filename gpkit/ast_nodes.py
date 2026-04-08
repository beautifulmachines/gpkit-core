"AST node dataclass hierarchy for expression trees"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import IRSerializationError
from .util.repr_conventions import (
    PI_STR,
    _render_ast_node,
    _render_ast_node_latex,
    latex_unitstr,
    unitstr,
)

if TYPE_CHECKING:
    from .varkey import VarKey


@dataclass(frozen=True)
class ASTNode:
    """Base class for all AST nodes."""

    def str_without(self, excluded=()):
        "Render this node as a string, for integration with strify/parse_ast."
        raise NotImplementedError

    def to_ir(self):
        "Serialize this AST node to an IR dict."
        raise NotImplementedError


@dataclass(frozen=True)
class VarNode(ASTNode):
    """Reference to a variable.  Holds the VarKey for rich rendering."""

    varkey: VarKey

    @property
    def ref(self):
        "Qualified path string for IR serialization."
        return self.varkey.ref

    def str_without(self, excluded=()):
        return self.varkey.str_without(excluded)

    def latex(self, excluded=()):
        "Render this variable node as a LaTeX string."
        return self.varkey.latex(excluded)

    def to_ir(self):
        return {"node": "var", "ref": self.ref}


@dataclass(frozen=True)
class ConstNode(ASTNode):
    """A numeric constant in the expression tree."""

    value: float

    def str_without(self, excluded=()):
        return f"{self.value:.3g}"

    def latex(self, _excluded=()):
        "Render this constant as a LaTeX string."
        return f"{self.value:.4g}"

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

    def str_without(self, excluded=()):
        return PI_STR

    def latex(self, _excluded=()):
        return r"\pi"

    def to_ir(self):
        return {"node": "pi"}


@dataclass(frozen=True)
class UnitsNode(ASTNode):
    """A pure-units leaf in the expression tree (e.g. units.m, units.lbf).

    Renders as the unit string so that expressions like ``4 * units.m``
    display the physical units of the constant to the reader.
    """

    units: object  # pint Quantity

    def str_without(self, excluded=()):
        if "units" in excluded:
            return ""
        return unitstr(self.units, "[%s]")

    def latex(self, excluded=()):
        "Render this units node as a LaTeX string."
        # Suppressed by "ast_units" (used when Nomial.latex appends unit suffix
        # separately), but not by plain "units" (which only suppresses variable
        # unit annotations — constants must always show their dimensional context).
        if "ast_units" in excluded:
            return ""
        return latex_unitstr(self.units).lstrip("~")

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

    def str_without(self, excluded=()):
        return _render_ast_node(self, excluded)

    def latex(self, excluded=()):
        "Render this expression node as a LaTeX string."
        return _render_ast_node_latex(self, excluded)

    def to_ir(self):
        return {
            "node": "expr",
            "op": self.op,
            "children": [_child_to_ir(c) for c in self.children],
        }


def _child_to_ir(child):
    "Serialize an AST child (node or raw value) to IR."
    if isinstance(child, ASTNode):
        return child.to_ir()
    if isinstance(child, (int, float)):
        return child
    # numpy scalars and similar numeric types
    try:
        return float(child)
    except (TypeError, ValueError) as exc:
        raise IRSerializationError(
            f"Cannot serialize AST child of type {type(child).__name__}: {child!r}"
        ) from exc


def ast_from_ir(ir_dict, var_registry):
    """Reconstruct an AST node from its IR dict.

    Parameters
    ----------
    ir_dict : dict or number
        The IR representation of an AST node.
    var_registry : dict
        Mapping from ref strings to VarKey objects.
    """
    if not isinstance(ir_dict, dict):
        assert var_registry is None
        return ir_dict  # raw number passthrough (e.g., exponent in pow)
    node = ir_dict["node"]
    if node == "var":
        return VarNode(var_registry[ir_dict["ref"]])
    if node == "const":
        return ConstNode(ir_dict["value"])
    if node == "pi":
        return PiNode()
    if node == "expr":
        children = []
        for c in ir_dict["children"]:
            if isinstance(c, dict):
                children.append(ast_from_ir(c, var_registry))
            else:
                children.append(c)  # raw number
        return ExprNode(ir_dict["op"], tuple(children))
    raise ValueError(f"Unknown AST IR node type: {node}")


def to_ast(obj):
    """Convert an operand to an AST node for expression tree construction.

    Used by math.py and array.py when building ASTs for nomial operations.
    Returns VarNode for Variables, the existing .ast for nomials that have one,
    or the object unchanged (raw numbers, index specs, etc. for strify to handle).
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
                # For valued constants, prefer the existing AST (which may carry
                # UnitsNode children) over a bare ConstNode that loses unit info
                if hasattr(obj, "ast") and obj.ast is not None:
                    return obj.ast
                return ConstNode(c_float)
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "ast") and obj.ast is not None:
        return obj.ast
    if hasattr(obj, "key"):
        return VarNode(obj.key)
    return obj
