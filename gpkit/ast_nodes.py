"AST node dataclass hierarchy for expression trees"

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .varkey import VarKey


@dataclass(frozen=True)
class ASTNode:
    """Base class for all AST nodes."""

    def str_without(self, excluded=()):
        "Render this node as a string, for integration with strify/parse_ast."
        raise NotImplementedError


@dataclass(frozen=True)
class VarNode(ASTNode):
    """Reference to a variable.  Holds the VarKey for rich rendering."""

    varkey: VarKey

    @property
    def ref(self):
        "Qualified path string for IR serialization."
        return self.varkey.var_ref

    def str_without(self, excluded=()):
        return self.varkey.str_without(excluded)


@dataclass(frozen=True)
class ConstNode(ASTNode):
    """A numeric constant in the expression tree."""

    value: float

    def str_without(self, excluded=()):
        return f"{self.value:.3g}"


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
        # Lazy import to break circular dependency: repr_conventions imports
        # from ast_nodes (to dispatch on node types), and we import back.
        from .util.repr_conventions import _render_ast_node  # noqa: C0415

        return _render_ast_node(self, excluded)


def to_ast(obj):
    """Convert an operand to an AST node for expression tree construction.

    Used by math.py and array.py when building ASTs for nomial operations.
    Returns VarNode for Variables, the existing .ast for nomials that have one,
    or the object unchanged (raw numbers, index specs, etc. for strify to handle).
    """
    if isinstance(obj, ASTNode):
        return obj
    if hasattr(obj, "ast") and obj.ast is not None:
        return obj.ast
    if hasattr(obj, "key"):
        return VarNode(obj.key)
    return obj
