"Base classes for nomial-layer constraints."

from operator import eq, ge, le
from typing import ClassVar

import numpy as np

from ..ast_nodes import ASTNode, ConstNode, to_ast
from ..util.globals import NamedVariables
from ..util.repr_conventions import UNICODE_EXPONENTS, ReprMixin
from ..util.small_classes import Numbers
from ..util.small_scripts import try_str_without


class SingleEquationConstraint(ReprMixin):
    "Constraint expressible in a single equation."

    latex_opers: ClassVar = {"<=": "\\leq", ">=": "\\geq", "=": "="}
    unicode_opers: ClassVar = {"<=": "≤", ">=": "≥", "=": "="}
    func_opers: ClassVar = {"<=": le, ">=": ge, "=": eq}

    def __init__(self, left, oper, right):
        self.left, self.oper, self.right = left, oper, right

    def str_without(self, excluded=("units",)):
        "String representation without attributes in excluded list"
        leftstr = try_str_without(self.left, excluded)
        rightstr = try_str_without(self.right, excluded)
        rlines = rightstr.split("\n")
        if len(rlines) > 1:
            indent = len("%s %s " % (leftstr.split("\n")[-1], self.oper))
            rightstr = ("\n" + " " * indent).join(rlines)
        if UNICODE_EXPONENTS:
            oper = self.unicode_opers[self.oper]
        else:
            oper = self.oper
        return f"{leftstr} {oper} {rightstr}"

    def to_ir(self):
        "Serialize the common constraint structure to an IR dict."
        return {
            "oper": self.oper,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }

    def latex(self, excluded=("units",), aligned=False):
        "Latex representation without attributes in excluded list"
        lhs = try_str_without(self.left, excluded, latex=True)
        op = self.latex_opers[self.oper]
        rhs = try_str_without(self.right, excluded, latex=True)
        if aligned:
            return f"{lhs} &{op} {rhs}"
        return f"{lhs} {op} {rhs}"


class ArrayConstraint(SingleEquationConstraint, list):
    """One written relation between arrays, standing for one per element.

    `left` and `right` are the sides as written -- usually NomialArrays or the
    veckeys of VectorVariables -- and `constraints` holds the scalar constraint
    for each element, which is what reaches the solver.
    """

    def __init__(self, constraints, left, oper, right):
        self.constraints = constraints
        self.lineage = tuple(NamedVariables.lineage)
        list.__init__(self, constraints)
        SingleEquationConstraint.__init__(self, left, oper, right)

    def __iter__(self):
        yield from self.constraints.flat

    def elements(self):
        "(idx, constraint) for each element, in the order the solver gets them."
        return np.ndenumerate(self.constraints)

    def lines_without(self, excluded):
        "Returns lines for indentation in hierarchical printing."
        return self.str_without(excluded).split("\n")

    def to_ir(self):
        """Serialize the relation as written; its elements serialize themselves.

        A side keeps its AST only if it has one: an array assembled element by
        element, as NomialArray([x, y]), was never written as one expression.
        """
        ir = {"type": "ArrayConstraint", "oper": self.oper}
        ir["shape"] = list(self.constraints.shape)
        if self.lineage:
            ir["lineage"] = [[name, num] for name, num in self.lineage]
        for name, side in (("left", self.left), ("right", self.right)):
            ast = ConstNode(float(side)) if isinstance(side, Numbers) else to_ast(side)
            ir[name] = {"ast": ast.to_ir()} if isinstance(ast, ASTNode) else {}
        return ir

    def __bool__(self):
        "Allows the use of '=' NomialArrays as truth elements."
        return False if self.oper != "=" else bool(self.constraints.all())
