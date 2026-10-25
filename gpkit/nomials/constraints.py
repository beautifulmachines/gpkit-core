"Base classes for nomial-layer constraints."

from operator import eq, ge, le
from typing import ClassVar

import numpy as np

from ..ast_nodes import ast_to_ir, written_ast
from ..display.render import Rendered
from ..util.globals import NamedVariables
from ..util.repr_conventions import ReprMixin


class SingleEquationConstraint(Rendered, ReprMixin):
    "Constraint expressible in a single equation."

    func_opers: ClassVar = {"<=": le, ">=": ge, "=": eq}

    def __init__(self, left, oper, right):
        self.left, self.oper, self.right = left, oper, right

    def to_ir(self):
        "Serialize the common constraint structure to an IR dict."
        return {
            "oper": self.oper,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }


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

    @property
    def written(self) -> bool:
        "Whether both sides have a written form, so it prints as one vector line."
        return all(written_ast(side) is not None for side in (self.left, self.right))

    def to_ir(self):
        """Serialize the relation as written; its elements serialize themselves.

        A side has an AST only if all of it was written: an array assembled
        element by element, as NomialArray([x, y]), was never one expression,
        whether it is the whole side or only part of one.
        """
        ir = {"type": "ArrayConstraint", "oper": self.oper}
        ir["shape"] = list(self.constraints.shape)
        if self.lineage:
            ir["lineage"] = [[name, num] for name, num in self.lineage]
        for name, side in (("left", self.left), ("right", self.right)):
            ast = written_ast(side)
            ir[name] = {} if ast is None else {"ast": ast_to_ir(ast)}
        return ir

    def __bool__(self):
        "Allows the use of '=' NomialArrays as truth elements."
        return False if self.oper != "=" else bool(self.constraints.all())
