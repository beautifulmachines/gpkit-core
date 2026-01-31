"Implements SingleEquationConstraint"

from operator import eq, ge, le

from ..util.repr_conventions import UNICODE_EXPONENTS, ReprMixin
from ..util.small_scripts import try_str_without


class SingleEquationConstraint(ReprMixin):
    "Constraint expressible in a single equation."

    latex_opers = {"<=": "\\leq", ">=": "\\geq", "=": "="}
    unicode_opers = {"<=": "≤", ">=": "≥", "=": "="}
    func_opers = {"<=": le, ">=": ge, "=": eq}

    def __init__(self, left, oper, right):
        self.left, self.oper, self.right = left, oper, right

    def str_without(self, excluded="units"):
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
        return " ".join((leftstr, oper, rightstr))

    def to_ir(self):
        "Serialize the common constraint structure to an IR dict."
        return {
            "oper": self.oper,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }

    def latex(self, excluded="units"):
        "Latex representation without attributes in excluded list"
        return " ".join(
            (
                try_str_without(self.left, excluded, latex=True),
                self.latex_opers[self.oper],
                try_str_without(self.right, excluded, latex=True),
            )
        )
