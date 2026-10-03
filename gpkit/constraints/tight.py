"Implements Tight"

from ..util.globals import SignomialsEnabled
from ..util.small_scripts import appendsolwarning, mag
from .set import ConstraintSet


class Tight(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."

    reltol = 1e-3

    def __init__(self, constraints, *, reltol=None, **kwargs):
        super().__init__(constraints)
        self.reltol = reltol or self.reltol
        self.__dict__.update(kwargs)  # NOTE: for Berk's use in labelling

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super().process_result(result)
        for constraint in self.flat():
            with SignomialsEnabled():
                leftval = constraint.left.sub(result.variables).value
                rightval = constraint.right.sub(result.variables).value
            rel_diff = mag(abs(1 - leftval / rightval))
            if rel_diff >= self.reltol:
                if hasattr(leftval, "magnitude"):
                    rightval = rightval.to(leftval.units).magnitude
                    leftval = leftval.magnitude
                cstr = constraint.str_without({"units", "lineage"})
                appendsolwarning(
                    f"{leftval:.4g} {constraint.oper} {rightval:.4g} : {cstr}",
                    result,
                    "Unexpectedly Loose Constraints",
                    subject=constraint,
                    value=rel_diff,
                )
