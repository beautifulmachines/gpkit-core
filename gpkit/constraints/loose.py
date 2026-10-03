"Implements Loose"

from ..util.small_scripts import appendsolwarning
from .set import ConstraintSet


class Loose(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."

    senstol = 1e-5
    raiseerror = False

    def __init__(self, constraints, *, senstol=None):
        super().__init__(constraints)
        self.senstol = senstol or self.senstol

    def process_result(self, result):
        "Checks that all constraints are satisfied with equality"
        super().process_result(result)
        for constraint in self.flat():
            c_senss = result.sens.constraints.get(constraint, 0)
            if c_senss >= self.senstol:
                cstr = constraint.str_without({"units", "lineage"})
                appendsolwarning(
                    f"{c_senss:+6.2g} : {cstr}",
                    result,
                    "Unexpectedly Tight Constraints",
                    subject=constraint,
                    value=c_senss,
                )
                if self.raiseerror:
                    raise RuntimeWarning(
                        f"{cstr} is not loose: it has a sensitivity of"
                        f" {c_senss:+.4g}. (Allowable: {self.senstol:.4g})"
                    )
