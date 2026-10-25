"Implements Loose"

from ..display.render import TEXT, render
from ..util.small_scripts import appendsolwarning
from .set import ConstraintSet


class Loose(ConstraintSet):
    "ConstraintSet whose inequalities must result in an equality."

    senstol = 1e-5
    raiseerror = False

    def __init__(self, constraints, *, senstol=None):
        super().__init__(constraints)
        self.senstol = senstol or self.senstol

    def audit(self, result, keys):
        "Checks that every constraint's sensitivity is (nearly) zero"
        senss = result.sens.constraints_by_key
        for constraint in self.flat():
            key = keys.get(constraint)
            c_senss = senss.get(key, 0)
            if c_senss >= self.senstol:
                cstr = render(constraint, TEXT, {"units", "lineage"})
                appendsolwarning(
                    f"{c_senss:+6.2g} : {cstr}",
                    result,
                    "Unexpectedly Tight Constraints",
                    subject=key,
                    value=c_senss,
                )
                if self.raiseerror:
                    raise RuntimeWarning(
                        f"{cstr} is not loose: it has a sensitivity of"
                        f" {c_senss:+.4g}. (Allowable: {self.senstol:.4g})"
                    )
