"Implement CostedConstraintSet"

from collections import namedtuple

import numpy as np

from ..nomials import Variable
from ..util.repr_conventions import lineagestr, unitstr
from ..util.small_scripts import maybe_flatten
from .set import ConstraintSet

Objective = namedtuple("Objective", ["sense", "expr", "variable", "units", "value"])


def _reciprocal_if_1_over_x(cost):
    """Return (True, inner_expr) if cost is 1/x form, else (False, cost).

    Detects the pattern coeff=1, all-negative exponents — the 1/x form that
    GPs use to express maximization.
    """
    hmap = cost.hmap
    if len(hmap) != 1:
        return False, cost
    (exp,) = hmap.keys()
    coeff = hmap[exp]
    exp_dict = dict(exp)
    if abs(coeff - 1.0) < 1e-10 and exp_dict and all(v < 0 for v in exp_dict.values()):
        # Build the inner expression from VarKeys rather than computing 1/cost.
        # 1/cost encodes div(1, cost.ast) in its AST, causing str/latex to
        # render as 1/(1/x) instead of x.
        inner = None
        for vk, e in exp_dict.items():
            term = Variable(vk) if e == -1 else Variable(vk) ** (-e)
            inner = term if inner is None else inner * term
        return True, inner
    return False, cost


class CostedConstraintSet(ConstraintSet):
    """A ConstraintSet with a cost

    Arguments
    ---------
    cost : gpkit.Posynomial
    constraints : Iterable
    substitutions : dict
    """

    lineage = None

    def __init__(self, cost, constraints, substitutions=None):
        self.cost = maybe_flatten(cost)
        if isinstance(self.cost, np.ndarray):  # if it's still a vector
            raise ValueError("Cost must be scalar, not the vector {cost}.")
        subs = {k: k.value for k in self.cost.vks if k.value is not None}
        if substitutions:
            subs.update(substitutions)
        ConstraintSet.__init__(self, constraints, subs, bonusvks=self.cost.vks)

    def constrained_varkeys(self):
        "Return all varkeys in the cost and non-ConstraintSet constraints"
        constrained_varkeys = ConstraintSet.constrained_varkeys(self)
        constrained_varkeys.update(self.cost.vks)
        return constrained_varkeys

    def _rootlines(self, excluded=()):
        "String showing cost, to be used when this is the top constraint"
        if self.cost.vks:
            description = [
                "",
                "Cost Function",
                "-------------",
                " " + self.cost.str_without(excluded),
                "",
                "Constraints",
                "-----------",
            ]
        else:  # don't print the cost if it's a constant
            description = ["", "Constraints", "-----------"]
        if self.lineage:
            fullname = lineagestr(self)
            description = [fullname, "=" * len(fullname)] + description
        return description

    def _rootlatex(self, excluded=()):
        "Latex showing cost, to be used when this is the top constraint"
        return "\n".join(
            [
                "\\text{minimize}",
                f"    & {self.cost.latex(excluded)} \\\\",
                "\\text{subject to}",
            ]
        )

    def objective_info(self, solution=None):
        """Return an Objective namedtuple describing this model's cost.

        Fields:
          sense    -- "minimize" or "maximize"
          expr     -- the expression being optimized: cost itself for minimize,
                      or the inner x extracted from a 1/x cost for maximize
          variable -- VarKey if expr is a single variable, else None
          units    -- unit string for expr
          value    -- attained float at solution (the value of expr, not
                      solution.cost), or None

        Assumes cost is var or 1/var (single-variable case).
        """
        if not self.cost.vks:
            return Objective(
                sense="minimize", expr=self.cost, variable=None, units="", value=None
            )
        is_recip, expr = _reciprocal_if_1_over_x(self.cost)
        sense = "maximize" if is_recip else "minimize"
        vks = list(expr.vks)
        variable = vks[0] if len(vks) == 1 else None
        units = unitstr(expr)
        value = None
        if solution is not None:
            value = 1.0 / float(solution.cost) if is_recip else float(solution.cost)
        return Objective(
            sense=sense, expr=expr, variable=variable, units=units, value=value
        )

    @property
    def n_free(self):
        """Number of variables not fixed by substitutions."""
        return len(self.vks) - len(self.substitutions)

    @property
    def n_constraints(self):
        """Total number of constraints in this model and its children."""
        return sum(1 for _ in self.flat())
