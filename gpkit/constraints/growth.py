"""Growth allowance machinery.

Variables declared with ``growth=<fraction>`` participate in a uniform
allowance scheme: ``m.grown_from(expr)`` emits the bookkeeping constraints
that bound ``m`` from below by ``expr`` plus an allowance variable
``m_growth``, where ``m_growth >= theta * f_growth_m * expr`` and ``theta``
is a single shared scalar (default-substituted to 1.0) that lets future
models scale all allowances uniformly.
"""

from ..nomials.variables import Variable
from ..varkey import VarKey
from .tight import Tight


class GrowthAllowanceConstraints(Tight):
    """Typed marker for the constraint group emitted by ``grown_from``.

    Subclassing ``Tight`` means the slack-warning machinery fires if either
    of the two growth constraints isn't tight at the solution.
    """


class GrowthAllowance:
    """Owns the shared theta variable and builds growth constraint sets."""

    _theta = None

    @classmethod
    def theta(cls):
        "Return the singleton theta Variable (lineage='growth', value=1.0)."
        if cls._theta is None:
            cls._theta = Variable(
                VarKey(name="theta", lineage=(("growth", 0),), value=1.0)
            )
        return cls._theta

    @classmethod
    def make_constraints(cls, parent, expr):
        "Build the GrowthAllowanceConstraints set for parent variable + expr."
        parent_vk = parent.key
        if parent_vk.growth is None:
            raise ValueError(
                f"{parent_vk!r} was not declared with a growth allowance; "
                "construct it as Variable(..., growth=<fraction>) first"
            )
        growth = sibling_growth(parent_vk)
        f_growth = sibling_fraction(parent_vk)
        theta = cls.theta()
        return GrowthAllowanceConstraints(
            [growth >= theta * f_growth * expr, parent >= expr + growth]
        )


def sibling_growth(parent_vk):
    "The allowance Variable for a growth-enabled parent VarKey."
    return Variable(
        VarKey(
            name=f"{parent_vk.name}_growth",
            lineage=parent_vk.lineage,
            units=parent_vk.units,
        )
    )


def sibling_fraction(parent_vk):
    "The dimensionless fraction Variable for a growth-enabled parent VarKey."
    return Variable(
        VarKey(
            name=f"f_growth_{parent_vk.name}",
            lineage=parent_vk.lineage,
            value=parent_vk.growth,
        )
    )


def _grown_from(self, expr):
    "Bound self below by expr plus its declared growth allowance."
    return GrowthAllowance.make_constraints(self, expr)


def _growth_sibling(self):
    "The allowance Variable sibling of this growth-enabled Variable."
    return sibling_growth(self.key)


def _f_growth_sibling(self):
    "The fraction Variable sibling of this growth-enabled Variable."
    return sibling_fraction(self.key)


Variable.grown_from = _grown_from
Variable.growth = property(_growth_sibling)
Variable.f_growth = property(_f_growth_sibling)
