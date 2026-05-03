"""Growth allowance helpers (nomials-only layer).

Constructs the auto-generated sibling Variables (allowance + fraction) and
the shared theta singleton, plus a builder that returns the two bookkeeping
constraints as a plain list. The Tight-subclass wrapper that adds slack
diagnostics lives in :mod:`gpkit.constraints.growth` (a higher layer).
"""

from ..util.globals import NamedVariables
from ..varkey import VarKey
from .variables import Variable

_THETA_LINEAGE = (("growth", 0),)
_THETA = None  # singleton, lazy-initialized on first call to theta()


def theta():
    "The shared theta Variable (lineage 'growth', value=1.0)."
    global _THETA  # pylint: disable=global-statement
    if _THETA is None:
        _THETA = Variable(VarKey(name="theta", lineage=_THETA_LINEAGE, value=1.0))
    return _THETA


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


def make_growth_constraints(parent, expr):
    """Build the two bookkeeping constraints for a growth-enabled Variable.

    Returns a plain list ``[allowance_constraint, total_constraint]``. To
    additionally get slack-warning diagnostics, wrap with
    :class:`gpkit.constraints.growth.GrowthAllowanceConstraints` (or call
    ``GrowthAllowance.make_constraints(parent, expr)``).
    """
    parent_vk = parent.key
    if parent_vk.growth is None:
        raise ValueError(
            f"{parent_vk!r} was not declared with a growth allowance; "
            "construct it as Variable(..., growth=<fraction>) first"
        )
    growth = sibling_growth(parent_vk)
    f_growth = sibling_fraction(parent_vk)
    # Register siblings into the parent's setup-time variable list so they
    # appear in model.unique_varkeys (and thus in tools like apply_subs that
    # look up substitutable variables by name in unique_varkeys).
    if parent_vk.lineage in NamedVariables.namedvars:
        NamedVariables.namedvars[parent_vk.lineage].extend([growth, f_growth])
    th = theta()
    return [
        growth >= th * f_growth * expr,
        parent >= expr + growth,
    ]
