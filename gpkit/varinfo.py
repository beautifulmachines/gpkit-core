"""The variables of a posed problem: what role each plays, and its values.

A model plus its substitutions poses a problem, and that partitions the model's
variables into the ones the solver chooses and the ones prescribed for it. This
module computes that partition and pairs it with values, once, so no consumer
has to re-derive it.
"""

from dataclasses import dataclass
from enum import Enum

from .nomials.substitution import parse_linked, parse_subs
from .varkey import VarKey


class VarKind(Enum):
    "The role a variable plays in a posed problem."

    FREE = "free"  # the solver chooses it
    FIXED = "fixed"  # prescribed by a substitution
    LINKED = "linked"  # computed from other constants at solve time
    MIXED = "mixed"  # a vector whose elements are not all the same kind


@dataclass(frozen=True, slots=True)
class VarInfo:
    """One variable of a posed problem, with its keys, role and values.

    `key` is what names the variable -- a veckey for a vector, the key itself for
    a scalar -- while `vks` is what the solver sees: the element keys in index
    order, or the single key of a scalar. `values` and `sensitivities` are
    positional against `vks`, with None where there is none.

    Carries no name, LaTeX or unit string: those depend on the output surface and
    belong to a renderer (#319). Spell `key` with a DisplayScope to get them.
    """

    key: VarKey
    vks: tuple
    kind: VarKind
    values: tuple
    sensitivities: tuple


def _element_keys(vks, key):
    "The solver-visible keys `key` stands for, in index order."
    if key.shape and key.idx is None:
        present = (vk for vk in vks.by_vec(key).flat if vk is not None)
        return tuple(sorted(present, key=lambda vk: vk.idx))
    return (key,)


def _kind(elements, fixed, linked):
    "One kind for the whole variable, or MIXED where its elements differ."
    kinds = {
        VarKind.FIXED
        if vk in fixed
        else VarKind.LINKED
        if vk in linked
        else VarKind.FREE
        for vk in elements
    }
    if len(kinds) == 1:
        return kinds.pop()
    return VarKind.MIXED


def _lookup(mapping, key):
    "The plain magnitude stored for key, or None if there is none."
    if mapping is None or key not in mapping:
        return None
    value = mapping[key]
    value = getattr(value, "magnitude", value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_variables(vks, substitutions, solution=None) -> list:
    """The VarInfo for every variable in `vks`, ordered by name then ref.

    Name first because a ref sorts on its separators: `#shape` puts "CD" before
    "CDA" while `|units` puts "CDA" before "CD", so ref order alone would order
    two variables differently according to their shapes. Ref breaks ties.

    `substitutions` determines each variable's kind, through the same
    `parse_subs`/`parse_linked` the solver uses -- so a kind here cannot disagree
    with what was solved, and passing `solution` never changes one.

    `solution` only supplies data. Without it, a fixed variable's values come
    from `substitutions` and there are no sensitivities. With it, values come
    from `solution.variables` and sensitivities from `solution.sens.variables`;
    nothing else on it is read.
    """
    fixed = parse_subs(vks, substitutions)
    linked = parse_linked(vks, substitutions)
    values = solution.variables if solution is not None else fixed
    sens = solution.sens.variables if solution is not None else None

    infos = []
    for key in vks.vector_parent_keys():
        elements = _element_keys(vks, key)
        infos.append(
            VarInfo(
                key=key,
                vks=elements,
                kind=_kind(elements, fixed, linked),
                values=tuple(_lookup(values, vk) for vk in elements),
                sensitivities=tuple(_lookup(sens, vk) for vk in elements),
            )
        )
    return sorted(infos, key=lambda info: (info.key.name, info.key.ref))
