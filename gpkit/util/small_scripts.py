"""Assorted helper methods"""

from collections.abc import Iterable


def veclinkedfn(linkedfn, i):
    "Generate an indexed linking function."

    def newlinkedfn(c):
        "Linked function that pulls out a particular index"
        return linkedfn(c)[i]

    return newlinkedfn


def initsolwarning(result, category="uncategorized"):
    "Creates a results dictionary for a particular category of warning."
    if "warnings" not in result:
        result["warnings"] = {}
    if category not in result["warnings"]:
        result["warnings"][category] = []


def appendsolwarning(msg, data, result, category="uncategorized"):
    "Append a particular category of warnings to a solution."
    result["warnings"][category].append((msg, data))


def maybe_flatten(value):
    "Extract values from 0-d numpy arrays, if necessary"
    if hasattr(value, "size") and value.size == 1:
        return value.item()
    return value


def try_str_without(item, excluded, *, latex=False):
    "Try to call item.str_without(excluded); fall back to str(item)"
    if latex and hasattr(item, "latex"):
        return item.latex(excluded)
    if hasattr(item, "str_without"):
        return item.str_without(excluded)
    return str(item)


def mag(c):
    "Return magnitude of a Number or Quantity"
    return getattr(c, "magnitude", c)


def is_sweepvar(sub):
    "Determines if a given substitution indicates a sweep."
    return splitsweep(sub)[0]


def splitsweep(sub):
    "Splits a substitution into (is_sweepvar, sweepval)"
    try:
        sweep, value = sub
        if sweep == "sweep" and (
            isinstance(value, Iterable) or hasattr(value, "__call__")
        ):
            return True, value
    except (TypeError, ValueError):
        pass
    return False, None
