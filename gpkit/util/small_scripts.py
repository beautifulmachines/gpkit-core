"""Assorted helper methods"""


def veclinkedfn(linkedfn, i):
    "Generate an indexed linking function."

    def newlinkedfn(c):
        "Linked function that pulls out a particular index"
        return linkedfn(c)[i]

    newlinkedfn.original_fn = linkedfn
    newlinkedfn.idx = i
    return newlinkedfn


def appendsolwarning(
    message, result, category="uncategorized", subject=None, value=None
):
    """Record a warning on a solution.

    message is the line printed; subject is what the warning is about (a
    constraint, a variable, or None); value is the number that triggered it.
    """
    warnings = result.meta.setdefault("warnings", {})
    warnings.setdefault(category, []).append(
        {"message": message, "subject": subject, "value": value}
    )


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
