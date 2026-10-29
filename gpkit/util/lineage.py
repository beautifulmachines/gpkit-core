"""A lineage: where a variable or constraint sits in the model tree.

A lineage is a tuple of (model name, instance number) pairs, root first.  It is
part of identity -- refs, instance ids and per-model sensitivities are all
spelled from it -- so its one spelling lives here, not in the display layer.
"""

Lineage = tuple[tuple[str, int], ...]


def segments(lineage, modelnums=True) -> list[str]:
    "Each model on a lineage as written: Wing, or Wing2 for a numbered instance."
    return [
        f"{name}{num}" if num and modelnums else name for name, num in lineage or ()
    ]


def lineagestr(lineage, modelnums=True) -> str:
    "A lineage, or anything that has one, as a dotted path: Aircraft.Wing2."
    if not isinstance(lineage, tuple):
        lineage = getattr(lineage, "lineage", None)
    return ".".join(segments(lineage, modelnums))
