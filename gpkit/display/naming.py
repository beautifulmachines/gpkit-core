"""Naming variables relative to a point in the model tree.

The model hierarchy is gpkit's only namespace: a VarKey's ``lineage`` is a path
in the model tree, and naming a variable for display is producing a path
relative to some node of that tree.  ``DisplayScope`` is that node plus the set
of variables being displayed alongside it, and it answers the single question:
*given this variable and this place I am showing it, what do I call it?*

One rule::

    render vk's path relative to `anchor`, then shorten it to the shortest
    form that stays unambiguous among `shown`

A variable at or below the anchor renders as a suffix of its descent path
(``t``, ``Spar.t``).  A variable outside the anchor renders as an ascending
relative path (``..Fuselage.Tank.m``), which is never shortened: a shortened
suffix still resolves under the anchor, but an ascending path with segments
removed denotes a different node.  So a foreign variable never renders as a
bare name, however unique that name happens to be.

A scope also carries ``excluded``, the format flags saying which parts to show.
Naming policy and format flags stay separate concerns, but they share one
lifetime and one call path -- a renderer that needs one needs the other -- so a
scope is what gets threaded through rendering.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache

from ..util.repr_conventions import latexify, merge_subscript

Lineage = tuple[tuple[str, int], ...]


def _segments(lineage: Lineage, modelnums: bool = True) -> list[str]:
    "Lineage tuple -> display segments, e.g. Aircraft, Wing2."
    return [
        f"{name}{num}" if (num and modelnums) else name for name, num in lineage or ()
    ]


def _kept(descent: Lineage, depth: int) -> Lineage:
    "The last `depth` segments of a descent path."
    return descent[len(descent) - depth :] if depth else ()


def _relative(anchor: Lineage, lineage: Lineage) -> tuple[int, Lineage]:
    "Path of `lineage` relative to `anchor`, as (ascent, descent lineage)."
    shared = 0
    while (
        shared < len(anchor)
        and shared < len(lineage)
        and anchor[shared] == lineage[shared]
    ):
        shared += 1
    return len(anchor) - shared, lineage[shared:]


@lru_cache(maxsize=256)
def _resolve(anchor: Lineage, shown: frozenset) -> dict:
    """Shortest unambiguous path for every key in `shown`.

    Returns {canonical key: (ascent, kept descent lineage)} so that text and
    latex render the same resolved path rather than re-deriving it.  Vector
    elements resolve through their parent veckey, so siblings share one name
    and never count as colliding with each other.

    Memoized on (anchor, shown): a section renders many constraints against one
    scope, and scopes differing only in format flags share a resolution.
    """
    full = {}  # canonical key -> (ascent, descent lineage)
    for vk in shown:
        key = vk.veckey or vk
        if key not in full:
            full[key] = _relative(anchor, key.lineage or ())

    # Ascending paths are fixed at full length; only descendants shorten.  The
    # two never collide: an ascending name starts with a dot, a descendant's
    # never does.
    resolved = {key: path for key, path in full.items() if path[0]}
    descents = {key: full[key][1] for key in full if key not in resolved}
    depths = dict.fromkeys(descents, 0)

    while True:
        groups = defaultdict(list)
        for key, depth in depths.items():
            shown_as = _segments(_kept(descents[key], depth)) + [key.name]
            groups[tuple(shown_as)].append(key)
        widened = False
        for keys in groups.values():
            if len(keys) == 1:
                continue
            for key in keys:  # widen every member, not just one
                if depths[key] < len(descents[key]):
                    depths[key] += 1
                    widened = True
        if not widened:
            break

    for key, depth in depths.items():
        resolved[key] = (0, _kept(descents[key], depth))
    return resolved


@dataclass(frozen=True)
class DisplayScope:
    """How to name variables displayed together under one node of the model tree.

    Arguments
    ---------
    anchor : tuple
        Lineage tuple of the model-tree node this display context sits at.
        Paths are rendered relative to it.  ``()`` (the default) anchors at the
        root, reducing the rule to "shortest unambiguous suffix".
    shown : iterable of VarKey
        The VarKeys displayed together here; determines how much of each path
        is needed to stay unambiguous.
    excluded : iterable of str
        Format flags -- which parts to render ("units", "idx", "vec",
        "lineage", "modelnums", ...).  A scope is accepted anywhere a bare set
        of these flags is, and supports ``in`` for the same reason.

    Whether a variable is *owned* by this context is derived, not passed: it is
    owned exactly when its lineage is at or below ``anchor``.  Ownership and
    shortening therefore cannot disagree about what a section contains.
    """

    anchor: Lineage = ()
    shown: frozenset = frozenset()
    excluded: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "anchor", tuple(self.anchor or ()))
        object.__setattr__(self, "shown", frozenset(self.shown or ()))
        object.__setattr__(self, "excluded", frozenset(self.excluded or ()))

    # -- format-flag protocol, so a scope goes wherever a flag set went ------

    def __contains__(self, flag) -> bool:
        return flag in self.excluded

    def __iter__(self):
        return iter(self.excluded)

    def also_excluding(self, *flags) -> "DisplayScope":
        "This scope with additional format flags set."
        return replace(self, excluded=self.excluded.union(flags))

    # -- the naming rule ---------------------------------------------------

    def owns(self, vk) -> bool:
        "Whether vk lives at or below this scope's anchor."
        return _relative(self.anchor, vk.lineage or ())[0] == 0

    def path(self, vk) -> tuple[int, Lineage]:
        """Resolved (ascent, descent lineage) for vk.

        Keys outside `shown` get their full relative path -- nothing local
        licenses shortening a name this scope was not told about.
        """
        key = vk.veckey or vk
        resolved = _resolve(self.anchor, self.shown).get(key)
        if resolved is not None:
            return resolved
        return _relative(self.anchor, key.lineage or ())

    # -- the name-resolver protocol ----------------------------------------

    def name(self, vk) -> str:
        """Display name for vk: its path relative to the anchor, shortened.

        The path is a prefix, as in ``Spar.t``; any index decoration is the
        caller's to append, so that naming and formatting stay separable.
        """
        key = vk.veckey or vk
        if "lineage" in self.excluded:
            return key.name
        ascent, descent = self.path(key)
        parts = ["." * ascent] if ascent else []
        parts.extend(_segments(descent, "modelnums" not in self.excluded))
        parts.append(key.name)
        return ".".join(parts)

    def latex_path(self, vk) -> str:
        """vk's relative path as latex subscript content, or "" if it has none.

        Segments are lowercased and model numbers dropped; an ascent of N
        renders as a run of N+1 dots, matching text's "..".
        """
        key = vk.veckey or vk
        if "lineage" in self.excluded:
            return ""
        ascent, descent = self.path(key)
        if not ascent and not descent:
            return ""
        sub = [r"\text{" + "." * (ascent + 1) + "}"] if ascent else []
        sub += [
            r"\text{" + seg.lower() + "}" for seg in _segments(descent, modelnums=False)
        ]
        return ",".join(sub)

    def latex(self, vk) -> str:
        """Latex name for vk, with its relative path as a subscript.

        The subscript is latex's prefix slot, so this composes in the same
        order as ``name``.  Index decoration is the caller's to append.
        """
        key = vk.veckey or vk
        name = latexify(key.name)
        sub = self.latex_path(key)
        return merge_subscript(name, sub) if sub else name
