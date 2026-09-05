"""Naming variables relative to a point in the model tree.

The model hierarchy is gpkit's only namespace: a VarKey's ``lineage`` is a path
in the model tree, and naming a variable for display is producing a path
relative to some node of that tree.  ``DisplayScope`` is that node plus the set
of variables being displayed alongside it, and it answers the single question:
*given this variable and this place I am showing it, what do I call it?*

One rule: a variable below the anchor is named by its path down from the
anchor, shortened until unambiguous among `shown`; any other variable is named
by its full path.  So a section anchored at ``Aircraft.Wing`` calls its own
``S`` just ``S``, a sub-model's variable ``Spar.t``, and one owned elsewhere
``Aircraft.Fuselage.Tank.m`` -- never a bare name, which would imply it was
local.

``abbreviate`` shortens the foreign names too, for renderers that publish
``legend()`` alongside to say where they live.

A scope also carries ``excluded``, the format flags saying which parts to show.
Naming and formatting stay separate concerns, but share one lifetime and one
call path, so a scope is what gets threaded through rendering.
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


def _kept(path: Lineage, depth: int) -> Lineage:
    "The last `depth` segments of a path."
    return path[len(path) - depth :] if depth else ()


def _contains(anchor: Lineage, lineage: Lineage) -> bool:
    "Whether `lineage` names a node at or below `anchor`."
    return lineage[: len(anchor)] == anchor


@lru_cache(maxsize=256)
def _resolve(anchor: Lineage, shown: frozenset, abbreviate: bool) -> dict:
    """Displayed path for every key in `shown`, as {canonical key: lineage}.

    Vector elements resolve through their parent veckey, so siblings share one
    name and never count as colliding with each other.  Memoized because a
    section renders many constraints against one scope.
    """
    paths, shortenable = {}, {}
    for vk in shown:
        key = vk.veckey or vk
        if key in paths:
            continue
        lineage = key.lineage or ()
        if _contains(anchor, lineage):
            paths[key] = lineage[len(anchor) :]
            shortenable[key] = True
        else:
            paths[key] = lineage
            shortenable[key] = abbreviate

    resolved = {k: v for k, v in paths.items() if not shortenable[k]}
    depths = dict.fromkeys((k for k in paths if shortenable[k]), 0)

    while True:
        groups = defaultdict(list)
        for key, depth in depths.items():
            shown_as = _segments(_kept(paths[key], depth)) + [key.name]
            groups[tuple(shown_as)].append(key)
        widened = False
        for keys in groups.values():
            if len(keys) == 1:
                continue
            for key in keys:  # widen every member, not just one
                if depths[key] < len(paths[key]):
                    depths[key] += 1
                    widened = True
        if not widened:
            break

    for key, depth in depths.items():
        resolved[key] = _kept(paths[key], depth)
    return resolved


@dataclass(frozen=True)
class DisplayScope:
    """How to name variables displayed together under one node of the model tree.

    Arguments
    ---------
    anchor : tuple
        Lineage of the model-tree node this display sits at.  ``()`` contains
        everything, reducing the rule to "shortest unambiguous suffix".
    shown : iterable of VarKey
        The VarKeys displayed together here, which set how much of each path
        is needed to stay unambiguous.
    excluded : iterable of str
        Format flags ("units", "idx", "vec", "lineage", "modelnums", ...).  A
        scope goes anywhere a bare set of these does, and supports ``in``.
    abbreviate : bool
        Shorten variables the anchor does not contain.  Set only when
        rendering ``legend()`` nearby, which says where they live.

    Ownership is derived from `anchor`, not passed, so it cannot disagree with
    the shortening about what this section contains.
    """

    anchor: Lineage = ()
    shown: frozenset = frozenset()
    excluded: frozenset = frozenset()
    abbreviate: bool = False

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
        return _contains(self.anchor, vk.lineage or ())

    def path(self, vk) -> Lineage:
        "Lineage segments shown before vk's name; unshown keys get a full path."
        key = vk.veckey or vk
        resolved = _resolve(self.anchor, self.shown, self.abbreviate).get(key)
        if resolved is not None:
            return resolved
        lineage = key.lineage or ()
        return lineage[len(self.anchor) :] if self.owns(key) else lineage

    def legend(self) -> dict:
        "{display name: full dotted path} for variables the anchor lacks."
        entries = {}
        for vk in self.shown:
            key = vk.veckey or vk
            if self.owns(key):
                continue
            entries[self.name(key)] = ".".join(
                _segments(key.lineage or ()) + [key.name]
            )
        return entries

    # -- the name-resolver protocol ----------------------------------------

    def name(self, vk) -> str:
        "Display name for vk; index decoration is the caller's to append."
        key = vk.veckey or vk
        if "lineage" in self.excluded:
            return key.name
        segments = _segments(self.path(key), "modelnums" not in self.excluded)
        return ".".join(segments + [key.name])

    def latex_path(self, vk) -> str:
        'vk\'s resolved path as latex subscript content, or "" if it has none.'
        key = vk.veckey or vk
        if "lineage" in self.excluded:
            return ""
        segments = _segments(self.path(key), modelnums=False)
        return ",".join(r"\text{" + seg.lower() + "}" for seg in segments)

    def latex(self, vk) -> str:
        """Latex name for vk, path in the subscript.

        The subscript is latex's prefix slot, so this composes in ``name``'s
        order: path, name, then the index the caller appends.
        """
        key = vk.veckey or vk
        name = latexify(key.name)
        sub = self.latex_path(key)
        return merge_subscript(name, sub) if sub else name
