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
"""

from collections import defaultdict
from dataclasses import dataclass, field

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


def _render(ascent: int, descent: list[str], name: str) -> str:
    """Join an ascent count, descent segments, and a variable name into a path.

    An ascent of N is written as a segment of N dots, so joining on "." yields
    the familiar relative-path prefix: one level up is "..name".
    """
    parts = ["." * ascent] if ascent else []
    parts.extend(descent)
    parts.append(name)
    return ".".join(parts)


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

    Whether a variable is *owned* by this context is derived, not passed: it is
    owned exactly when its lineage is at or below ``anchor``.  Keeping that
    derived is what makes a display-set/collision-scope mismatch unrepresentable.
    """

    anchor: Lineage = ()
    shown: frozenset = frozenset()

    _paths: dict = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        object.__setattr__(self, "anchor", tuple(self.anchor or ()))
        object.__setattr__(self, "shown", frozenset(self.shown or ()))
        object.__setattr__(self, "_paths", self._resolve())

    # -- the naming rule ---------------------------------------------------

    def _relative(self, vk) -> tuple[int, Lineage]:
        "Path of vk relative to the anchor, as (ascent, descent lineage)."
        lineage = vk.lineage or ()
        shared = 0
        while (
            shared < len(self.anchor)
            and shared < len(lineage)
            and self.anchor[shared] == lineage[shared]
        ):
            shared += 1
        return len(self.anchor) - shared, lineage[shared:]

    def owns(self, vk) -> bool:
        "Whether vk lives at or below this scope's anchor."
        return self._relative(vk)[0] == 0

    def _resolve(self) -> dict:
        """Shortest unambiguous path for every key in `shown`.

        Returns {canonical key: (ascent, kept descent lineage)} so that text and
        latex render the same resolved path rather than re-deriving it.  Vector
        elements resolve through their parent veckey, so siblings share one name
        and never count as colliding with each other.
        """
        full = {}  # canonical key -> (ascent, descent lineage)
        for vk in self.shown:
            key = vk.veckey or vk
            if key not in full:
                full[key] = self._relative(key)

        # Ascending paths are fixed at full length; only descendants shorten.
        # They cannot collide with each other: an ascending name starts with a
        # dot and a descendant's never does.
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

    # -- the name-resolver protocol ----------------------------------------

    def name(self, vk) -> str:
        """Display name for vk: its path relative to the anchor, shortened.

        Keys outside `shown` get their full relative path — nothing local
        licenses shortening a name this scope was not told about.
        """
        key = vk.veckey or vk
        ascent, descent = self._paths.get(key) or self._relative(key)
        return _render(ascent, _segments(descent), key.name)

    def latex(self, vk) -> str:
        """Latex name for vk, with its relative path as a subscript.

        Mirrors ``VarKey.latex``: model numbers are dropped and segments are
        lowercased.  An ascent of N renders as a run of N+1 dots, matching the
        leading ".." of the text form.
        """
        key = vk.veckey or vk
        ascent, descent = self._paths.get(key) or self._relative(key)
        name = latexify(key.name)
        if not ascent and not descent:
            return name
        sub = [r"\text{" + "." * (ascent + 1) + "}"] if ascent else []
        sub += [
            r"\text{" + seg.lower() + "}" for seg in _segments(descent, modelnums=False)
        ]
        return merge_subscript(name, ",".join(sub))
