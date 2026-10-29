"""Defines the ConstraintKey class"""

from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class ConstraintKey:
    """Identifies one constraint by where it sits in a model.

    The constraint counterpart of VarKey, and the same shape: a path to the
    thing holding it, something to tell siblings apart, and a ref that equality
    and hashing run on -- so two builds of one model give keys that compare
    equal, which is what lets two solutions be compared.

    *path* names the containers the constraint sits in: the models above it, and
    the named group holding it if there is one.  It is a plain sequence because
    that nesting has no fixed depth -- models hold groups, groups hold models --
    so a path of any length renders the same way.

    A variable is told apart by its name.  Nothing names an individual
    constraint, so *index* -- its position among the constraints written in
    whatever holds it -- stands in.  That is a placeholder for a name, not a
    claim that constraints are fundamentally numbered: if they become nameable,
    the name renders in the index's place and consumers, which only compare
    refs, do not notice.

    A vector constraint is one written relation standing for one solver
    relation per element, so it has a VarKey's two levels: the written
    constraint carries a *shape*, and each element adds its *idx*.  The shape is
    part of the identity, as it is in a VarKey's ref: re-discretizing moves
    every point rather than adding one, so element 2 of 3 and element 2 of 4
    are different relations.
    """

    path: tuple = ()
    index: int = 0
    idx: tuple | None = None
    shape: tuple = ()

    ref: str = field(default="", init=False, repr=False)
    _hashvalue: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        ref = f"{'.'.join(self.path)}[{self.index}]"
        if self.idx is not None:
            ref += f"[{','.join(map(str, self.idx))}]"
        if self.shape:
            ref += f"#{','.join(map(str, self.shape))}"
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "_hashvalue", hash(ref))

    @property
    def veckey(self):
        "The vector constraint an element belongs to, as VarKey.veckey; else None."
        if self.idx is None:
            return None
        return ConstraintKey(self.path, self.index, shape=self.shape)

    def element(self, idx) -> "ConstraintKey":
        "The key of this vector constraint's element at idx."
        return ConstraintKey(self.path, self.index, tuple(idx), self.shape)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.ref == other
        if not isinstance(other, ConstraintKey):
            return NotImplemented
        return self.ref == other.ref

    def __hash__(self):
        return self._hashvalue

    def __str__(self):
        return self.ref

    def __repr__(self):
        return f"gpkit.ConstraintKey({self})"

    def to_ir(self):
        "Serialize this ConstraintKey to an IR dict."
        ir = {"path": list(self.path), "index": self.index}
        if self.idx is not None:
            ir["idx"] = list(self.idx)
            ir["veckey_ref"] = self.veckey.ref
        if self.shape:
            ir["shape"] = list(self.shape)
        return ir
