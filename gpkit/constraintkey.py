"""Defines the ConstraintKey class"""

from dataclasses import dataclass, field

from .util.repr_conventions import ReprMixin


@dataclass(frozen=True, eq=False)
class ConstraintKey(ReprMixin):
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
    constraint, so *index* -- its position among the constraints of whatever
    holds it -- stands in.  That is a placeholder for a name, not a claim that
    constraints are fundamentally numbered: if they become nameable, the name
    renders in the index's place and consumers, which only compare refs, do not
    notice.
    """

    path: tuple = ()
    index: int = 0

    ref: str = field(default="", init=False, repr=False)
    _hashvalue: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        ref = f"{'.'.join(self.path)}[{self.index}]"
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "_hashvalue", hash(ref))

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

    def to_ir(self):
        "Serialize this ConstraintKey to an IR dict."
        return {"path": list(self.path), "index": self.index}

    @classmethod
    def from_ir(cls, ir_dict):
        "Reconstruct a ConstraintKey from an IR dict."
        return cls(path=tuple(ir_dict.get("path", ())), index=ir_dict["index"])
