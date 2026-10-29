"""Defines the VarKey class"""

from dataclasses import dataclass, field, replace
from typing import Any

from .units import qty
from .util.lineage import lineagestr, segments
from .util.small_classes import Count


def marks_vector(idx, shape, excluded) -> bool:
    'Whether a name stands for a whole vector: "[:]" in text, bold in latex.'
    return bool(shape) and not idx and "vec" not in excluded and "idx" not in excluded


def decorated(name: str, idx=None, shape=(), excluded=()) -> str:
    "name with the index suffix a vector element or vector parent is written with."
    if "idx" in excluded:
        return name
    if idx:
        return name + f"[{','.join(map(str, idx))}]"
    if marks_vector(idx, shape, excluded):
        return name + "[:]"
    return name


@dataclass(frozen=True, eq=False)
class VarKey:
    """An object to correspond to each 'variable name'.

    Arguments
    ---------
    name : str
        Name of this Variable.

    **kwargs :
        lineage, units, label, idx, shape, veckey, value, choices

    Returns
    -------
    VarKey with the given name and attributes.
    """

    unique_id = Count().next

    # Init fields
    name: str = ""
    lineage: tuple = None
    units: Any = None
    unitrepr: str = ""  # preserved through replace(); set from units if empty
    label: str = ""
    idx: tuple = None
    shape: tuple = ()
    veckey: "VarKey" = None
    value: Any = None
    choices: tuple = None
    growth: float = None

    # Derived fields (computed in __post_init__)
    key: "VarKey" = field(default=None, init=False, repr=False)
    keys: frozenset = field(default=frozenset(), init=False, repr=False)
    ref: str = field(default="", init=False, repr=False)
    _hashvalue: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        # Normalize name
        name = self.name or "\\fbox{%s}" % VarKey.unique_id()
        object.__setattr__(self, "name", name)

        # Normalize lineage (None → ())
        if self.lineage is None:
            object.__setattr__(self, "lineage", ())

        # Normalize units and derive unitrepr (if not already set)
        if self.units in ("", "-", None):
            object.__setattr__(self, "units", None)
            if not self.unitrepr:
                object.__setattr__(self, "unitrepr", "-")
        else:
            # Capture original string before normalizing (if unitrepr not set)
            if not self.unitrepr and isinstance(self.units, str):
                object.__setattr__(self, "unitrepr", self.units)
            units = qty(self.units)
            object.__setattr__(self, "units", units)
            # If still no unitrepr (units was already a Quantity), derive it.
            # Full names, not symbols: a symbol need not parse back ("$" doesn't),
            # and unitrepr is what to_ir() and to_toml() hand out.
            if not self.unitrepr:
                object.__setattr__(self, "unitrepr", f"{units.units:C}")

        # Set key = self
        object.__setattr__(self, "key", self)

        # Compute ref and hash
        ref = self._compute_ref()
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "_hashvalue", hash(ref))

        # The strings a lookup may name this key by (VarSet.resolve)
        path = self._path(modelnums=False)
        keys = {self.name, decorated(path, self.idx, self.shape, {"vec"})}

        # Auto-create veckey if idx is set but veckey is not
        if self.idx is not None:
            if self.veckey is None:
                vk = replace(self, idx=None, veckey=None)
                object.__setattr__(self, "veckey", vk)
            else:
                keys.add(self.veckey)
                keys.add(path)

        object.__setattr__(self, "keys", frozenset(keys))

    def __reduce__(self):
        """Pickle support: serialize unitrepr (string) not units (Quantity)."""
        value = self.value
        if callable(value):
            value = f"unpickleable function {value}"
        state = {
            "name": self.name,
            "lineage": self.lineage or None,
            "units": self.unitrepr if self.unitrepr != "-" else None,
            "label": self.label or None,
            "idx": self.idx,
            "shape": self.shape or None,
            "value": value,
            "choices": self.choices,
            "growth": self.growth,
        }
        return (VarKey, (), {k: v for k, v in state.items() if v is not None})

    def __setstate__(self, state):
        """Unpickle: reconstruct VarKey from state dict."""
        new_vk = VarKey(**state)
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, getattr(new_vk, name))

    def _path(self, modelnums=True):
        "The full dotted path to this key's name: Aircraft.Wing.S."
        return ".".join([*segments(self.lineage, modelnums), self.name])

    def __str__(self):
        "The full path and index, Aircraft.Wing.c[1]; display.naming names it in context."
        return decorated(self._path(), self.idx, self.shape)

    __repr__ = __str__

    def lineagestr(self, modelnums=True):
        "The dotted path to the model this key belongs to."
        return lineagestr(self.lineage, modelnums)

    def _compute_ref(self):
        "Canonical identity string: [lineage.]name[idx][#shape][|units]"
        ref = ".".join([*segments(self.lineage), self.name])
        if self.idx:
            ref += f"[{','.join(map(str, self.idx))}]"
        if self.shape:
            ref += f"#{','.join(map(str, self.shape))}"
        if self.units is not None:
            ref += f"|{self.units.units:~}"  # canonical format for identity
        return ref

    def __hash__(self):
        return self._hashvalue

    def to_ir(self):
        "Serialize this VarKey to an IR dict."
        ir = {"name": self.name}
        if self.lineage:
            ir["lineage"] = [list(pair) for pair in self.lineage]
        # As declared, not canonicalized: this is what to_toml() writes back out,
        # and a spelling pint accepted on the way in it can read on the way back.
        if self.unitrepr and self.unitrepr != "-":
            ir["units"] = self.unitrepr
        if self.label:
            ir["label"] = self.label
        if self.idx is not None:
            ir["idx"] = list(self.idx)
            if self.veckey is not None:
                ir["veckey_ref"] = self.veckey.ref
        if self.shape:
            ir["shape"] = list(self.shape)
        return ir

    @property
    def models(self):
        "Returns a tuple of just the names of models in self.lineage"
        if not self.lineage:
            return ()
        return next(zip(*self.lineage))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.ref == other
        if not isinstance(other, VarKey):
            return NotImplemented
        return self.ref == other.ref
