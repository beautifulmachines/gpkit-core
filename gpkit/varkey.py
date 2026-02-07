"""Defines the VarKey class"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from .units import qty
from .util.repr_conventions import ReprMixin
from .util.small_classes import Count

_lineage_ctx: ContextVar[dict] = ContextVar("lineage_ctx", default={})


@contextmanager
def lineage_display_context(mapping):
    """Set VarKey→int lineage display depth for the enclosed block."""
    token = _lineage_ctx.set(mapping)
    try:
        yield
    finally:
        _lineage_ctx.reset(token)


def necessarylineage(vk):
    """Display lineage depth for a VarKey in the current context."""
    return _lineage_ctx.get().get(vk)


@dataclass(frozen=True, eq=False)
class VarKey(ReprMixin):  # pylint:disable=too-many-instance-attributes
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
    subscripts = ("lineage", "idx")

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
            # If still no unitrepr (units was already a Quantity), derive it
            if not self.unitrepr:
                object.__setattr__(self, "unitrepr", f"{units.units:~}")

        # Set key = self
        object.__setattr__(self, "key", self)

        # Compute ref and hash
        ref = self._compute_ref()
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "_hashvalue", hash(ref))

        # Compute keys set
        fullstr = self.str_without({"hiddenlineage", "modelnums", "vec"})
        keys = {self.name, fullstr}

        # Auto-create veckey if idx is set but veckey is not
        if self.idx is not None:
            if self.veckey is None:
                vk = replace(self, idx=None, veckey=None)
                object.__setattr__(self, "veckey", vk)
            else:
                keys.add(self.veckey)
                keys.add(self.str_without({"idx", "modelnums"}))

        object.__setattr__(self, "keys", frozenset(keys))

    def __reduce__(self):
        """Pickle support: serialize unitrepr (string) not units (Quantity)."""
        value = self.value
        if callable(value):
            value = f"unpickleable function {value}"
        state = {
            "name": self.name,
            "lineage": self.lineage if self.lineage else None,
            "units": self.unitrepr if self.unitrepr != "-" else None,
            "label": self.label if self.label else None,
            "idx": self.idx,
            "shape": self.shape if self.shape else None,
            "value": value,
            "choices": self.choices,
        }
        state = {k: v for k, v in state.items() if v is not None}
        return (VarKey, (), state)

    def __setstate__(self, state):
        """Unpickle: reconstruct VarKey from state dict."""
        new_vk = VarKey(**state)
        # pylint: disable=no-member  # __dataclass_fields__ exists on dataclasses
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, getattr(new_vk, name))

    def str_without(self, excluded=()):  # pylint:disable=too-many-branches
        "Returns string without certain fields (such as 'lineage')."
        name = self.name
        if "lineage" not in excluded and self.lineage:
            namespace = self.lineagestr("modelnums" not in excluded).split(".")
            for ex in excluded:
                if ex[0:7] == ":MAGIC:":
                    to_replace = ex[7:]
                    if not to_replace:
                        continue
                    to_replace = to_replace.split(".")
                    replaced = 0
                    for modelname in to_replace:
                        if not namespace or namespace[0] != modelname:
                            break
                        replaced += 1
                        namespace = namespace[1:]
                    if len(to_replace) > replaced:
                        namespace.insert(0, "." * (len(to_replace) - replaced))
            if "hiddenlineage" not in excluded:
                lineage_map = _lineage_ctx.get()
                lineage_depth = lineage_map.get(self)
                if lineage_depth is None and self.veckey:
                    lineage_depth = lineage_map.get(self.veckey)
                if lineage_depth is not None:
                    if lineage_depth > 0:
                        namespace = namespace[-lineage_depth:]
                    else:
                        namespace = None
            if namespace:
                name = ".".join(namespace) + "." + name
        if "idx" not in excluded:
            if self.idx:
                name += f"[{','.join(map(str, self.idx))}]"
            elif "vec" not in excluded and self.shape:
                name += "[:]"
        return name

    __repr__ = str_without

    def _compute_ref(self):
        "Canonical identity string: [lineage.]name[idx][#shape][|units]"
        parts = []
        if self.lineage:
            parts.extend(f"{name}{num}" for name, num in self.lineage)
        parts.append(self.name)
        ref = ".".join(parts)
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
        if self.unitrepr and self.unitrepr != "-":
            ir["units"] = self.unitrepr
        if self.label:
            ir["label"] = self.label
        if self.idx is not None:
            ir["idx"] = list(self.idx)
        if self.shape:
            ir["shape"] = list(self.shape)
        return ir

    @classmethod
    def from_ir(cls, ir_dict):
        "Reconstruct a VarKey from an IR dict."
        name = ir_dict["name"]
        kwargs = {}
        if "lineage" in ir_dict:
            kwargs["lineage"] = tuple(tuple(pair) for pair in ir_dict["lineage"])
        if "units" in ir_dict:
            kwargs["units"] = ir_dict["units"]
        if "label" in ir_dict:
            kwargs["label"] = ir_dict["label"]
        if "idx" in ir_dict:
            kwargs["idx"] = tuple(ir_dict["idx"])
        if "shape" in ir_dict:
            kwargs["shape"] = tuple(ir_dict["shape"])
        return cls(name, **kwargs)

    @property
    def models(self):
        "Returns a tuple of just the names of models in self.lineage"
        if not self.lineage:
            return ()
        return tuple(zip(*self.lineage))[0]

    def latex(self, excluded=()):
        "Returns latex representation."
        name = self.name
        if "vec" not in excluded and "idx" not in excluded and self.shape:
            name = "\\vec{%s}" % name
        if "idx" not in excluded and self.idx:
            name = "{%s}_{%s}" % (name, ",".join(map(str, self.idx)))
        if "lineage" not in excluded and self.lineage:
            name = "{%s}_{%s}" % (name, self.lineagestr("modelnums" not in excluded))
        return name

    def __eq__(self, other):
        if not isinstance(other, VarKey):
            return False
        return self.ref == other.ref
