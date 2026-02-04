"""Defines the VarKey class"""

from .units import qty
from .util.repr_conventions import ReprMixin
from .util.small_classes import Count


class VarKey(ReprMixin):  # pylint:disable=too-many-instance-attributes
    """An object to correspond to each 'variable name'.

    Arguments
    ---------
    name : str
        Name of this Variable, or object to derive this Variable from.

    **descr :
        Any additional attributes, which become the descr attribute (a dict).

    Returns
    -------
    VarKey with the given name and descr.
    """

    unique_id = Count().next
    subscripts = ("lineage", "idx")

    _DESCR_DEFAULTS = {
        **dict.fromkeys(
            [
                "lineage",
                "value",
                "constant",
                "evalfn",
                "vecfn",
                "idx",
                "shape",
                "veckey",
                "necessarylineage",
                "choices",
                "gradients",
            ]
        ),
        "label": "",
    }

    def __init__(self, name=None, **descr):
        # NOTE: Python arg handling guarantees 'name' won't appear in descr
        self.descr = {**self._DESCR_DEFAULTS, **descr}
        self.descr["name"] = name or "\\fbox{%s}" % VarKey.unique_id()
        unitrepr = self.descr.get("unitrepr") or self.descr.get("units")
        if unitrepr in ["", "-", None]:  # dimensionless
            self.descr["units"] = None
            self.descr["unitrepr"] = "-"
        else:
            self.descr["units"] = qty(unitrepr)
            self.descr["unitrepr"] = unitrepr

        self.key = self
        self.ref = self._compute_ref()
        self._hashvalue = hash(self.ref)
        fullstr = self.str_without({"hiddenlineage", "modelnums", "vec"})
        self.keys = set((self.name, fullstr))

        if self.descr["idx"] is not None:
            if self.descr["veckey"] is None:
                vecdescr = self.descr.copy()
                vecdescr["idx"] = None
                self.veckey = VarKey(**vecdescr)
            else:
                self.keys.add(self.veckey)
                self.keys.add(self.str_without({"idx", "modelnums"}))

    def __getstate__(self):
        "Stores varkey as its metadata dictionary, removing functions"
        state = self.descr.copy()
        for key, value in state.items():
            if getattr(value, "__call__", None):
                state[key] = "unpickleable function {value}"
        return state

    def __setstate__(self, state):
        "Restores varkey from its metadata dictionary"
        self.__init__(**state)

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
                necessarylineage = self.necessarylineage
                if necessarylineage is None and self.veckey:
                    necessarylineage = self.veckey.necessarylineage
                if necessarylineage is not None:
                    if necessarylineage > 0:
                        namespace = namespace[-necessarylineage:]
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
        if self.unitrepr != "-":
            ref += f"|{self.unitrepr}"
        return ref

    def __hash__(self):
        return self._hashvalue

    def __getattr__(self, attr):
        if attr in self.descr:
            return self.descr[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

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
        descr = {}
        if "lineage" in ir_dict:
            descr["lineage"] = tuple(tuple(pair) for pair in ir_dict["lineage"])
        if "units" in ir_dict:
            descr["unitrepr"] = ir_dict["units"]
        if "label" in ir_dict:
            descr["label"] = ir_dict["label"]
        if "idx" in ir_dict:
            descr["idx"] = tuple(ir_dict["idx"])
        if "shape" in ir_dict:
            descr["shape"] = tuple(ir_dict["shape"])
        return cls(name, **descr)

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
