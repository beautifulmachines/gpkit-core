"Implements the VarMap class"

from collections.abc import MutableMapping

import numpy as np

from .units import Quantity
from .util.small_scripts import is_sweepvar, veclinkedfn


def _nested_lookup(nested_keys, val_dict):
    if nested_keys is None:
        return float("nan")
    if isinstance(nested_keys, np.ndarray):
        return [_nested_lookup(row, val_dict) for row in nested_keys]
    return val_dict[nested_keys]


def is_veckey(key):
    "return True iff this key corresponds to a VectorVariable"
    if getattr(key, "shape", None) and not getattr(key, "idx", None):
        # it has a shape but no index
        return True
    return False


class VarSet(set):
    """
    A lightweight *set* of :class:`gpkit.VarKey` objects with constant-time
    look-ups by **canonical name** or **vector parent (veckey)**.

    Unlike :class:`~gpkit.varmap.VarMap`, *VarSet stores no values — it is
    purely a membership container plus two derived indices:

    Attributes
    ----------
    _by_name : dict[str, set[VarKey]]
        Maps each canonical name string (``VarKey.name``) to the set of
        keys sharing that name. Maps to veckey, not elements, for vectors.
    _by_vec : dict[VarKey, numpy.ndarray]
        Maps a vector parent key (veckey) to an ``ndarray(dtype=object)``
        whose element at index *i* is the scalar key representing
        ``veckey[i]``.  Positions for which no element key has been
        registered contain ``None``.

    Invariants
    ----------
    * Every key in the VarSet appears exactly **once** in the underlying
      Python set and once in either index.
    * The indices are **derivative**—mutating the container via
      :py:meth:`add`, :py:meth:`update`, or :py:meth:`discard` keeps
      ``_by_name`` and ``_by_vec`` in sync automatically.
    * ``len(varset)`` equals the number of scalar VarKey instances
      currently registered (any parent veckeys are not included)

    See Also
    --------
    VarMap
        Mapping from VarKey → value that contains a VarSet internally.
    """

    def __init__(self, keys=()):
        super().__init__()
        self._by_name = {}
        self._by_vec = {}
        self.update(keys)

    def add(self, key):
        self._register_key(key)
        super().add(key)

    def discard(self, key):
        if key not in self:
            return
        name = key.name
        self._by_name[name].discard(key)
        if not self._by_name[name]:
            del self._by_name[name]
        # handle vector element case
        idx = getattr(key, "idx", None)
        if idx is not None:
            veckey = key.veckey
            self._by_vec[veckey][idx] = None
            if not self._by_vec[veckey].any():
                del self._by_vec[veckey]
        super().discard(key)

    def by_name(self, name):
        """Return all VarKeys for a given name string."""
        return set(self._by_name.get(name, set()))

    def by_vec(self, veckey):
        "Return np.array of keys for a given veckey"
        return np.array(self._by_vec.get(veckey, ()))

    def keys(self, key):
        "set of all keys in self that the input key refers to"
        key = getattr(key, "key", None) or key  # Variable case
        out = {key} if super().__contains__(key) else set()
        for k in self.by_name(key).union((key,) if is_veckey(key) else ()):
            if is_veckey(k):
                out.update(self.by_vec(k).flat)
                out.discard(None)  # present if vec elements missing
            else:
                out.add(k)
        return out

    def resolve(self, key):
        "Return *one* canonical VarKey for (Variable, str, or veckey) key"
        if hasattr(key, "key"):
            return key.key
        vks = self.by_name(key)
        if not vks:
            raise KeyError(f"unrecognized key {key}")
        if len(vks) == 1:
            (vk,) = vks
            return vk
        raise KeyError(f"{key} refers to multiple keys {vks}")

    def clean(self, mapping):
        "return a *new dict* with all keys in mapping resolved."
        return {self.resolve(k): v for k, v in mapping.items()}

    def update(self, keys):
        for k in keys:
            self.add(k)

    def vector_parent_keys(self):
        "set(self) but with veckeys and no individual vector element keys"
        ks = set(self)
        for vk, vks in self._by_vec.items():
            ks -= set(vks.flat)
            ks.add(vk)
        return ks

    def _register_key(self, key):
        "adds the key to _by_name and, if applicable, _by_vec"
        name = getattr(key, "name", None)
        if name is None:
            raise TypeError("VarSet keys must be VarKey instances")
        if name not in self._by_name:
            self._by_name[name] = set()
        idx = getattr(key, "idx", None)
        self._by_name[name].add(key if not idx else key.veckey)
        if idx is not None:
            if key.veckey not in self._by_vec:
                self._by_vec[key.veckey] = np.empty(key.shape, dtype=object)
            self._by_vec[key.veckey][idx] = key

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return key in self._by_name or key in self._by_vec


class VarMap(MutableMapping):
    """A simple mapping from VarKey to value, with lookup by canonical
    name string or veckey

    Maintains:
      - _data: dict mapping VarKey to value
      - _by_name: dict mapping str (VarKey.name) to set of VarKeys
      - _by_vec: dict mapping veckey to keys stored in nested list structure
    """

    def __init__(self, *args, **kwargs):
        self._data = {}
        self._varset = VarSet()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        _, val = self.item(key)
        return val

    def item(self, key):
        "get the (varkey, value) pair associated with a (str or key)"
        key = self._varset.resolve(key)
        try:
            return (key, self._data[key])  # single varkey case
        except KeyError as kerr:
            key_arr = self._varset.by_vec(key)
            if key_arr.any():
                return (key, _nested_lookup(key_arr, self._data))
            raise kerr

    @property
    def varset(self):
        "public access to varset. used by SolutionArray.set_necessarylineage"
        return self._varset

    def __setitem__(self, key, value):
        key = self._varset.resolve(key)
        if isinstance(value, Quantity):
            value = value.to(key.units).magnitude
        if is_veckey(key):
            if key not in self._varset._by_vec:
                raise NotImplementedError
            if is_sweepvar(value):
                self._data[key] = value
                return
            if hasattr(value, "__call__"):  # a linked vector-function
                # this case temporarily borrowed from keymap, should refactor
                key.vecfn = value
                value = np.empty(key.shape, dtype="object")
                it = np.nditer(value, flags=["multi_index", "refs_ok"])
                while not it.finished:
                    i = it.multi_index
                    it.iternext()
                    value[i] = veclinkedfn(key.vecfn, i)
            # to setitem via a veckey, the keys must already be registered.
            vks = set(self.varset._by_vec[key].flat)
            if np.prod(key.shape) != len(vks):
                raise ValueError(
                    f"{key} shape is {key.shape}, but _by_vec[{key}] only has "
                    f" {len(vks)} registered keys."
                )  # now we know all vks are present
            if not np.shape(value):
                self.update({vk: value for vk in vks})
                return
            if np.shape(value) == key.shape:
                for vk in vks:
                    self[vk] = np.array(value)[vk.idx]
                return
            raise NotImplementedError
        self._varset.add(key)
        self._data[key] = value

    def register_keys(self, keys):
        "register a set of keys to this mapping, without values yet"
        self._varset.update(keys)

    def __delitem__(self, key):
        key = self._varset.resolve(key)
        if is_veckey(key):
            raise NotImplementedError
        del self._data[key]
        self._varset.discard(key)

    def vector_parent_items(self):
        "like items, but using veckeys and ignoring element keys/items"
        return ((k, self[k]) for k in self._varset.vector_parent_keys())

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        key = getattr(key, "key", key)  # handle Variable case
        if isinstance(key, str) or is_veckey(key):
            return key in self.varset
        return key in self._data

    def update(self, *args, **kwargs):  # pylint: disable=arguments-differ
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def quantity(self, key):
        "Return a quantity corresponding to self[key]"
        clean_key, val = self.item(key)
        return Quantity(val, clean_key.units or "dimensionless")
